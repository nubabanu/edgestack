"""Profile-only portfolio sizing, stress limits, and persistent drawdown state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from edgestack.exceptions import DataError, ValidationError
from edgestack.recommendation.financing import (
    annualized_financing,
    funding_rate_is_stale,
)
from edgestack.recommendation.schemas import (
    AssetKind,
    BaseRecommendationV2,
    ConstraintResultV2,
    DrawdownState,
    EvidenceGrade,
    PortfolioRecommendationV2,
    RecommendationStatus,
    RiskProfileV2,
    RiskStateV2,
    WeightV2,
)
from edgestack.validation.clustered import stationary_bootstrap_indices

MAX_SYSTEM_LEVERAGE = 5.0
MAX_SESSION_INCREASE = 0.25
RESET_MIN_SESSIONS = 20


@dataclass(frozen=True)
class RiskInputsV2:
    session: date
    stressed_forecast_volatility: float
    parametric_995_one_day_loss: float
    historical_995_one_day_loss: float
    bootstrapped_99_path_drawdown: float
    liquidity_position_limits: dict[str, float]
    funding_rate: float
    funding_rate_as_of: date
    monitoring_healthy: bool = True
    stress_acceptable: bool = True
    data_fresh: bool = True

    def __post_init__(self) -> None:
        values = (
            self.stressed_forecast_volatility,
            self.parametric_995_one_day_loss,
            self.historical_995_one_day_loss,
            self.bootstrapped_99_path_drawdown,
            self.funding_rate,
        )
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValidationError("risk inputs must be finite and non-negative")
        if any(
            not np.isfinite(value) or value < 0 for value in self.liquidity_position_limits.values()
        ):
            raise ValidationError("liquidity position limits must be finite and non-negative")


class RiskStateStore:
    """Atomic, monotonic persistence for the default or paper drawdown state."""

    def __init__(self, path: Path):
        self.path = path

    def load(self, *, initial_equity: float) -> RiskStateV2:
        if not self.path.exists():
            return RiskStateV2.initial(initial_equity)
        try:
            return RiskStateV2.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DataError(f"invalid risk state {self.path}: {exc}") from exc

    def save(self, state: RiskStateV2) -> None:
        if self.path.exists():
            previous = self.load(initial_equity=state.current_equity)
            if state.state_version <= previous.state_version:
                raise DataError("risk state version must increase monotonically")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(state.model_dump(mode="json"), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(self.path)


def estimate_risk_inputs(
    portfolio_returns: pd.Series,
    *,
    session: date,
    liquidity_position_limits: dict[str, float],
    funding_rate: float,
    funding_rate_as_of: date,
    n_boot: int = 2_000,
    seed: int = 42,
) -> RiskInputsV2:
    """Estimate deterministic unit-exposure stress inputs from daily returns."""
    values = portfolio_returns.dropna().to_numpy(dtype=float)
    if len(values) < 60:
        raise ValidationError("risk estimation needs at least 60 portfolio sessions")
    daily_volatility = float(values.std(ddof=1))
    short = values[-60:]
    stressed_volatility = max(
        daily_volatility * np.sqrt(252), float(short.std(ddof=1)) * np.sqrt(252) * 1.5
    )
    z_995 = NormalDist().inv_cdf(0.995)
    parametric_loss = max(0.0, z_995 * daily_volatility - float(values.mean()))
    historical_loss = max(0.0, -float(np.quantile(values, 0.005)))
    indices = stationary_bootstrap_indices(
        len(values), n_boot=n_boot, mean_block_length=20, seed=seed
    )[:, :20]
    paths = np.cumprod(1.0 + values[indices], axis=1)
    running_peaks = np.maximum.accumulate(np.column_stack([np.ones(n_boot), paths]), axis=1)
    paths_with_origin = np.column_stack([np.ones(n_boot), paths])
    drawdowns = 1.0 - paths_with_origin / running_peaks
    path_loss = float(np.quantile(drawdowns.max(axis=1), 0.99))
    return RiskInputsV2(
        session=session,
        stressed_forecast_volatility=stressed_volatility,
        parametric_995_one_day_loss=parametric_loss,
        historical_995_one_day_loss=historical_loss,
        bootstrapped_99_path_drawdown=path_loss,
        liquidity_position_limits=liquidity_position_limits,
        funding_rate=funding_rate,
        funding_rate_as_of=funding_rate_as_of,
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return MAX_SYSTEM_LEVERAGE if denominator <= 0 else max(0.0, numerator / denominator)


def _advance_drawdown_state(
    state: RiskStateV2,
    profile: RiskProfileV2,
    inputs: RiskInputsV2,
    *,
    reset_requested: bool,
    data_healthy: bool,
) -> tuple[RiskStateV2, tuple[str, ...]]:
    equity = profile.account_equity
    peak = max(state.peak_equity, equity)
    drawdown = max(0.0, 1.0 - equity / peak)
    half_limit = profile.maximum_drawdown / 2.0
    new_session = state.last_session is None or inputs.session > state.last_session
    warnings: list[str] = []
    latched = state.cash_latched
    sessions = state.sessions_since_latch

    if not latched and drawdown >= profile.maximum_drawdown:
        latched = True
        sessions = 0
    elif latched and new_session:
        sessions += 1

    eligible = (
        latched
        and sessions >= RESET_MIN_SESSIONS
        and drawdown < half_limit
        and inputs.monitoring_healthy
        and inputs.stress_acceptable
        and inputs.data_fresh
        and data_healthy
    )
    if reset_requested:
        if eligible:
            latched = False
            eligible = False
            sessions = 0
        else:
            warnings.append("drawdown reset denied: state is not eligible")

    if latched:
        drawdown_state = DrawdownState.RESET_ELIGIBLE if eligible else DrawdownState.CASH_LATCHED
    elif drawdown >= half_limit:
        drawdown_state = DrawdownState.DELEVERAGING
    else:
        drawdown_state = DrawdownState.NORMAL
    return (
        RiskStateV2.model_validate(
            {
                **state.model_dump(),
                "state_version": state.state_version + 1,
                "peak_equity": peak,
                "current_equity": equity,
                "current_drawdown": drawdown,
                "drawdown_state": drawdown_state,
                "cash_latched": latched,
                "reset_eligible": eligible,
                "sessions_since_latch": sessions,
                "last_session": inputs.session,
            }
        ),
        tuple(warnings),
    )


def _sizing_weights(
    base: BaseRecommendationV2, state: RiskStateV2
) -> tuple[tuple[WeightV2, ...], float, bool]:
    stale = not base.freshness.is_fresh and base.freshness.complete and base.freshness.compatible
    if base.status is not RecommendationStatus.NO_ALLOCATION:
        return base.unlevered_base_weights, MAX_SYSTEM_LEVERAGE, False
    if stale and state.previous_target_weights:
        risky = tuple(
            weight
            for weight in state.previous_target_weights
            if weight.asset_kind is not AssetKind.CASH and abs(weight.weight) > 0
        )
        gross = sum(abs(weight.weight) for weight in risky)
        if gross > 0:
            normalized = tuple(
                weight.model_copy(update={"weight": weight.weight / gross}) for weight in risky
            )
            return normalized, gross, True
    return (), 0.0, stale


def size_recommendation(
    *,
    base: BaseRecommendationV2,
    profile: RiskProfileV2,
    state: RiskStateV2,
    inputs: RiskInputsV2,
    reset_requested: bool = False,
) -> PortfolioRecommendationV2:
    """Apply uniform profile sizing without mutating the recommendation base."""
    next_state, state_warnings = _advance_drawdown_state(
        state,
        profile,
        inputs,
        reset_requested=reset_requested,
        data_healthy=(
            base.freshness.is_fresh and base.freshness.complete and base.freshness.compatible
        ),
    )
    sizing_weights, stale_position_limit, preserving_stale = _sizing_weights(base, state)
    worst_one_day = max(inputs.parametric_995_one_day_loss, inputs.historical_995_one_day_loss)
    remaining_drawdown = max(0.0, profile.maximum_drawdown - next_state.current_drawdown)
    limits: dict[str, float] = {
        "user_cap": profile.maximum_gross_leverage,
        "target_volatility": _safe_ratio(
            profile.target_volatility, inputs.stressed_forecast_volatility
        ),
        "one_day_loss_budget": _safe_ratio(0.25 * profile.maximum_drawdown, worst_one_day),
        "remaining_drawdown_budget": _safe_ratio(
            remaining_drawdown, inputs.bootstrapped_99_path_drawdown
        ),
        "system_ceiling": MAX_SYSTEM_LEVERAGE,
    }

    liquidity_limit = MAX_SYSTEM_LEVERAGE
    stock_limit = MAX_SYSTEM_LEVERAGE
    sector_limit = MAX_SYSTEM_LEVERAGE
    sector_weights: dict[str, float] = {}
    for weight in sizing_weights:
        absolute = abs(weight.weight)
        if absolute <= 0:
            continue
        liquidity_limit = min(
            liquidity_limit,
            inputs.liquidity_position_limits.get(weight.symbol, 0.0) / absolute,
        )
        if weight.asset_kind is AssetKind.STOCK:
            stock_limit = min(stock_limit, profile.per_stock_cap / absolute)
            sector_weights[weight.sector] = sector_weights.get(weight.sector, 0.0) + absolute
    for sector_weight in sector_weights.values():
        sector_limit = min(sector_limit, profile.sector_cap / sector_weight)
    limits["liquidity"] = liquidity_limit
    limits["post_leverage_stock_cap"] = stock_limit
    limits["post_leverage_sector_cap"] = sector_limit

    if next_state.cash_latched:
        drawdown_limit = 0.0
    elif next_state.drawdown_state is DrawdownState.DELEVERAGING:
        fraction = max(
            0.0,
            2.0
            * (profile.maximum_drawdown - next_state.current_drawdown)
            / profile.maximum_drawdown,
        )
        drawdown_limit = MAX_SYSTEM_LEVERAGE * fraction
    else:
        drawdown_limit = MAX_SYSTEM_LEVERAGE
    limits["drawdown_state"] = drawdown_limit
    limits["market_freshness"] = stale_position_limit

    funding_stale = funding_rate_is_stale(inputs.funding_rate_as_of, inputs.session)
    limits["funding_freshness"] = 1.0 if funding_stale else MAX_SYSTEM_LEVERAGE
    if not sizing_weights or not inputs.stress_acceptable or not inputs.monitoring_healthy:
        limits["allocation_health"] = 0.0
    else:
        limits["allocation_health"] = MAX_SYSTEM_LEVERAGE

    requested = max(0.0, min(limits.values()))
    previous = state.previous_effective_leverage
    may_increase = state.last_session is None or inputs.session > state.last_session
    ramp_limit = previous + MAX_SESSION_INCREASE if may_increase else previous
    effective = min(requested, ramp_limit) if requested > previous else requested
    effective = min(MAX_SYSTEM_LEVERAGE, max(0.0, effective))
    limits["session_increase"] = ramp_limit
    pre_ramp_binding = {name for name, limit in limits.items() if abs(limit - requested) < 1e-9}
    if effective < requested - 1e-9:
        binding = {"session_increase"}
    else:
        binding = pre_ramp_binding - {"session_increase"}

    targets = tuple(
        weight.model_copy(update={"weight": weight.weight * effective}) for weight in sizing_weights
    )
    targets += (
        WeightV2(
            symbol="CASH",
            weight=1.0 - effective,
            asset_kind=AssetKind.CASH,
            sector="cash",
        ),
    )
    next_state = RiskStateV2.model_validate(
        {
            **next_state.model_dump(),
            "previous_effective_leverage": effective,
            "previous_target_weights": targets,
        }
    )
    _cash_income, funding_cost, _financing_net = annualized_financing(
        effective, inputs.funding_rate, profile.funding_spread_bps
    )
    evidence = (
        EvidenceGrade.PROMOTED
        if base.status is RecommendationStatus.ACTIVE
        else EvidenceGrade.WATCHLIST
        if base.status is RecommendationStatus.WATCHLIST_ONLY
        else EvidenceGrade.POLICY
        if base.status is RecommendationStatus.BASELINE_ONLY
        else EvidenceGrade.INSUFFICIENT
    )
    hard_no_allocation = not sizing_weights or next_state.cash_latched
    status = RecommendationStatus.NO_ALLOCATION if hard_no_allocation else base.status
    warnings = list(base.warnings) + list(state_warnings)
    if funding_stale:
        warnings.append("DGS3MO is over five US business days stale; increases above 1x disabled")
    if preserving_stale:
        warnings.append("stale market data: preserving or reducing prior positions only")
    constraints = tuple(
        ConstraintResultV2(
            name=name,
            leverage_limit=max(0.0, min(MAX_SYSTEM_LEVERAGE, limit)),
            binding=name in binding,
        )
        for name, limit in limits.items()
    )
    return PortfolioRecommendationV2(
        status=status,
        as_of=base.as_of,
        execution_at=base.execution_at,
        artifact_version=base.artifact_version,
        data_version=base.data_version,
        policy_version=base.policy_version,
        risk_profile_hash=profile.profile_hash,
        input_risk_state_version=state.state_version,
        output_risk_state=next_state,
        baseline_weights=base.baseline_weights,
        base_recommendation_weights=base.unlevered_base_weights,
        personalized_target_weights=targets,
        effective_leverage=effective,
        constraints=constraints,
        binding_constraints=tuple(sorted(binding)),
        expected_net_return=base.expected_net_return * effective - funding_cost,
        expected_volatility=inputs.stressed_forecast_volatility * effective,
        funding_cost=funding_cost,
        turnover=sum(
            abs(
                target.weight
                - next(
                    (
                        old.weight
                        for old in state.previous_target_weights
                        if old.symbol == target.symbol
                    ),
                    0.0,
                )
            )
            for target in targets
        ),
        one_day_stress_loss=worst_one_day * effective,
        multi_session_stress_loss=inputs.bootstrapped_99_path_drawdown * effective,
        evidence_grade=evidence,
        freshness=base.freshness,
        warnings=tuple(warnings),
        compatibility_metadata={
            "sizing_only": "true",
            "funding_series": "DGS3MO",
            "funding_rate_as_of": inputs.funding_rate_as_of.isoformat(),
        },
    )
