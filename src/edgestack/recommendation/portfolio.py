"""Profile-independent expected returns, covariance, and portfolio optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from edgestack.exceptions import ValidationError
from edgestack.recommendation.hashing import stable_hash
from edgestack.recommendation.schemas import (
    AssetKind,
    BaselinePolicyV2,
    BaseRecommendationV2,
    FreshnessV2,
    RecommendationStatus,
    SleeveContributionV2,
    WatchlistEntryV2,
    WeightV2,
)

TRADING_SESSIONS = 252
ESTIMATION_WINDOWS = (60, 126, 252)
WINDOW_WEIGHTS = (0.5, 0.3, 0.2)


@dataclass(frozen=True)
class ExpectedReturnEstimate:
    annualized_mean: float
    annualized_standard_error: float
    effective_sample_size: float
    shrinkage_prior_n: float = 100.0

    @property
    def shrinkage(self) -> float:
        n = max(0.0, self.effective_sample_size)
        return n / (n + self.shrinkage_prior_n)

    @property
    def shrunk_mean(self) -> float:
        return self.annualized_mean * self.shrinkage

    @property
    def lower_95(self) -> float:
        return self.shrunk_mean - 1.96 * self.annualized_standard_error


@dataclass(frozen=True)
class AssetMetadata:
    symbol: str
    asset_kind: AssetKind
    sector: str
    median_adv: float
    broad_policy_asset: bool = False
    half_spread_bps: float = 2.5
    impact_coefficient: float = 0.10


@dataclass(frozen=True)
class CovarianceEstimate:
    symbols: tuple[str, ...]
    annualized: np.ndarray
    stressed_annualized: np.ndarray
    version: str


@dataclass(frozen=True)
class OptimizerResult:
    weights: dict[str, float]
    expected_net_return: float
    expected_volatility: float
    turnover: float
    marginal_utilities: dict[str, float]
    success: bool
    reason: str = ""


def estimate_covariance(returns: pd.DataFrame) -> CovarianceEstimate:
    """Blend 60/126/252-session Ledoit-Wolf estimates and stress correlation."""
    clean = returns.astype(float).replace([np.inf, -np.inf], np.nan)
    symbols = tuple(str(c) for c in clean.columns)
    if not symbols or len(clean) < 20:
        raise ValidationError("covariance needs at least one asset and 20 sessions")
    estimates: list[np.ndarray] = []
    weights: list[float] = []
    for window, weight in zip(ESTIMATION_WINDOWS, WINDOW_WEIGHTS, strict=True):
        sample = clean.tail(window).dropna(how="any")
        if len(sample) < 20:
            continue
        estimates.append(LedoitWolf().fit(sample.to_numpy()).covariance_ * TRADING_SESSIONS)
        weights.append(weight)
    if not estimates:
        raise ValidationError("insufficient aligned returns for shrinkage covariance")
    normalized = np.asarray(weights) / sum(weights)
    covariance = sum(
        weight * estimate for weight, estimate in zip(normalized, estimates, strict=True)
    )
    covariance = _nearest_psd(covariance)

    short = estimates[0]
    base_vol = np.sqrt(np.maximum(np.diag(covariance), 0))
    short_vol = np.sqrt(np.maximum(np.diag(short), 0))
    stressed_vol = np.maximum(base_vol, 1.5 * short_vol)
    base_corr = _covariance_to_correlation(covariance)
    stressed_corr = np.where(np.eye(len(symbols), dtype=bool), 1.0, np.maximum(base_corr, 0.75))
    stressed_corr = _nearest_correlation(stressed_corr)
    stressed = stressed_corr * np.outer(stressed_vol, stressed_vol)
    version = stable_hash(
        {
            "symbols": symbols,
            "windows": ESTIMATION_WINDOWS,
            "weights": tuple(float(x) for x in normalized),
            "covariance": covariance.round(12).tolist(),
            "stress": "max(base,1.5x_short),corr_floor_0.75",
        }
    )[:16]
    return CovarianceEstimate(symbols, covariance, stressed, version)


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(values, 1e-12)
    return (vectors * clipped) @ vectors.T


def _covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.maximum(np.diag(covariance), 1e-18))
    corr = covariance / np.outer(vol, vol)
    return np.clip(corr, -1.0, 1.0)


def _nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    psd = _nearest_psd(matrix)
    scale = np.sqrt(np.maximum(np.diag(psd), 1e-18))
    correlation = psd / np.outer(scale, scale)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def optimize_base_portfolio(
    *,
    baseline_weights: dict[str, float],
    expected_returns_lower_95: dict[str, float],
    covariance: CovarianceEstimate,
    metadata: dict[str, AssetMetadata],
    account_equity_reference: float = 100_000.0,
    previous_weights: dict[str, float] | None = None,
    max_stock_weight: float = 0.03,
    max_sector_weight: float = 0.20,
    risk_aversion: float = 3.0,
    turnover_penalty: float = 0.002,
) -> OptimizerResult:
    symbols = covariance.symbols
    if set(symbols) != set(metadata):
        raise ValidationError("covariance and asset metadata symbols must match")
    baseline = np.array([baseline_weights.get(s, 0.0) for s in symbols], dtype=float)
    if abs(baseline.sum() - 1.0) > 1e-8:
        raise ValidationError("baseline weights must sum to one over optimizer assets")
    expected = np.array([expected_returns_lower_95.get(s, 0.0) for s in symbols], dtype=float)
    if not any(value > 0 for value in expected):
        volatility = float(np.sqrt(max(0.0, baseline @ covariance.annualized @ baseline)))
        return OptimizerResult(
            weights=dict(zip(symbols, baseline, strict=True)),
            expected_net_return=0.0,
            expected_volatility=volatility,
            turnover=0.0,
            marginal_utilities={s: float(expected[i]) for i, s in enumerate(symbols)},
            success=True,
            reason="no positive lower-confidence active return; baseline preserved",
        )
    prior_map = previous_weights or baseline_weights
    prior = np.array([prior_map.get(s, 0.0) for s in symbols], dtype=float)
    bounds = []
    for symbol in symbols:
        item = metadata[symbol]
        position_capacity = 0.10 * max(0.0, item.median_adv) * 5.0 / account_equity_reference
        if item.asset_kind is AssetKind.STOCK:
            cap = min(max_stock_weight, position_capacity)
        else:
            cap = min(1.0, position_capacity) if item.median_adv > 0 else 0.0
        bounds.append((0.0, max(0.0, cap)))

    constraints: list[dict] = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    sectors = sorted(
        {item.sector for item in metadata.values() if item.asset_kind is AssetKind.STOCK}
    )
    for sector in sectors:
        indices = np.array(
            [i for i, s in enumerate(symbols) if metadata[s].sector == sector], dtype=int
        )
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w, idx=indices: float(max_sector_weight - w[idx].sum()),
            }
        )

    def objective(weights: np.ndarray) -> float:
        active_return = float(expected @ weights)
        risk = float(weights @ covariance.stressed_annualized @ weights)
        trades = np.sqrt((weights - prior) ** 2 + 1e-12)
        turnover = float(trades.sum())
        spread = sum(
            float(trades[i]) * metadata[symbol].half_spread_bps / 10_000
            for i, symbol in enumerate(symbols)
        )
        impact = sum(
            metadata[symbol].impact_coefficient
            * float(trades[i]) ** 2
            * account_equity_reference
            / max(metadata[symbol].median_adv, 1.0)
            for i, symbol in enumerate(symbols)
        )
        concentration = 0.01 * float(np.square(weights).sum())
        utility = active_return - 0.5 * risk_aversion * risk
        return -(utility - turnover_penalty * turnover - spread - impact - concentration)

    result = minimize(
        objective,
        baseline,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not result.success:
        return OptimizerResult(
            weights=dict(zip(symbols, baseline, strict=True)),
            expected_net_return=0.0,
            expected_volatility=float(
                np.sqrt(max(0.0, baseline @ covariance.annualized @ baseline))
            ),
            turnover=0.0,
            marginal_utilities={},
            success=False,
            reason=str(result.message),
        )
    weights = np.clip(result.x, 0.0, None)
    weights /= weights.sum()
    expected_net = float(expected @ weights)
    volatility = float(np.sqrt(max(0.0, weights @ covariance.annualized @ weights)))
    turnover = float(np.abs(weights - prior).sum())
    gradient = expected - risk_aversion * covariance.stressed_annualized @ weights
    return OptimizerResult(
        weights=dict(zip(symbols, (float(x) for x in weights), strict=True)),
        expected_net_return=expected_net,
        expected_volatility=volatility,
        turnover=turnover,
        marginal_utilities=dict(zip(symbols, (float(x) for x in gradient), strict=True)),
        success=True,
    )


def build_base_recommendation(
    *,
    policy: BaselinePolicyV2,
    promoted_sleeves: tuple[SleeveContributionV2, ...],
    watchlist: tuple[WatchlistEntryV2, ...],
    returns: pd.DataFrame,
    metadata: dict[str, AssetMetadata],
    expected_estimates: dict[str, ExpectedReturnEstimate],
    freshness: FreshnessV2,
    as_of: pd.Timestamp,
    execution_at: pd.Timestamp,
    data_version: str,
    artifact_version: str,
) -> BaseRecommendationV2:
    baseline = {weight.symbol: weight.weight for weight in policy.weights}
    all_symbols = tuple(str(c) for c in returns.columns)
    baseline_full = {symbol: baseline.get(symbol, 0.0) for symbol in all_symbols}
    hard_failure = not freshness.is_fresh or not freshness.complete or not freshness.compatible
    invalid_baseline = (
        not set(baseline).issubset(all_symbols)
        or not set(all_symbols).issubset(metadata)
        or len(returns) < 20
        or returns[list(baseline)].tail(20).isna().any().any()
    )
    if hard_failure or invalid_baseline:
        reason = (
            "hard freshness or compatibility constraint"
            if hard_failure
            else "invalid or incomplete baseline data"
        )
        policy_by_symbol = {weight.symbol: weight for weight in policy.weights}
        zero_symbols = tuple(dict.fromkeys((*baseline, *all_symbols)))
        zeros = tuple(
            WeightV2(
                symbol=symbol,
                weight=0.0,
                asset_kind=(
                    metadata[symbol].asset_kind
                    if symbol in metadata
                    else policy_by_symbol[symbol].asset_kind
                ),
                sector=(metadata[symbol].sector if symbol in metadata else "unknown"),
            )
            for symbol in zero_symbols
        )
        return BaseRecommendationV2(
            status=RecommendationStatus.NO_ALLOCATION,
            as_of=as_of.to_pydatetime(),
            execution_at=execution_at.to_pydatetime(),
            artifact_version=artifact_version,
            data_version=data_version,
            policy_version=policy.policy_version,
            baseline_weights=policy.weights,
            unlevered_base_weights=zeros,
            covariance_version="unavailable",
            watchlist=watchlist,
            freshness=freshness,
            warnings=(f"{reason}: no new allocation",),
        )
    covariance = estimate_covariance(returns.loc[:, list(all_symbols)])
    promoted_symbols = {
        weight.symbol
        for sleeve in promoted_sleeves
        for weight in sleeve.symbol_weights
        if weight.weight != 0
    }
    lower = {
        symbol: estimate.lower_95
        for symbol, estimate in expected_estimates.items()
        if symbol in promoted_symbols
    }
    optimized = optimize_base_portfolio(
        baseline_weights=baseline_full,
        expected_returns_lower_95=lower,
        covariance=covariance,
        metadata=metadata,
    )
    weights = tuple(
        WeightV2(
            symbol=symbol,
            weight=optimized.weights[symbol],
            asset_kind=metadata[symbol].asset_kind,
            sector=metadata[symbol].sector,
        )
        for symbol in all_symbols
    )
    active = bool(promoted_sleeves) and any(
        abs(optimized.weights[symbol] - baseline_full[symbol]) > 1e-8 for symbol in all_symbols
    )
    status = (
        RecommendationStatus.ACTIVE
        if active
        else RecommendationStatus.WATCHLIST_ONLY
        if watchlist
        else RecommendationStatus.BASELINE_ONLY
    )
    compound = tuple(sleeve for sleeve in promoted_sleeves if sleeve.compound)
    standalone = tuple(sleeve for sleeve in promoted_sleeves if not sleeve.compound)
    warnings = () if optimized.success else (f"optimizer fallback: {optimized.reason}",)
    return BaseRecommendationV2(
        status=status,
        as_of=as_of.to_pydatetime(),
        execution_at=execution_at.to_pydatetime(),
        artifact_version=artifact_version,
        data_version=data_version,
        policy_version=policy.policy_version,
        baseline_weights=policy.weights,
        promoted_sleeves=standalone,
        promoted_compound_sleeves=compound,
        unlevered_base_weights=weights,
        expected_returns={
            symbol: estimate.shrunk_mean for symbol, estimate in expected_estimates.items()
        },
        expected_net_return=optimized.expected_net_return,
        expected_volatility=optimized.expected_volatility,
        covariance_version=covariance.version,
        turnover_estimate=optimized.turnover,
        evidence_grades={s.sleeve_id: s.evidence_grade for s in promoted_sleeves},
        watchlist=watchlist,
        freshness=freshness,
        warnings=warnings,
    )
