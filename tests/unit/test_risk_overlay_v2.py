"""Risk sizing, financing, drawdown, and persistence acceptance tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from edgestack.exceptions import DataError
from edgestack.recommendation.financing import (
    accrue_cash_financing,
    annualized_financing,
    funding_rate_is_stale,
)
from edgestack.recommendation.policy import load_baseline_policy
from edgestack.recommendation.risk import (
    RiskInputsV2,
    RiskStateStore,
    estimate_risk_inputs,
    size_recommendation,
)
from edgestack.recommendation.schemas import (
    AssetKind,
    BaseRecommendationV2,
    DrawdownState,
    FreshnessV2,
    RecommendationStatus,
    RiskProfileV2,
    RiskStateV2,
    WeightV2,
)

SESSION = date(2026, 7, 16)


def _freshness(*, fresh: bool = True) -> FreshnessV2:
    return FreshnessV2(
        as_of=datetime(2026, 7, 16, 20, tzinfo=UTC),
        expected_session=SESSION,
        is_fresh=fresh,
        age_business_days=0 if fresh else 2,
        complete=True,
        compatible=True,
    )


def _base(
    weights: tuple[WeightV2, ...] | None = None,
    *,
    status: RecommendationStatus = RecommendationStatus.BASELINE_ONLY,
    fresh: bool = True,
) -> BaseRecommendationV2:
    policy = load_baseline_policy()
    selected = weights or policy.weights
    if status is RecommendationStatus.NO_ALLOCATION:
        selected = tuple(weight.model_copy(update={"weight": 0.0}) for weight in selected)
    return BaseRecommendationV2(
        status=status,
        as_of=datetime(2026, 7, 16, 20, tzinfo=UTC),
        execution_at=datetime(2026, 7, 17, 13, 30, tzinfo=UTC),
        artifact_version="artifact-v2",
        data_version="data-v2",
        policy_version=policy.policy_version,
        baseline_weights=policy.weights,
        unlevered_base_weights=selected,
        expected_net_return=0.04,
        expected_volatility=0.08,
        covariance_version="cov-v2",
        freshness=_freshness(fresh=fresh),
    )


def _inputs(
    *,
    session: date = SESSION,
    funding_as_of: date = SESSION,
    limits: dict[str, float] | None = None,
) -> RiskInputsV2:
    return RiskInputsV2(
        session=session,
        stressed_forecast_volatility=0.02,
        parametric_995_one_day_loss=0.004,
        historical_995_one_day_loss=0.005,
        bootstrapped_99_path_drawdown=0.02,
        liquidity_position_limits=limits or {"SPY": 10.0, "TLT": 10.0, "SHY": 10.0, "GLD": 10.0},
        funding_rate=0.04,
        funding_rate_as_of=funding_as_of,
    )


def _state(*, leverage: float = 5.0) -> RiskStateV2:
    return RiskStateV2(
        previous_effective_leverage=leverage,
        peak_equity=100_000,
        current_equity=100_000,
    )


def _risky_weights(recommendation) -> dict[str, float]:
    return {
        weight.symbol: weight.weight
        for weight in recommendation.personalized_target_weights
        if weight.asset_kind is not AssetKind.CASH
    }


def test_profiles_change_only_uniform_sizing_and_not_base_or_expected_alpha() -> None:
    base = _base()
    conservative = size_recommendation(
        base=base,
        profile=RiskProfileV2(maximum_gross_leverage=1.0),
        state=_state(),
        inputs=_inputs(),
    )
    aggressive = size_recommendation(
        base=base,
        profile=RiskProfileV2(maximum_gross_leverage=3.0),
        state=_state(),
        inputs=_inputs(),
    )

    assert conservative.base_recommendation_weights == aggressive.base_recommendation_weights
    assert conservative.effective_leverage == 1.0
    assert aggressive.effective_leverage == 3.0
    normalized = {
        symbol: weight / aggressive.effective_leverage
        for symbol, weight in _risky_weights(aggressive).items()
    }
    assert normalized == {weight.symbol: weight.weight for weight in base.unlevered_base_weights}
    assert base.expected_net_return == 0.04


@settings(max_examples=30, deadline=None)
@given(
    user_cap=st.floats(min_value=0, max_value=5, allow_nan=False, allow_infinity=False),
    stock_cap=st.floats(min_value=0.005, max_value=0.10, allow_nan=False),
    sector_cap=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
)
def test_leverage_and_post_leverage_concentration_property(
    user_cap: float, stock_cap: float, sector_cap: float
) -> None:
    weights = (
        WeightV2(symbol="SPY", weight=0.94, asset_kind=AssetKind.ETF, sector="broad"),
        WeightV2(symbol="AAA", weight=0.03, asset_kind=AssetKind.STOCK, sector="technology"),
        WeightV2(symbol="BBB", weight=0.03, asset_kind=AssetKind.STOCK, sector="technology"),
    )
    recommendation = size_recommendation(
        base=_base(weights),
        profile=RiskProfileV2(
            maximum_gross_leverage=user_cap,
            per_stock_cap=stock_cap,
            sector_cap=sector_cap,
        ),
        state=_state(),
        inputs=_inputs(limits={"SPY": 10.0, "AAA": 10.0, "BBB": 10.0}),
    )
    targets = _risky_weights(recommendation)

    assert recommendation.effective_leverage <= min(user_cap, 5.0) + 1e-10
    assert targets["AAA"] <= stock_cap + 1e-10
    assert targets["BBB"] <= stock_cap + 1e-10
    assert targets["AAA"] + targets["BBB"] <= sector_cap + 1e-10


def test_leverage_increases_by_quarter_per_session_and_reduces_immediately() -> None:
    base = _base()
    profile = RiskProfileV2(maximum_gross_leverage=3.0)
    first = size_recommendation(
        base=base, profile=profile, state=RiskStateV2.initial(100_000), inputs=_inputs()
    )
    repeated = size_recommendation(
        base=base, profile=profile, state=first.output_risk_state, inputs=_inputs()
    )
    second = size_recommendation(
        base=base,
        profile=profile,
        state=first.output_risk_state,
        inputs=_inputs(session=SESSION + timedelta(days=1)),
    )
    reduced = size_recommendation(
        base=base,
        profile=RiskProfileV2(maximum_gross_leverage=0.10),
        state=second.output_risk_state,
        inputs=_inputs(session=SESSION + timedelta(days=2)),
    )

    assert first.effective_leverage == 0.25
    assert repeated.effective_leverage == 0.25
    assert second.effective_leverage == 0.50
    assert reduced.effective_leverage == 0.10


def test_quarter_kelly_is_an_independent_binding_leverage_constraint() -> None:
    recommendation = size_recommendation(
        base=_base(),
        profile=RiskProfileV2(maximum_gross_leverage=5.0),
        state=_state(),
        inputs=replace(_inputs(), quarter_kelly_leverage_limit=0.40),
    )

    assert recommendation.effective_leverage == 0.40
    assert recommendation.binding_constraints == ("quarter_kelly",)


def test_stale_data_preserves_or_reduces_prior_positions_and_cannot_add_names() -> None:
    previous = (
        WeightV2(symbol="SPY", weight=0.50, asset_kind=AssetKind.ETF, sector="broad"),
        WeightV2(symbol="CASH", weight=0.50, asset_kind=AssetKind.CASH, sector="cash"),
    )
    state = RiskStateV2(
        previous_effective_leverage=0.50,
        peak_equity=100_000,
        current_equity=100_000,
        previous_target_weights=previous,
    )
    stale = _base(status=RecommendationStatus.NO_ALLOCATION, fresh=False)
    recommendation = size_recommendation(
        base=stale,
        profile=RiskProfileV2(maximum_gross_leverage=5.0),
        state=state,
        inputs=_inputs(limits={"SPY": 10.0}),
    )

    assert recommendation.effective_leverage == 0.50
    assert _risky_weights(recommendation) == {"SPY": 0.50}
    assert "stale market data" in " ".join(recommendation.warnings)


def test_funding_staleness_caps_leverage_and_higher_spread_reduces_net_return() -> None:
    base = _base()
    low_spread = size_recommendation(
        base=base,
        profile=RiskProfileV2(maximum_gross_leverage=3.0, funding_spread_bps=200),
        state=_state(),
        inputs=_inputs(),
    )
    high_spread = size_recommendation(
        base=base,
        profile=RiskProfileV2(maximum_gross_leverage=3.0, funding_spread_bps=800),
        state=_state(),
        inputs=_inputs(),
    )
    stale = size_recommendation(
        base=base,
        profile=RiskProfileV2(maximum_gross_leverage=3.0),
        state=_state(),
        inputs=_inputs(funding_as_of=SESSION - timedelta(days=14)),
    )

    assert high_spread.effective_leverage == low_spread.effective_leverage == 3.0
    assert high_spread.funding_cost > low_spread.funding_cost
    assert high_spread.expected_net_return < low_spread.expected_net_return
    assert stale.effective_leverage == 1.0
    assert funding_rate_is_stale(SESSION - timedelta(days=14), SESSION)


def test_drawdown_latch_survives_recovery_until_eligible_and_explicitly_reset() -> None:
    base = _base()
    breached = size_recommendation(
        base=base,
        profile=RiskProfileV2(account_equity=80_000),
        state=RiskStateV2.initial(100_000),
        inputs=_inputs(),
    )
    assert breached.status is RecommendationStatus.NO_ALLOCATION
    assert breached.output_risk_state.drawdown_state is DrawdownState.CASH_LATCHED

    state = breached.output_risk_state
    recovered = RiskProfileV2(account_equity=95_000)
    for offset in range(1, 21):
        result = size_recommendation(
            base=base,
            profile=recovered,
            state=state,
            inputs=_inputs(session=SESSION + timedelta(days=offset)),
        )
        state = result.output_risk_state
    assert state.drawdown_state is DrawdownState.RESET_ELIGIBLE
    assert result.effective_leverage == 0

    reset = size_recommendation(
        base=base,
        profile=recovered,
        state=state,
        inputs=_inputs(session=SESSION + timedelta(days=21)),
        reset_requested=True,
    )
    assert reset.output_risk_state.drawdown_state is DrawdownState.NORMAL
    assert not reset.output_risk_state.cash_latched
    assert reset.effective_leverage == 0.25


def test_financing_uses_actual_calendar_days_and_positive_cash_earns_base_rate() -> None:
    assert accrue_cash_financing(
        1_000,
        base_rate=0.0365,
        funding_spread_bps=200,
        start=date(2026, 7, 3),
        end=date(2026, 7, 6),
    ) == pytest.approx(0.30)
    assert accrue_cash_financing(
        -1_000,
        base_rate=0.0365,
        funding_spread_bps=200,
        start=date(2026, 7, 3),
        end=date(2026, 7, 6),
    ) == pytest.approx(-1_000 * 0.0565 * 3 / 365)
    cash_income, funding_cost, net = annualized_financing(0.5, 0.04, 200)
    assert (cash_income, funding_cost, net) == (0.02, 0.0, 0.02)


def test_stress_estimation_and_risk_state_store_are_deterministic(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0002, 0.01, 300))
    first = estimate_risk_inputs(
        returns,
        session=SESSION,
        liquidity_position_limits={"SPY": 2.0},
        funding_rate=0.04,
        funding_rate_as_of=SESSION,
        n_boot=100,
    )
    second = estimate_risk_inputs(
        returns,
        session=SESSION,
        liquidity_position_limits={"SPY": 2.0},
        funding_rate=0.04,
        funding_rate_as_of=SESSION,
        n_boot=100,
    )
    assert first == second

    store = RiskStateStore(tmp_path / "risk" / "state.json")
    state = RiskStateV2(
        state_version=1,
        peak_equity=100_000,
        current_equity=100_000,
    )
    store.save(state)
    assert store.load(initial_equity=1).state_version == 1
    with pytest.raises(DataError, match="monotonically"):
        store.save(state)
