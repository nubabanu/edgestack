"""Canonical target-weight paper execution and cash-flow tests."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from edgestack.config import EdgeStackConfig
from edgestack.execution.fills import Bar
from edgestack.paper.broker import get_broker
from edgestack.paper.canonical import (
    CanonicalPaperStateV2,
    CorporateActionInputV2,
    PaperPositionV2,
    execute_paper_session,
    queue_recommendation_target,
)
from edgestack.recommendation.policy import load_baseline_policy
from edgestack.recommendation.risk import RiskInputsV2, size_recommendation
from edgestack.recommendation.schemas import (
    BaseRecommendationV2,
    FreshnessV2,
    RecommendationStatus,
    RiskProfileV2,
    RiskStateV2,
)

SIGNAL_SESSION = date(2026, 7, 15)
EXECUTION_SESSION = date(2026, 7, 16)


def _recommendation():
    policy = load_baseline_policy()
    as_of = datetime(2026, 7, 15, 20, tzinfo=UTC)
    freshness = FreshnessV2(
        as_of=as_of,
        expected_session=SIGNAL_SESSION,
        is_fresh=True,
        age_business_days=0,
        complete=True,
        compatible=True,
    )
    base = BaseRecommendationV2(
        status=RecommendationStatus.BASELINE_ONLY,
        as_of=as_of,
        execution_at=datetime(2026, 7, 16, 13, 30, tzinfo=UTC),
        artifact_version="artifact",
        data_version="data",
        policy_version=policy.policy_version,
        baseline_weights=policy.weights,
        unlevered_base_weights=policy.weights,
        covariance_version="cov",
        expected_volatility=0.08,
        freshness=freshness,
    )
    inputs = RiskInputsV2(
        session=SIGNAL_SESSION,
        stressed_forecast_volatility=0.10,
        parametric_995_one_day_loss=0.02,
        historical_995_one_day_loss=0.02,
        bootstrapped_99_path_drawdown=0.08,
        liquidity_position_limits={weight.symbol: 5 for weight in policy.weights},
        funding_rate=0.04,
        funding_rate_as_of=SIGNAL_SESSION,
    )
    profile = RiskProfileV2()
    risk_state = RiskStateV2(
        previous_effective_leverage=1,
        peak_equity=100_000,
        current_equity=100_000,
    )
    return size_recommendation(
        base=base,
        profile=profile,
        state=risk_state,
        inputs=inputs,
    )


def _bars(*, volume: float = 1_000_000, close: float = 101) -> dict[str, Bar]:
    return {
        symbol: Bar(
            session=pd.Timestamp(EXECUTION_SESSION),
            open=100,
            high=max(102, close),
            low=99,
            close=close,
            volume=volume,
        )
        for symbol in ("SPY", "TLT", "SHY", "GLD")
    }


def test_target_orders_fill_at_next_open_and_returns_use_actual_fills() -> None:
    recommendation = _recommendation()
    state = queue_recommendation_target(
        CanonicalPaperStateV2.initial(100_000, recommendation.output_risk_state),
        recommendation,
    )
    updated = execute_paper_session(
        state,
        session=EXECUTION_SESSION,
        bars=_bars(),
        corporate_actions=(),
        broker=get_broker(EdgeStackConfig()),
        base_funding_rate=0.04,
        funding_spread_bps=200,
    )

    assert len(updated.positions) == 4
    assert len(updated.order_intents) == 4
    assert len(updated.fills) == 4
    assert updated.open_orders == ()
    assert updated.pending_target is None
    assert all(fill.price == 100 for fill in updated.fills)
    assert updated.cumulative_transaction_costs > 0
    assert updated.realized_returns[-1].actual_fill_return == pytest.approx(
        updated.current_equity / 100_000 - 1
    )


def test_participation_cap_preserves_partial_orders_for_later_sessions() -> None:
    recommendation = _recommendation()
    state = queue_recommendation_target(
        CanonicalPaperStateV2.initial(100_000, recommendation.output_risk_state),
        recommendation,
    )
    updated = execute_paper_session(
        state,
        session=EXECUTION_SESSION,
        bars=_bars(volume=100),
        corporate_actions=(),
        broker=get_broker(EdgeStackConfig()),
        base_funding_rate=0.04,
        funding_spread_bps=200,
    )

    assert len(updated.open_orders) == 4
    assert all(order.status == "PARTIAL" for order in updated.open_orders)
    assert all(fill.quantity == 10 for fill in updated.fills)


def test_missing_bar_delays_whole_target_without_inventing_a_fill() -> None:
    recommendation = _recommendation()
    state = queue_recommendation_target(
        CanonicalPaperStateV2.initial(100_000, recommendation.output_risk_state),
        recommendation,
    )
    bars = _bars()
    bars.pop("GLD")
    updated = execute_paper_session(
        state,
        session=EXECUTION_SESSION,
        bars=bars,
        corporate_actions=(),
        broker=get_broker(EdgeStackConfig()),
        base_funding_rate=0.04,
        funding_spread_bps=200,
    )

    assert updated.pending_target == state.pending_target
    assert updated.fills == ()
    assert updated.positions == ()


def test_splits_and_dividends_adjust_positions_and_cash_explicitly() -> None:
    risk_state = RiskStateV2.initial(1_000)
    state = CanonicalPaperStateV2(
        last_session=date(2026, 7, 13),
        cash=0,
        current_equity=1_000,
        positions=(
            PaperPositionV2(symbol="AAA", quantity=10, average_fill_price=100, last_price=100),
        ),
        risk_state=risk_state,
    )
    bar = Bar(
        session=pd.Timestamp(EXECUTION_SESSION),
        open=50,
        high=51,
        low=49,
        close=50,
        volume=1_000_000,
    )
    updated = execute_paper_session(
        state,
        session=EXECUTION_SESSION,
        bars={"AAA": bar},
        corporate_actions=(
            CorporateActionInputV2(
                symbol="AAA", session=date(2026, 7, 14), action_type="split", value=2
            ),
            CorporateActionInputV2(
                symbol="AAA", session=date(2026, 7, 15), action_type="dividend", value=1
            ),
        ),
        broker=get_broker(EdgeStackConfig()),
        base_funding_rate=0.04,
        funding_spread_bps=200,
    )

    assert updated.positions[0].quantity == 20
    assert updated.positions[0].average_fill_price == 50
    assert updated.cash == 20
    assert updated.cumulative_dividends == 20
    assert updated.current_equity == 1_020


def test_negative_cash_is_charged_base_rate_plus_spread_over_calendar_gap() -> None:
    state = CanonicalPaperStateV2(
        last_session=date(2026, 7, 13),
        cash=-100,
        current_equity=1_000,
        positions=(
            PaperPositionV2(symbol="AAA", quantity=11, average_fill_price=100, last_price=100),
        ),
        risk_state=RiskStateV2.initial(1_000),
    )
    bar = Bar(
        session=pd.Timestamp(EXECUTION_SESSION),
        open=100,
        high=101,
        low=99,
        close=100,
        volume=1_000_000,
    )
    updated = execute_paper_session(
        state,
        session=EXECUTION_SESSION,
        bars={"AAA": bar},
        corporate_actions=(),
        broker=get_broker(EdgeStackConfig()),
        base_funding_rate=0.04,
        funding_spread_bps=200,
    )

    expected = -100 * 0.06 * 3 / 365
    assert updated.realized_returns[-1].financing_cash_flow == pytest.approx(expected)
    assert updated.current_equity == pytest.approx(1_000 + expected)
