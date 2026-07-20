from __future__ import annotations

import numpy as np
import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.execution.costs import CostModel
from edgestack.execution.fills import Bar
from edgestack.execution.orders import Order, OrderStatus, OrderType
from edgestack.paper.broker import SimulatedBroker, get_broker
from edgestack.paper.shadow import ShadowBook, shadow_books_are_isolated
from edgestack.recommendation.growth import (
    annualized_log_growth,
    log_growth_superiority,
    shrinkage_quarter_kelly_limit,
)
from edgestack.recommendation.inverse_etf import account_inverse_etf
from edgestack.research.evaluate import (
    _daily_strategy_returns,
    _event_candidate_returns,
    daily_position_for_next_open,
)


def test_waiting_days_and_opportunity_cost_are_in_the_growth_test() -> None:
    index = pd.bdate_range("2020-01-01", periods=500)
    strategy = pd.Series(0.0, index=index)
    strategy.iloc[::20] = 0.002
    spy = pd.Series(0.0005, index=index)
    comparators = {
        "buy_now_spy": spy,
        "risk_matched_spy": spy,
        "diversified_baseline": spy,
        "financed_spy_same_risk": spy,
    }
    bounds = log_growth_superiority(strategy, comparators, n_boot=200)
    assert all(value < 0 for value in bounds.values())
    assert annualized_log_growth(strategy) < annualized_log_growth(spy)


def test_shrinkage_quarter_kelly_is_bounded_and_rejects_weak_growth() -> None:
    weak = pd.Series(np.zeros(300))
    strong = pd.Series(np.full(1_000, 0.001))
    assert shrinkage_quarter_kelly_limit(weak, stressed_annual_variance=0.04) == 0.0
    assert 0.0 < shrinkage_quarter_kelly_limit(strong, stressed_annual_variance=0.04) <= 5.0


def test_promoted_daily_signal_only_activates_for_the_next_open() -> None:
    prices = pd.DataFrame(
        {"SPY": [100.0, 101.0, 102.0, 101.0, 103.0]},
        index=pd.bdate_range("2026-07-13", periods=5),
    )
    candidate = {"rule": "breakout", "symbol": "SPY", "lookback": 3, "hold": 1}

    assert daily_position_for_next_open(candidate, prices) == 1.0
    assert daily_position_for_next_open(candidate, prices.iloc[:-1]) == 0.0


def test_daily_backtest_uses_entry_open_to_next_open_not_prior_overnight() -> None:
    index = pd.bdate_range("2026-07-13", periods=5)
    closes = pd.DataFrame({"SPY": [100.0, 100.0, 100.0, 100.0, 100.0]}, index=index)
    opens = pd.DataFrame({"SPY": [100.0, 120.0, 132.0, 132.0, 132.0]}, index=index)
    cash = pd.Series(0.0, index=index)
    candidate = {"rule": "calendar", "symbol": "SPY", "weekday": 0, "hold": 1}

    strategy, position = _daily_strategy_returns(
        candidate,
        closes,
        cash,
        cost_per_exposure_change=0.0,
        execution_prices=opens,
    )

    assert position.loc[index[1]] == 1.0
    assert np.isclose(strategy.loc[index[1]], 0.10)
    assert strategy.loc[index[0]] == 0.0


def test_event_backtest_uses_entry_open_to_next_open_not_prior_overnight() -> None:
    index = pd.bdate_range("2026-07-13", periods=5)
    closes = pd.DataFrame(
        {"AAA": [100.0] * 5, "SPY": [100.0] * 5},
        index=index,
    )
    opens = pd.DataFrame(
        {"AAA": [100.0, 120.0, 132.0, 132.0, 132.0], "SPY": [100.0] * 5},
        index=index,
    )
    event_signals = pd.DataFrame({"AAA": [True, False, False, False, False]}, index=index)
    short_signals = pd.DataFrame(False, index=index, columns=["AAA"])
    zeros = pd.Series(0.0, index=index)

    strategy, weights = _event_candidate_returns(
        {"rule": "earnings_drift", "horizon": 1},
        closes,
        ("AAA",),
        event_signals,
        short_signals,
        zeros,
        zeros,
        zeros,
        cost_per_exposure_change=0.0,
        execution_prices=opens,
    )

    assert weights.loc[index[1], "AAA"] == 1.0
    assert np.isclose(strategy.loc[index[1]], 0.10)
    assert strategy.loc[index[0]] == 0.0


def test_inverse_etf_accounts_for_daily_reset_and_all_drags() -> None:
    index_returns = pd.Series([0.10, -0.0909090909])
    result = account_inverse_etf(
        index_returns,
        annual_tracking_difference=0.01,
        annual_expense_ratio=0.01,
        annual_financing_rate=0.02,
        round_trip_cost_bps=20,
    )
    assert result.compounded_return < result.static_inverse_return
    assert result.total_drag < 0


def test_simulated_broker_contract_is_idempotent_and_shadow_books_are_isolated() -> None:
    broker = get_broker(EdgeStackConfig())
    assert isinstance(broker, SimulatedBroker)
    order = Order("SPY", 200.0, OrderType.MARKET_ON_OPEN, pd.Timestamp("2026-07-20"))
    preview = broker.preview_order(order, reference_price=600.0, reference_volume=1_000.0)
    assert preview.maximum_fill_quantity == 100.0
    first = broker.submit_order(order, idempotency_key="strategy:20260720:entry")
    second = broker.submit_order(order, idempotency_key="strategy:20260720:entry")
    assert first.order_id == second.order_id
    bar = Bar(pd.Timestamp("2026-07-20"), 600.0, 605.0, 595.0, 603.0, 1_000.0)
    fill = broker.execute_against_bar(order, bar, at_open_phase=True)
    assert fill is not None and abs(fill.quantity) == 100.0
    assert order.status is OrderStatus.PARTIAL
    report = broker.reconcile((order.order_id,))
    assert report.clean

    other = SimulatedBroker(CostModel.from_config(EdgeStackConfig()))
    book_a = ShadowBook("a", broker)
    book_b = ShadowBook("b", other)
    assert shadow_books_are_isolated(book_a, book_b)
    book_a.execute(
        Order("SPY", 10.0, OrderType.MARKET_ON_OPEN, pd.Timestamp("2026-07-20")),
        bar,
        at_open_phase=True,
    )
    assert "SPY" in book_a.positions and "SPY" not in book_b.positions
