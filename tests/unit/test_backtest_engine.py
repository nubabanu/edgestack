"""Backtest-engine accounting and behavior tests on hand-built panels."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from edgestack.backtest.engine import BacktestEngine, TradeIntent
from edgestack.config import EdgeStackConfig
from edgestack.types import CostScenario, Side


def _cfg(**risk_overrides) -> EdgeStackConfig:
    risk = {"max_position_weight": 0.10, "max_positions": 5,
            "max_gross_exposure": 1.0, "max_net_exposure": 1.0}
    risk.update(risk_overrides)
    return EdgeStackConfig.model_validate({"risk": risk})


def _panel(opens: list[float], lows: list[float] | None = None,
           highs: list[float] | None = None) -> pd.DataFrame:
    n = len(opens)
    dates = pd.bdate_range("2020-01-01", periods=n)
    opens_arr = np.asarray(opens, dtype=float)
    highs_arr = np.asarray(highs, dtype=float) if highs else opens_arr * 1.01
    lows_arr = np.asarray(lows, dtype=float) if lows else opens_arr * 0.99
    return pd.DataFrame(
        {
            "symbol": "TST", "date": dates, "open": opens_arr,
            "high": np.maximum(highs_arr, opens_arr),
            "low": np.minimum(lows_arr, opens_arr),
            "close": opens_arr, "volume": 1e9, "adj_close": opens_arr,
        }
    )


def _intent(session: pd.Timestamp, horizon: int = 3, stop: float = 50.0,
            target: float = 500.0, side: Side = Side.LONG) -> TradeIntent:
    return TradeIntent(symbol="TST", signal_session=session, side=side,
                       horizon=horizon, stop_price=stop, target_price=target)


def test_time_exit_accounting_is_exact() -> None:
    # Signal at session 1 close -> entry at open of session 2 (100.0),
    # horizon 3 -> time exit at open of session 5 (109.0).
    opens = [100, 100, 100, 103, 106, 109, 109, 109]
    panel = _panel(opens)
    engine = BacktestEngine(panel, _cfg(), CostScenario.BASE)
    intent = _intent(panel["date"].iloc[1])
    ledger = engine.run([intent], initial_cash=100_000)

    trades = ledger.trades_frame()
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "time_exit"
    assert trade["entry_price"] == 100.0
    assert trade["exit_price"] == 109.0
    # 10% weight of 100k equity at ref close 100 -> 100 shares.
    assert trade["quantity"] == 100.0
    gross_pnl = 100 * (109 - 100)
    assert trade["net_pnl"] < gross_pnl  # both legs' friction included
    assert trade["net_pnl"] > gross_pnl - 50

    equity = ledger.equity_frame()
    assert equity["equity"].iloc[-1] == pytest.approx(100_000 + trade["net_pnl"], abs=1e-6)


def test_gap_through_stop_exits_at_open() -> None:
    opens = [100, 100, 100, 80, 80, 80, 80]  # session 3 gaps far below the stop
    panel = _panel(opens)
    engine = BacktestEngine(panel, _cfg(), CostScenario.BASE)
    intent = _intent(panel["date"].iloc[1], horizon=5, stop=95.0)
    ledger = engine.run([intent])
    trade = ledger.trades_frame().iloc[0]
    assert trade["exit_reason"] == "stop_loss"
    assert trade["exit_price"] == 80.0  # at the gapped open, NOT the stop price


def test_target_exit_when_touched() -> None:
    opens = [100, 100, 100, 100, 100, 100]
    highs = [101, 101, 101, 112, 112, 112]
    panel = _panel(opens, highs=highs)
    engine = BacktestEngine(panel, _cfg(), CostScenario.BASE)
    intent = _intent(panel["date"].iloc[1], horizon=5, target=110.0)
    ledger = engine.run([intent])
    trade = ledger.trades_frame().iloc[0]
    assert trade["exit_reason"] == "target"
    assert trade["exit_price"] == 110.0


def test_short_trade_profits_from_decline() -> None:
    opens = [100, 100, 100, 96, 92, 88, 88, 88]
    panel = _panel(opens)
    engine = BacktestEngine(panel, _cfg(), CostScenario.BASE)
    intent = _intent(panel["date"].iloc[1], horizon=3, side=Side.SHORT,
                     stop=150.0, target=10.0)
    ledger = engine.run([intent])
    trade = ledger.trades_frame().iloc[0]
    assert trade["side"] == "SHORT"
    assert trade["net_pnl"] > 0
    # Shorts pay borrow: some borrow cash must have been charged.
    assert ledger.equity_frame()["borrow_paid"].sum() > 0


def test_position_and_exposure_limits_respected() -> None:
    opens = [100.0] * 10
    panel = _panel(opens)
    cfg = _cfg(max_positions=1)
    engine = BacktestEngine(panel, cfg, CostScenario.BASE)
    intents = [_intent(panel["date"].iloc[1], horizon=6),
               _intent(panel["date"].iloc[2], horizon=6)]
    ledger = engine.run(intents)
    # Second intent arrives while the first position is open: refused.
    assert len(ledger.trades_frame()) == 1


def test_many_same_day_signals_respect_gross_and_position_caps() -> None:
    """Pending next-open entries must count against the caps: a burst of
    signals in one session cannot lever the book past max_gross_exposure."""
    n_symbols, n_sessions = 40, 12
    dates = pd.bdate_range("2020-01-01", periods=n_sessions)
    frames = []
    for i in range(n_symbols):
        opens = np.full(n_sessions, 50.0 + i)
        frames.append(pd.DataFrame({
            "symbol": f"S{i:02d}", "date": dates, "open": opens,
            "high": opens * 1.01, "low": opens * 0.99, "close": opens,
            "volume": 1e9, "adj_close": opens,
        }))
    panel = pd.concat(frames, ignore_index=True)
    cfg = _cfg(max_position_weight=0.10, max_positions=30,
               max_gross_exposure=0.5, max_net_exposure=0.5)
    engine = BacktestEngine(panel, cfg, CostScenario.BASE)
    intents = [
        TradeIntent(symbol=f"S{i:02d}", signal_session=dates[1], side=Side.LONG,
                    horizon=8, stop_price=1.0, target_price=1e6)
        for i in range(n_symbols)
    ]
    ledger = engine.run(intents, initial_cash=100_000)
    eq = ledger.equity_frame()
    assert (eq["gross_exposure"] <= 0.5 * eq["equity"] + 1e-6).all()
    # 0.5 gross / 0.10 per position -> at most 5 concurrent positions.
    assert eq["n_positions"].max() <= 5


def test_property_worse_cost_scenarios_never_finish_richer() -> None:
    rng = np.random.default_rng(3)
    opens = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60))))
    panel = _panel(opens)
    intents = [_intent(panel["date"].iloc[i], horizon=4) for i in (1, 10, 20, 30, 40)]
    finals = []
    for scenario in (CostScenario.OPTIMISTIC, CostScenario.BASE,
                     CostScenario.CONSERVATIVE, CostScenario.STRESS):
        engine = BacktestEngine(panel, _cfg(), scenario)
        ledger = engine.run(intents)
        finals.append(ledger.equity_frame()["equity"].iloc[-1])
    assert all(a >= b - 1e-9 for a, b in itertools.pairwise(finals))
