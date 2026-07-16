"""Label-construction tests with hand-computed expectations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edgestack.exceptions import DataError
from edgestack.labels.forward_returns import forward_return_labels, session_returns
from edgestack.labels.triple_barrier import TripleBarrierConfig, triple_barrier_labels


def _panel_from_opens(opens: list[float], symbol: str = "TST") -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=len(opens))
    opens_arr = np.asarray(opens, dtype=float)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": opens_arr,
            "high": opens_arr * 1.01,
            "low": opens_arr * 0.99,
            "close": opens_arr,
            "volume": 1e6,
            "adj_close": opens_arr,
        }
    )


def test_forward_return_exact_values() -> None:
    # Signal at T (index 0), delay 1, horizon 2: entry open[1]=110, exit open[3]=121.
    panel = _panel_from_opens([100, 110, 100, 121, 130, 140])
    labels = forward_return_labels(panel, (2,), benchmark_symbol="TST")
    first = labels.iloc[0]
    assert first["date"] == pd.Timestamp("2020-01-01")
    assert first["entry_date"] == pd.Timestamp("2020-01-02")
    assert first["label_end"] == pd.Timestamp("2020-01-06")  # skips the weekend
    assert first["gross_ret"] == pytest.approx(121 / 110 - 1)
    # Benchmark == the symbol itself here, so excess return is zero.
    assert first["excess_ret"] == pytest.approx(0.0)


def test_rows_without_resolved_outcome_are_dropped() -> None:
    panel = _panel_from_opens([100, 110, 100, 121, 130, 140])
    labels = forward_return_labels(panel, (2,), benchmark_symbol="TST")
    # 6 sessions, delay 1 + horizon 2 -> last valid signal is index 2 (exit at 5).
    assert len(labels) == 3
    assert labels["label_end"].max() == pd.Timestamp(panel["date"].iloc[-1])


def test_same_close_execution_is_blocked() -> None:
    panel = _panel_from_opens([100, 110, 120])
    with pytest.raises(DataError, match="cannot be executed"):
        forward_return_labels(panel, (1,), benchmark_symbol="TST", execution_delay=0)
    with pytest.raises(DataError, match=">= 1"):
        triple_barrier_labels(panel, TripleBarrierConfig(execution_delay=0))


def test_session_return_decomposition() -> None:
    bars = pd.DataFrame(
        {"open": [100.0, 104.0], "close": [102.0, 103.0]},
        index=pd.bdate_range("2020-01-01", periods=2),
    )
    out = session_returns(bars)
    assert out["close_to_open"].iloc[1] == pytest.approx(104 / 102 - 1)  # overnight
    assert out["open_to_close"].iloc[1] == pytest.approx(103 / 104 - 1)  # intraday
    assert out["close_to_close"].iloc[1] == pytest.approx(103 / 102 - 1)


def _tb_panel(
    closes: np.ndarray, opens: np.ndarray | None = None, spread: float = 1.0
) -> pd.DataFrame:
    n = len(closes)
    opens_arr = closes.copy() if opens is None else opens
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "symbol": "TB",
            "date": dates,
            "open": opens_arr,
            "high": np.maximum(closes, opens_arr) + spread,
            "low": np.minimum(closes, opens_arr) - spread,
            "close": closes,
            "volume": 1e6,
            "adj_close": closes,
        }
    )


def test_triple_barrier_upper_hit_and_conservative_fill() -> None:
    # 30 warmup bars at 100 (TR = 2 -> ATR = 2), then an intraday surge:
    # open stays at 100 but the high (max(open, close) + 1 = 105) touches the
    # up barrier 100 + 2.5*2 = 105 without gapping.
    closes = np.full(40, 100.0)
    opens = closes.copy()
    closes[33] = 104.0
    panel = _tb_panel(closes, opens)
    cfg = TripleBarrierConfig(horizon=8, up_atr_multiple=2.5, down_atr_multiple=2.0)
    labels = triple_barrier_labels(panel, cfg)
    row = labels.loc[labels["date"] == panel["date"].iloc[30]].iloc[0]
    assert row["outcome"] == 1
    # Fill at the barrier, not at the (better) high.
    assert row["gross_ret"] == pytest.approx(105.0 / 100.0 - 1.0, abs=1e-9)


def test_triple_barrier_gap_through_stop_fills_at_open() -> None:
    closes = np.full(40, 100.0)
    opens = closes.copy()
    opens[34] = 90.0  # gaps far below the stop 100 - 2*2 = 96
    closes[34] = 90.0
    panel = _tb_panel(closes, opens)
    cfg = TripleBarrierConfig(horizon=8)
    labels = triple_barrier_labels(panel, cfg)
    row = labels.loc[labels["date"] == panel["date"].iloc[30]].iloc[0]
    assert row["outcome"] == -1
    # NOT filled at the stop price 96 — filled at the gapped open 90.
    assert row["gross_ret"] == pytest.approx(90.0 / 100.0 - 1.0, abs=1e-9)


def test_triple_barrier_time_exit() -> None:
    closes = np.full(45, 100.0)
    panel = _tb_panel(closes, spread=0.5)  # barriers never touched
    cfg = TripleBarrierConfig(horizon=5)
    labels = triple_barrier_labels(panel, cfg)
    row = labels.iloc[0]
    assert row["outcome"] == 0
    assert row["holding_sessions"] == 6  # exits at the open AFTER the 5-session window
    assert row["gross_ret"] == pytest.approx(0.0, abs=1e-9)


def test_triple_barrier_label_end_never_before_date() -> None:
    rng = np.random.default_rng(0)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 120)))
    labels = triple_barrier_labels(_tb_panel(closes), TripleBarrierConfig(horizon=10))
    assert (labels["label_end"] >= labels["date"]).all()
    assert (labels["entry_date"] > labels["date"]).all()
