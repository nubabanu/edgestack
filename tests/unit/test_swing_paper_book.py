"""Zone math and fill logic tests for the swing paper book."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scripts import swing_paper_book as book


def oscillator(low: float = 50.0, high: float = 120.0, cycles: int = 6) -> pd.Series:
    vals: list[float] = []
    for _ in range(cycles):
        vals += list(np.linspace(low, high, 90))
        vals += list(np.linspace(high, low, 90))
    idx = pd.bdate_range("2018-01-01", periods=len(vals))
    return pd.Series(vals, index=idx)


class TestCurrentZones:
    def test_oscillator_ending_at_trough_is_in_zone(self):
        px = oscillator()  # ends at the low
        zones = book.current_zones(px)
        assert zones is not None
        assert zones["dip_zone"] < 60 and zones["top_zone"] > 100
        assert zones["close"] <= zones["dip_zone"] * book.ENTRY_TOL

    def test_short_history_returns_none(self):
        px = oscillator(cycles=2)
        assert len(px) <= book.TRAILING
        assert book.current_zones(px) is None

    def test_trending_series_without_pivots_returns_none(self):
        idx = pd.bdate_range("2018-01-01", periods=1200)
        px = pd.Series(np.linspace(50, 500, 1200), index=idx)
        assert book.current_zones(px) is None


class TestFills:
    def test_adj_open_fill_uses_next_session_adjusted_open(self):
        idx = pd.bdate_range("2026-07-20", periods=3)
        df = pd.DataFrame(
            {
                "open": [100.0, 102.0, 104.0],
                "close": [101.0, 103.0, 105.0],
                "adj_close": [50.5, 51.5, 52.5],
            },
            index=idx,
        )
        fill = book.adj_open_fill(df, "2026-07-20")
        assert fill is not None
        fill_date, price = fill
        assert fill_date == "2026-07-21"
        assert abs(price - 102.0 * 51.5 / 103.0) < 1e-9

    def test_no_session_after_signal_returns_none(self):
        idx = pd.bdate_range("2026-07-20", periods=1)
        df = pd.DataFrame({"open": [100.0], "close": [101.0], "adj_close": [50.5]}, index=idx)
        assert book.adj_open_fill(df, "2026-07-20") is None
