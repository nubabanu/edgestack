"""Episode clustering and entry-rule tests for the oil shock study."""

from __future__ import annotations

import pandas as pd
from scripts import oil_shock_study as study


def series(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2010-01-04", periods=len(values))
    return pd.Series(values, index=idx)


class TestFindEpisodes:
    def test_two_qualifying_days_close_together_are_one_episode(self):
        closes = series([100.0, 105.0, 110.0, 110.0, 110.0])  # +5% then +4.8%
        assert len(study.find_episodes(closes)) == 1

    def test_qualifying_days_far_apart_are_two_episodes(self):
        flat = [100.0, 105.0] + [105.0] * study.REGIME_SESSIONS + [110.5]
        assert len(study.find_episodes(series(flat))) == 2

    def test_window_jump_without_big_day_qualifies(self):
        closes = series([100.0, 101.7, 103.4, 105.1, 106.9, 108.7, 108.7])
        assert len(study.find_episodes(closes)) == 1

    def test_quiet_series_has_no_episodes(self):
        assert study.find_episodes(series([100.0 + 0.1 * i for i in range(30)])) == []


class TestCrashReboundFilter:
    def test_rebound_after_crash_is_filtered(self):
        # -20% slide over 20 sessions, then a +5% bounce day
        values = [100.0 - 1.0 * i for i in range(21)] + [84.0]
        closes = series(values)
        t0 = len(values) - 1
        assert closes.iloc[t0] / closes.iloc[t0 - 1] - 1 >= study.SHOCK_DAY_RETURN
        assert study.is_crash_rebound(closes, t0) is True

    def test_shock_from_calm_base_is_kept(self):
        values = [100.0] * 21 + [105.0]
        assert study.is_crash_rebound(series(values), 21) is False


class TestEntrySignals:
    def test_r1_signals_on_shock_day(self):
        closes = series([100.0, 105.0, 106.0, 107.0])
        assert study.entry_signal_index(closes, 1, "R1") == 1

    def test_r2_first_down_close(self):
        closes = series([100.0, 105.0, 106.0, 105.5, 105.0])
        assert study.entry_signal_index(closes, 1, "R2") == 3

    def test_r3_pullback_from_post_shock_high(self):
        # high runs to 108, then 106.3 is a -1.57% pullback
        closes = series([100.0, 105.0, 108.0, 106.3, 106.0])
        assert study.entry_signal_index(closes, 1, "R3") == 3

    def test_r3_no_pullback_returns_none(self):
        closes = series([100.0, 105.0] + [105.0 + 0.2 * i for i in range(12)])
        assert study.entry_signal_index(closes, 1, "R3") is None


class TestEventGate:
    def test_insufficient_split_blocks_pass(self):
        # 400 sessions of mild upward drift with one event only
        closes = series([100.0 * (1.0005**i) for i in range(400)])
        adj = closes.copy()
        row = study.evaluate_trial(closes, adj, [50], 5)
        assert row["event_gate"] == "INSUFFICIENT"

    def test_overlapping_signals_are_deduped(self):
        closes = series([100.0 * (1.0005**i) for i in range(400)])
        row = study.evaluate_trial(closes, closes.copy(), [50, 51, 52, 53], 10)
        assert row["events_total"] == 1
