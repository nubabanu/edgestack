"""Zigzag extraction and swing-metric tests for the swing cycle scan."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scripts import swing_cycle_scan as scan


def sawtooth(low: float, high: float, cycles: int, leg: int = 60) -> pd.Series:
    """Clean oscillator: low -> high -> low, `cycles` times."""
    vals: list[float] = []
    for _ in range(cycles):
        vals += list(np.linspace(low, high, leg))
        vals += list(np.linspace(high, low, leg))
    idx = pd.bdate_range("2015-01-05", periods=len(vals))
    return pd.Series(vals, index=idx)


class TestZigzag:
    def test_sawtooth_pivots_recovered(self):
        px = sawtooth(50.0, 120.0, cycles=4)
        peaks, troughs = scan.zigzag(np.log(px.to_numpy()), 0.30)
        assert len(peaks) == 4
        assert len(troughs) >= 3  # final trough may be unconfirmed at series end
        assert all(abs(px.iloc[i] - 120.0) < 1e-6 for i in peaks)
        assert all(abs(px.iloc[i] - 50.0) < 1e-6 for i in troughs)

    def test_flat_series_has_no_pivots(self):
        px = np.log(np.full(500, 100.0))
        peaks, troughs = scan.zigzag(px, 0.15)
        assert peaks == [] and troughs == []

    def test_threshold_filters_small_swings(self):
        px = sawtooth(100.0, 118.0, cycles=5)  # ~17% swings
        peaks15, _ = scan.zigzag(np.log(px.to_numpy()), 0.15)
        peaks50, _ = scan.zigzag(np.log(px.to_numpy()), 0.50)
        assert len(peaks15) == 5
        assert len(peaks50) == 0

    def test_pivots_alternate(self):
        rng = np.random.default_rng(3)
        px = np.log(100 * np.exp(np.cumsum(rng.standard_normal(3000) * 0.03)))
        peaks, troughs = scan.zigzag(px, 0.15)
        import itertools

        merged = sorted([(i, "P") for i in peaks] + [(i, "T") for i in troughs])
        kinds = [k for _, k in merged]
        assert all(a != b for a, b in itertools.pairwise(kinds))


class TestAnalyze:
    def test_clean_oscillator_scores_high_with_tight_zones(self):
        px = sawtooth(50.0, 120.0, cycles=6)
        row = scan.analyze(px, 0.30)
        assert row is not None
        assert row["trough_zone_std"] < 0.02
        assert 45 < row["median_trough"] < 55
        assert 110 < row["median_peak"] < 125
        assert row["swings_per_year"] > 1

    def test_decayer_is_excluded(self):
        base = sawtooth(50.0, 120.0, cycles=6).to_numpy()
        decay = base * np.exp(-0.10 / 252 * np.arange(len(base)))  # -10%/yr drift
        px = pd.Series(decay, index=pd.bdate_range("2015-01-05", periods=len(decay)))
        assert scan.analyze(px, 0.30) is None

    def test_one_sided_series_is_excluded(self):
        idx = pd.bdate_range("2015-01-05", periods=800)
        px = pd.Series(np.linspace(50, 500, 800), index=idx)  # pure uptrend
        assert scan.analyze(px, 0.30) is None

    def test_swings_only_in_one_half_fail_split_gate(self):
        flat = list(np.full(720, 85.0))
        osc = sawtooth(50.0, 120.0, cycles=3).to_numpy().tolist()
        vals = osc + flat
        px = pd.Series(vals, index=pd.bdate_range("2015-01-05", periods=len(vals)))
        assert scan.analyze(px, 0.30) is None
