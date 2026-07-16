"""Unit + parity tests for the validated ensemble4 composite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edgestack.strategies import (
    ENSEMBLE_FAMILIES,
    backtest_exposure,
    ensemble_exposure,
    family_positions,
    seasonal_multiplier,
)

ZOO_CACHE = Path("data/cache/zoo/SPY.parquet")


@pytest.fixture()
def bars() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 600
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(3e-4, 0.01, n)), index=idx)
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    opn = low + (high - low) * rng.uniform(0, 1, n)
    return pd.DataFrame({"open": opn, "high": high, "low": low,
                         "close": close, "adj": close}, index=idx)


def test_family_positions_bounds_and_columns(bars: pd.DataFrame) -> None:
    fams = family_positions(bars)
    assert tuple(fams.columns) == ENSEMBLE_FAMILIES
    assert float(fams.min().min()) >= 0.0
    assert float(fams.max().max()) <= 1.5
    assert (fams["vol_target"] <= 1.5).all()
    # binary families are strictly 0/1
    for name in ("trend_or_dip", "breakout_20d", "reversion_3dn"):
        assert set(np.unique(fams[name].dropna())) <= {0.0, 1.0}


def test_ensemble_is_equal_weight_mean(bars: pd.DataFrame) -> None:
    fams = family_positions(bars)
    pd.testing.assert_series_equal(ensemble_exposure(bars), fams.mean(axis=1))
    # custom weights: full weight on one family reproduces that family
    solo = ensemble_exposure(bars, weights={"trend_or_dip": 1.0})
    pd.testing.assert_series_equal(solo, fams["trend_or_dip"], check_names=False)
    with pytest.raises(ValueError):
        ensemble_exposure(bars, weights={"trend_or_dip": 0.0})


def test_seasonal_multiplier_values(bars: pd.DataFrame) -> None:
    idx = pd.bdate_range("2026-08-25", "2026-12-05")
    mult = seasonal_multiplier(pd.DatetimeIndex(idx))
    assert (mult[idx.month == 9] == 0.5).all()
    assert (mult[idx.month == 10] == 1.5).all()
    assert mult[pd.Timestamp("2026-08-31")] == 1.0
    # 7th trading day of Nov-2026 in a weekday calendar is Nov 10 -> flat
    assert mult[pd.Timestamp("2026-11-10")] == 0.0
    assert mult[pd.Timestamp("2026-11-11")] == 1.5


def test_backtest_shifts_and_charges_costs(bars: pd.DataFrame) -> None:
    ret = bars["adj"].pct_change()
    pos = pd.Series(1.0, index=bars.index)
    strat = backtest_exposure(pos, ret)
    # first held session pays the 2 bps entry cost; afterwards matches B&H
    assert strat.iloc[2:].sub(ret.iloc[2:]).abs().max() < 1e-12
    assert strat.iloc[1] == pytest.approx(ret.iloc[1] - 0.0002)


@pytest.mark.skipif(not ZOO_CACHE.exists(), reason="zoo cache not present")
def test_parity_with_strategy_zoo_rules() -> None:
    import sys

    sys.path.insert(0, "scripts")
    from strategy_zoo import build_rules

    df = pd.read_parquet(ZOO_CACHE)
    df = (df.assign(date=df["dt"].dt.tz_localize(None).dt.normalize())
            .set_index("date").drop(columns="dt"))
    rules = build_rules(df)
    fams = family_positions(df)
    zoo_ens = (rules["cmb_trend_or_rsi2dip"] + rules["brk_20d_high_hold10"]
               + rules["vol_target_10pct"] + rules["mr_3down_days"]) / 4
    diff = (ensemble_exposure(df) - zoo_ens).abs().max()
    assert diff < 1e-9, f"package ensemble diverged from zoo reference: {diff}"
    assert (fams["trend_or_dip"] - rules["cmb_trend_or_rsi2dip"]).abs().max() < 1e-9
