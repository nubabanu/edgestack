"""Synthetic market generator invariants."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from edgestack.data.calendar import TradingCalendar
from edgestack.data.providers.synthetic import (
    GBM,
    CalendarEffect,
    DecayingEdge,
    Momentum,
    SyntheticMarket,
)

START, END = date(2018, 1, 1), date(2020, 12, 31)


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    market = SyntheticMarket(seed=7)
    return market.generate(("AAA", "BBB"), START, END)


def test_ohlc_invariants(bars: pd.DataFrame) -> None:
    assert (bars["high"] >= bars[["open", "close", "low"]].max(axis=1)).all()
    assert (bars["low"] <= bars[["open", "close", "high"]].min(axis=1)).all()
    assert (bars[["open", "high", "low", "close"]] > 0).all().all()
    assert (bars["volume"] >= 0).all()


def test_sessions_match_exchange_calendar(bars: pd.DataFrame) -> None:
    cal = TradingCalendar("XNYS")
    expected = cal.sessions(START, END)
    got = pd.DatetimeIndex(bars.loc[bars["symbol"] == "AAA", "date"])
    assert got.equals(expected)


def test_deterministic_per_seed_and_symbol() -> None:
    a = SyntheticMarket(seed=7).generate(("AAA",), START, END)
    b = SyntheticMarket(seed=7).generate(("AAA",), START, END)
    pd.testing.assert_frame_equal(a, b)
    c = SyntheticMarket(seed=8).generate(("AAA",), START, END)
    assert not np.allclose(a["close"], c["close"])


def test_adding_symbol_does_not_change_existing_path() -> None:
    solo = SyntheticMarket(seed=7).generate(("AAA",), START, END)
    both = SyntheticMarket(seed=7).generate(("AAA", "ZZZ"), START, END)
    pd.testing.assert_frame_equal(solo, both.loc[both["symbol"] == "AAA"].reset_index(drop=True))


def test_calendar_effect_shifts_selected_sessions() -> None:
    # Large injected drift so the test is decisive without being slow.
    market = SyntheticMarket(
        seed=11,
        base=GBM(mu=0.0, sigma=0.10),
        effects=(CalendarEffect(facts_column="is_friday", drift_bps=80.0),),
    )
    bars = market.generate(("EDG",), date(2012, 1, 1), END)
    rets = np.log(bars["close"] / bars["close"].shift()).dropna()
    dates = pd.DatetimeIndex(bars["date"]).to_series().iloc[1:]
    friday = dates.dt.dayofweek.to_numpy() == 4
    assert rets[friday].mean() > rets[~friday].mean() + 0.0005


def test_time_boxed_effect_is_inactive_outside_window() -> None:
    effect = CalendarEffect(
        facts_column="is_friday",
        drift_bps=80.0,
        active_start=date(2018, 1, 1),
        active_end=date(2018, 12, 31),
    )
    with_box = SyntheticMarket(seed=11, base=GBM(mu=0.0), effects=(effect,))
    plain = SyntheticMarket(seed=11, base=GBM(mu=0.0))
    boxed = with_box.generate(("EDG",), date(2019, 6, 1), date(2020, 6, 1))
    base = plain.generate(("EDG",), date(2019, 6, 1), date(2020, 6, 1))
    # Outside the active window the effect contributes nothing at all.
    pd.testing.assert_frame_equal(boxed, base)


def test_momentum_effect_induces_autocorrelation() -> None:
    plain = SyntheticMarket(seed=3, base=GBM(mu=0.0))
    trending = SyntheticMarket(seed=3, base=GBM(mu=0.0), effects=(Momentum(rho=0.5, lookback=5),))
    r_plain = np.log(plain.generate(("MOM",), date(2010, 1, 1), END)["close"]).diff().dropna()
    r_trend = np.log(trending.generate(("MOM",), date(2010, 1, 1), END)["close"]).diff().dropna()
    assert r_trend.autocorr(1) > r_plain.autocorr(1) + 0.02


def test_decaying_edge_fades() -> None:
    market = SyntheticMarket(
        seed=5,
        base=GBM(mu=0.0, sigma=0.08),
        effects=(DecayingEdge(facts_column="is_monday", initial_bps=60.0, half_life_sessions=200),),
    )
    bars = market.generate(("DEC",), date(2010, 1, 1), date(2019, 12, 31))
    rets = np.log(bars["close"] / bars["close"].shift())
    dates = pd.DatetimeIndex(bars["date"])
    monday = dates.dayofweek == 0
    early = (dates < pd.Timestamp("2012-01-01")) & monday
    late = (dates >= pd.Timestamp("2017-01-01")) & monday
    assert rets[early].mean() > rets[late].mean()
