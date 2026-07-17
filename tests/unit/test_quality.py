"""Data-quality gate tests."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from edgestack.data.calendar import TradingCalendar
from edgestack.data.providers.synthetic import SyntheticMarket
from edgestack.data.quality import assess_panel


@pytest.fixture(scope="module")
def cal() -> TradingCalendar:
    return TradingCalendar("XNYS")


@pytest.fixture(scope="module")
def clean_panel() -> pd.DataFrame:
    return SyntheticMarket(seed=1).generate(("CLN",), date(2019, 1, 1), date(2020, 12, 31))


def test_clean_synthetic_passes(clean_panel: pd.DataFrame, cal: TradingCalendar) -> None:
    report = assess_panel(clean_panel, cal)
    assert report.symbols[0].passed, report.summary()


def test_detects_missing_sessions(clean_panel: pd.DataFrame, cal: TradingCalendar) -> None:
    holey = clean_panel.iloc[::2]  # drop every other session
    report = assess_panel(holey, cal)
    assert any("missing" in issue for issue in report.symbols[0].issues)


def test_detects_split_like_cliff(clean_panel: pd.DataFrame, cal: TradingCalendar) -> None:
    df = clean_panel.copy()
    cliff_idx = len(df) // 2
    factor = 0.5  # unadjusted 2:1 split: raw closes cliff, adjusted closes stay smooth
    for col in ("open", "high", "low", "close"):
        df.loc[df.index[cliff_idx:], col] *= factor
    report = assess_panel(df, cal)
    assert any("corporate action" in issue for issue in report.symbols[0].issues)


def test_genuine_crash_in_both_series_is_not_quarantined(
    clean_panel: pd.DataFrame, cal: TradingCalendar
) -> None:
    df = clean_panel.copy()
    cliff_idx = len(df) // 2
    factor = 0.5  # a real one-day -50% shows in raw AND adjusted closes alike
    for col in ("open", "high", "low", "close", "adj_close"):
        df.loc[df.index[cliff_idx:], col] *= factor
    report = assess_panel(df, cal)
    assert not any("corporate action" in issue for issue in report.symbols[0].issues), (
        report.summary()
    )


def test_detects_stale_prices(clean_panel: pd.DataFrame, cal: TradingCalendar) -> None:
    df = clean_panel.copy()
    frozen = df["close"].iloc[100]
    idx = df.index[100:120]
    for col in ("open", "high", "low", "close"):
        df.loc[idx, col] = frozen
    report = assess_panel(df, cal)
    assert any("stale" in issue for issue in report.symbols[0].issues)


def test_detects_zero_volume_runs(clean_panel: pd.DataFrame, cal: TradingCalendar) -> None:
    df = clean_panel.copy()
    df.loc[df.index[50:60], "volume"] = 0.0
    report = assess_panel(df, cal)
    assert any("zero-volume" in issue for issue in report.symbols[0].issues)
