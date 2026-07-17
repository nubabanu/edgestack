"""Trading-calendar tests against known NYSE facts."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from edgestack.data.calendar import TradingCalendar


@pytest.fixture(scope="module")
def cal() -> TradingCalendar:
    return TradingCalendar("XNYS")


def test_known_holidays_are_not_sessions(cal: TradingCalendar) -> None:
    assert not cal.is_session(date(2023, 4, 7))  # Good Friday
    assert not cal.is_session(date(2023, 11, 23))  # Thanksgiving
    assert not cal.is_session(date(2023, 12, 25))  # Christmas
    assert not cal.is_session(date(2023, 1, 16))  # MLK Day
    assert cal.is_session(date(2023, 7, 3))  # early close, still a session
    assert not cal.is_session(date(2023, 7, 8))  # Saturday


def test_next_prev_session_skip_holidays(cal: TradingCalendar) -> None:
    # Thursday before Good Friday 2023 -> next session is Monday.
    assert cal.next_session(date(2023, 4, 6)) == pd.Timestamp("2023-04-10")
    assert cal.prev_session(date(2023, 4, 10)) == pd.Timestamp("2023-04-06")
    assert cal.next_session(date(2023, 4, 6), count=2) == pd.Timestamp("2023-04-11")
    # From a non-session day, next strictly-after session.
    assert cal.next_session(date(2023, 4, 8)) == pd.Timestamp("2023-04-10")


def test_pre_and_post_holiday_flags(cal: TradingCalendar) -> None:
    facts = cal.session_facts(date(2023, 11, 1), date(2023, 11, 30))
    # Wednesday 2023-11-22 precedes Thanksgiving.
    assert bool(facts.loc[pd.Timestamp("2023-11-22"), "is_pre_holiday"])
    # Friday 2023-11-24 follows Thanksgiving.
    assert bool(facts.loc[pd.Timestamp("2023-11-24"), "is_post_holiday"])
    # A plain Tuesday is neither.
    assert not bool(facts.loc[pd.Timestamp("2023-11-14"), "is_pre_holiday"])
    assert not bool(facts.loc[pd.Timestamp("2023-11-14"), "is_post_holiday"])
    # Ordinary Friday->Monday weekend is NOT flagged as a holiday gap.
    assert not bool(facts.loc[pd.Timestamp("2023-11-17"), "is_pre_holiday"])


def test_opex_days_third_friday(cal: TradingCalendar) -> None:
    opex = cal.opex_days(date(2023, 6, 1), date(2023, 6, 30))
    assert list(opex) == [pd.Timestamp("2023-06-16")]
    # April 2025: third Friday (2025-04-18) is Good Friday -> previous session.
    opex_apr = cal.opex_days(date(2025, 4, 1), date(2025, 4, 30))
    assert list(opex_apr) == [pd.Timestamp("2025-04-17")]


def test_turn_of_month_and_month_boundaries(cal: TradingCalendar) -> None:
    facts = cal.session_facts(date(2023, 1, 1), date(2023, 3, 31))
    assert bool(facts.loc[pd.Timestamp("2023-02-01"), "is_month_first"])
    assert bool(facts.loc[pd.Timestamp("2023-02-28"), "is_month_last"])
    assert bool(facts.loc[pd.Timestamp("2023-02-27"), "is_turn_of_month"])  # 2nd-last
    assert bool(facts.loc[pd.Timestamp("2023-03-02"), "is_turn_of_month"])  # 2nd of month
    assert not bool(facts.loc[pd.Timestamp("2023-02-15"), "is_turn_of_month"])
    # Boundary correctness at the requested-range edge: Jan 3 is the first
    # session of 2023 and must be flagged even though December is out of range.
    assert bool(facts.loc[pd.Timestamp("2023-01-03"), "is_month_first"])


def test_year_end_and_christmas_facts(cal: TradingCalendar) -> None:
    facts = cal.session_facts(date(2023, 12, 1), date(2024, 1, 10))
    assert bool(facts.loc[pd.Timestamp("2023-12-29"), "is_year_end_window"])
    assert not bool(facts.loc[pd.Timestamp("2023-12-15"), "is_year_end_window"])
    # 2023-12-22 is the last session before Christmas: 0 sessions remain.
    assert facts.loc[pd.Timestamp("2023-12-22"), "sessions_to_christmas"] == 0
    assert facts.loc[pd.Timestamp("2023-12-21"), "sessions_to_christmas"] == 1
    # Santa window covers the first sessions of January too.
    assert bool(facts.loc[pd.Timestamp("2024-01-03"), "is_santa_window"])


def test_sessions_between(cal: TradingCalendar) -> None:
    # Sessions strictly after Apr 3 up to Apr 10: Apr 4, 5, 6, 10
    # (2023-04-07, Good Friday, is skipped).
    assert cal.sessions_between(date(2023, 4, 3), date(2023, 4, 10)) == 4
    assert cal.sessions_between(date(2023, 4, 3), date(2023, 4, 3)) == 0
