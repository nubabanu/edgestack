from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from edgestack.data.calendar import TradingCalendar
from edgestack.research.market_wind import (
    COMPONENTS,
    bars_fingerprint,
    component_votes,
    score_bucket_report,
    snapshot_for_next_session,
)


def _bars(start: date = date(2022, 1, 3), end: date = date(2023, 4, 6)) -> pd.DataFrame:
    sessions = TradingCalendar().sessions(start, end)
    close = pd.Series(np.linspace(100.0, 140.0, len(sessions)), index=sessions)
    return pd.DataFrame({"close": close, "adj_close": close})


def test_component_names_are_neutral_and_calendar_windows_use_xnys() -> None:
    bars = _bars()
    votes = component_votes(bars)

    assert tuple(votes.columns) == COMPONENTS
    assert votes.loc["2023-03-31", "turn_of_month"] == 1
    assert votes.loc["2023-04-03", "turn_of_month"] == 1
    assert votes.loc["2023-04-06", "turn_of_month"] == 0
    assert votes.loc["2023-04-06", "trend_vol_regime"] == 1


def test_post_down_month_weekday_reversal_and_short_dip_votes() -> None:
    bars = _bars(end=date(2023, 5, 5))
    # Make March a down month and Friday 2023-04-28 the third consecutive
    # decline.  Monday 2023-05-01 therefore has both the calendar-known
    # previous-month feature and prior-close reversal/dip inputs.
    march = (bars.index.month == 3) & (bars.index.year == 2023)
    bars.loc[march, ["close", "adj_close"]] = np.linspace(150.0, 120.0, march.sum())[:, None]
    for session, value in (("2023-04-26", 119.0), ("2023-04-27", 117.0), ("2023-04-28", 115.0)):
        bars.loc[session, ["close", "adj_close"]] = value

    votes = component_votes(bars)

    assert votes.loc["2023-04-03", "post_down_month"] == 1
    assert votes.loc["2023-05-01", "weekday_reversal"] == 1
    assert votes.loc["2023-05-01", "short_term_dip"] == 1


def test_future_price_changes_cannot_change_an_earlier_vote() -> None:
    bars = _bars(end=date(2023, 5, 5))
    before = component_votes(bars).loc["2023-04-28"].copy()
    changed = bars.copy()
    changed.loc[changed.index > "2023-04-28", ["close", "adj_close"]] *= 10

    pd.testing.assert_series_equal(before, component_votes(changed).loc["2023-04-28"])


def test_live_snapshot_matches_historical_row_and_skips_good_friday() -> None:
    bars = _bars(end=date(2023, 4, 6))
    snapshot = snapshot_for_next_session(bars, evaluated_on=date(2023, 4, 6))
    target = pd.Timestamp("2023-04-10")
    extended = pd.concat(
        [bars, pd.DataFrame(index=pd.DatetimeIndex([target]), columns=bars.columns)]
    )
    expected = component_votes(extended).loc[target]

    assert snapshot.status == "READY"
    assert snapshot.for_session == date(2023, 4, 10)
    assert snapshot.action == "NO_ACTION"
    assert snapshot.votes == {name: int(expected[name]) for name in COMPONENTS}


def test_snapshot_fails_closed_for_short_or_stale_history() -> None:
    short = _bars(start=date(2023, 1, 3), end=date(2023, 4, 6))
    insufficient = snapshot_for_next_session(short, evaluated_on=date(2023, 4, 6))
    stale = snapshot_for_next_session(_bars(), evaluated_on=date(2023, 4, 14))

    assert insufficient.status == "UNAVAILABLE"
    assert "INSUFFICIENT_HISTORY" in insufficient.reasons
    assert stale.status == "UNAVAILABLE"
    assert any(reason.startswith("STALE_INPUT") for reason in stale.reasons)


def test_fingerprint_is_cutoff_stable_and_sensitive_before_cutoff() -> None:
    bars = _bars()
    initial = bars_fingerprint(bars, cutoff=date(2023, 4, 3))
    future_changed = bars.copy()
    future_changed.loc[future_changed.index > "2023-04-03", "close"] *= 2
    past_changed = bars.copy()
    past_changed.loc["2023-03-31", "close"] *= 2

    assert bars_fingerprint(future_changed, cutoff=date(2023, 4, 3)) == initial
    assert bars_fingerprint(past_changed, cutoff=date(2023, 4, 3)) != initial


def test_score_report_keeps_extreme_buckets_and_requires_strict_increase() -> None:
    score = pd.Series([-2, -1, 0, 1, 2, 3, 4, 5], dtype=float)
    returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.03, 0.06, 0.07, 0.08])
    report = score_bucket_report(score, returns)

    assert set(report["buckets"]) == {"-2", "-1", "0", "1", "2", "3", "4", "5"}
    assert report["strict_monotonic"] is False
