"""Exchange-aware trading calendar.

Wraps ``exchange_calendars`` (XNYS by default) and derives calendar facts —
pre/post-holiday sessions, turn-of-month, options expiration, year-end windows —
from the *actual* exchange calendar, never from naive weekday arithmetic.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from edgestack.exceptions import DataError


@lru_cache(maxsize=4)
def _get_calendar(name: str) -> xcals.ExchangeCalendar:
    # Widest supported window so historical studies are not silently clipped.
    return xcals.get_calendar(name, start=pd.Timestamp("1990-01-01"))


class TradingCalendar:
    """Thin, typed facade over an exchange_calendars calendar."""

    def __init__(self, name: str = "XNYS") -> None:
        self.name = name
        self._cal = _get_calendar(name)

    # -- session arithmetic -------------------------------------------------

    def sessions(self, start: date, end: date) -> pd.DatetimeIndex:
        """All sessions in ``[start, end]`` as tz-naive normalized timestamps."""
        if start > end:
            raise DataError(f"start {start} after end {end}")
        return self._cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))

    def is_session(self, day: date) -> bool:
        return bool(self._cal.is_session(pd.Timestamp(day)))

    def next_session(self, day: date, count: int = 1) -> pd.Timestamp:
        """The ``count``-th session strictly after ``day``."""
        if count < 1:
            raise DataError("count must be >= 1")
        ts = pd.Timestamp(day)
        session = self._cal.date_to_session(ts, direction="next")
        if session == ts:
            session = self._cal.next_session(session)
        for _ in range(count - 1):
            session = self._cal.next_session(session)
        return session

    def prev_session(self, day: date, count: int = 1) -> pd.Timestamp:
        """The ``count``-th session strictly before ``day``."""
        if count < 1:
            raise DataError("count must be >= 1")
        ts = pd.Timestamp(day)
        session = self._cal.date_to_session(ts, direction="previous")
        if session == ts:
            session = self._cal.previous_session(session)
        for _ in range(count - 1):
            session = self._cal.previous_session(session)
        return session

    def sessions_between(self, start: date, end: date) -> int:
        """Number of sessions strictly after ``start`` up to and including ``end``."""
        sess = self.sessions(start, end)
        if len(sess) and sess[0] == pd.Timestamp(start):
            return len(sess) - 1
        return len(sess)

    def market_open_at(self, session: pd.Timestamp) -> pd.Timestamp:
        """Timezone-aware market open for a session (first executable moment)."""
        return self._cal.session_open(session)

    # -- derived calendar facts ----------------------------------------------

    def opex_days(self, start: date, end: date) -> pd.DatetimeIndex:
        """Standard monthly options-expiration sessions (third Friday, or the
        preceding session when the third Friday is a holiday)."""
        sessions = self.sessions(start, end)
        out: list[pd.Timestamp] = []
        periods = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
        for period in periods:
            first = period.to_timestamp()
            # third Friday of the month
            offset = (4 - first.dayofweek) % 7
            third_friday = first + pd.Timedelta(days=offset + 14)
            if third_friday < pd.Timestamp(start) or third_friday > pd.Timestamp(end):
                continue
            if self._cal.is_session(third_friday):
                out.append(third_friday)
            else:
                prev = self._cal.date_to_session(third_friday, direction="previous")
                if prev in sessions:
                    out.append(prev)
        return pd.DatetimeIndex(out)

    def session_facts(self, start: date, end: date) -> pd.DataFrame:
        """Calendar facts per session, indexed by session date.

        Every column is derived from the true exchange calendar. Columns:
        weekday, is_monday, is_friday, is_pre_holiday, is_post_holiday,
        month, quarter, is_month_first, is_month_last, sessions_since_month_start,
        sessions_to_month_end, is_turn_of_month, is_quarter_end, is_year_end_window,
        is_opex_day, is_opex_week, is_santa_window, sessions_to_christmas.
        """
        # Compute on a padded window so gap/month-boundary logic is exact at the
        # edges of the requested range.
        pad_start = pd.Timestamp(start) - pd.Timedelta(days=45)
        pad_end = pd.Timestamp(end) + pd.Timedelta(days=45)
        sessions = self._cal.sessions_in_range(pad_start, pad_end)
        if len(sessions) < 3:
            raise DataError("calendar window too small")

        df = pd.DataFrame(index=sessions)
        df.index.name = "date"
        gap_next = sessions.to_series().diff().shift(-1).dt.days
        gap_prev = sessions.to_series().diff().dt.days
        weekday = sessions.dayofweek

        df["weekday"] = weekday
        df["is_monday"] = weekday == 0
        df["is_friday"] = weekday == 4
        # Normal gap is 1 day (mid-week) or 3 days (Friday->Monday). Anything
        # longer means the market was closed on a weekday: a holiday.
        normal_next = np.where(weekday == 4, 3, 1)
        normal_prev = np.where(weekday == 0, 3, 1)
        df["is_pre_holiday"] = gap_next.to_numpy() > normal_next
        df["is_post_holiday"] = gap_prev.to_numpy() > normal_prev

        month_key = sessions.to_period("M")
        month_pos = (
            pd.Series(np.arange(len(sessions)), index=sessions).groupby(month_key).cumcount()
        )
        month_len = pd.Series(month_key, index=sessions).map(pd.Series(month_key).value_counts())
        df["month"] = sessions.month
        df["quarter"] = sessions.quarter
        df["sessions_since_month_start"] = month_pos.to_numpy()
        df["sessions_to_month_end"] = (month_len - 1 - month_pos).to_numpy()
        df["is_month_first"] = df["sessions_since_month_start"] == 0
        df["is_month_last"] = df["sessions_to_month_end"] == 0
        # Turn of month: last 2 sessions of a month through first 3 of the next.
        df["is_turn_of_month"] = (df["sessions_to_month_end"] <= 1) | (
            df["sessions_since_month_start"] <= 2
        )
        df["is_quarter_end"] = df["is_month_last"] & df["month"].isin([3, 6, 9, 12])
        # Year-end window: last 5 sessions of December.
        df["is_year_end_window"] = (df["month"] == 12) & (df["sessions_to_month_end"] <= 4)

        opex = set(self.opex_days(pad_start.date(), pad_end.date()))
        df["is_opex_day"] = [s in opex for s in sessions]
        # OpEx week: sessions within the Monday..Friday week of an opex day.
        opex_weeks = {ts.isocalendar()[:2] for ts in opex}
        df["is_opex_week"] = [s.isocalendar()[:2] in opex_weeks for s in sessions]

        # Santa window: last 5 sessions of the year + first 2 of the next.
        year_key = sessions.year
        year_pos = pd.Series(np.arange(len(sessions)), index=sessions).groupby(year_key).cumcount()
        year_len = pd.Series(year_key, index=sessions).map(pd.Series(year_key).value_counts())
        sessions_to_year_end = (year_len - 1 - year_pos).to_numpy()
        df["is_santa_window"] = (sessions_to_year_end <= 4) | (year_pos.to_numpy() <= 1)

        # Sessions to Christmas: for sessions on/before Dec 24, the number of
        # sessions remaining before Dec 25 of the same year; NaN outside December.
        stc = np.full(len(sessions), np.nan)
        for year in np.unique(sessions.year):
            christmas = pd.Timestamp(int(year), 12, 25)
            mask = (sessions.year == year) & (sessions.month == 12) & (sessions < christmas)
            if mask.any():
                positions = np.where(mask)[0]
                stc[positions] = np.arange(len(positions))[::-1]
        df["sessions_to_christmas"] = stc

        return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
