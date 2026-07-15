"""Calendar features, sourced from the exchange calendar facts the engine merges.

None of these encode folklore ("Fridays are bullish"); they only expose the
calendar state so discovery can test whether any effect actually exists.
"""

from __future__ import annotations

import pandas as pd

from edgestack.features.registry import FeatureFn, feature
from edgestack.types import Family


def _make(column: str) -> FeatureFn:
    def fn(df: pd.DataFrame) -> pd.Series:
        return df[column].astype(float)

    fn.__doc__ = f"Exchange-calendar fact `{column}` as a float feature."
    return fn


for _col, _name in [
    ("weekday", "cal_weekday"),
    ("is_monday", "cal_is_monday"),
    ("is_friday", "cal_is_friday"),
    ("is_pre_holiday", "cal_pre_holiday"),
    ("is_post_holiday", "cal_post_holiday"),
    ("is_turn_of_month", "cal_turn_of_month"),
    ("month", "cal_month"),
    ("is_opex_week", "cal_opex_week"),
    ("is_year_end_window", "cal_year_end"),
    ("is_santa_window", "cal_santa_window"),
    ("sessions_to_month_end", "cal_sessions_to_month_end"),
]:
    feature(_name, Family.CALENDAR, inputs=(_col,), description=f"calendar fact {_col}")(
        _make(_col)
    )
