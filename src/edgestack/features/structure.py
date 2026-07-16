"""Market-structure features: breakouts, channels, pivots (with availability lag)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgestack.features.registry import feature
from edgestack.types import Family


@feature("dist_to_high_20", Family.STRUCTURE, min_history=21)
def dist_to_high_20(df: pd.DataFrame) -> pd.Series:
    """Close relative to the prior 20-session high (excluding today)."""
    prior_high = df["high"].rolling(20).max().shift(1)
    return df["close"] / prior_high - 1.0


@feature("dist_to_low_20", Family.STRUCTURE, min_history=21)
def dist_to_low_20(df: pd.DataFrame) -> pd.Series:
    """Close relative to the prior 20-session low (excluding today)."""
    prior_low = df["low"].rolling(20).min().shift(1)
    return df["close"] / prior_low - 1.0


@feature("breakout_20", Family.STRUCTURE, min_history=21)
def breakout_20(df: pd.DataFrame) -> pd.Series:
    """1.0 when the close exceeds the prior 20-session high."""
    prior_high = df["high"].rolling(20).max().shift(1)
    return (df["close"] > prior_high).astype(float).where(prior_high.notna())


@feature("breakdown_20", Family.STRUCTURE, min_history=21)
def breakdown_20(df: pd.DataFrame) -> pd.Series:
    """1.0 when the close falls below the prior 20-session low."""
    prior_low = df["low"].rolling(20).min().shift(1)
    return (df["close"] < prior_low).astype(float).where(prior_low.notna())


@feature("false_breakout_20", Family.STRUCTURE, min_history=27)
def false_breakout_20(df: pd.DataFrame) -> pd.Series:
    """Broke the 20-session high within the last 5 sessions but closed back below it."""
    prior_high = df["high"].rolling(20).max().shift(1)
    broke = (df["close"] > prior_high).astype(float)
    broke_recently = broke.shift(1).rolling(5).max()
    back_below = (df["close"] < prior_high).astype(float)
    return (broke_recently * back_below).where(prior_high.notna())


@feature("donchian_pos_55", Family.STRUCTURE, inputs=("high", "low", "close"), min_history=55)
def donchian_pos_55(df: pd.DataFrame) -> pd.Series:
    """Close position within the 55-session Donchian channel (0..1)."""
    high55 = df["high"].rolling(55).max()
    low55 = df["low"].rolling(55).min()
    rng = (high55 - low55).where(lambda s: s > 0)
    return (df["close"] - low55) / rng


@feature("days_since_high_20", Family.STRUCTURE, min_history=21)
def days_since_high_20(df: pd.DataFrame) -> pd.Series:
    """Sessions since the trailing 20-session high was set."""
    return df["high"].rolling(20).apply(lambda w: len(w) - 1 - int(np.argmax(w)), raw=True)


@feature(
    "pivot_high_2",
    Family.STRUCTURE,
    inputs=("high",),
    min_history=5,
    availability_lag=2,
    description="Confirmed swing high (higher than 2 bars on each side); needs 2 "
    "later bars to confirm, so it carries availability_lag=2.",
)
def pivot_high_2(df: pd.DataFrame) -> pd.Series:
    h = df["high"]
    is_pivot = (h > h.shift(1)) & (h > h.shift(2)) & (h > h.shift(-1)) & (h > h.shift(-2))
    return is_pivot.astype(float).where(h.shift(-2).notna())


@feature(
    "pivot_low_2",
    Family.STRUCTURE,
    inputs=("low",),
    min_history=5,
    availability_lag=2,
    description="Confirmed swing low; needs 2 later bars, availability_lag=2.",
)
def pivot_low_2(df: pd.DataFrame) -> pd.Series:
    low = df["low"]
    is_pivot = (
        (low < low.shift(1)) & (low < low.shift(2)) & (low < low.shift(-1)) & (low < low.shift(-2))
    )
    return is_pivot.astype(float).where(low.shift(-2).notna())
