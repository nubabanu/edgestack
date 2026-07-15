"""Candlestick / bar-structure features.

Numerical definitions with configurable-in-code thresholds; detections are
*features* for the evidence engine to weigh, never trade instructions.
"""

from __future__ import annotations

import pandas as pd

from edgestack.features.registry import feature
from edgestack.types import Family

_BODY_DOJI_MAX = 0.10      # body <= 10% of range
_WICK_HAMMER_MIN = 2.0     # lower wick >= 2x body
_WICK_OTHER_MAX = 0.30     # opposite wick <= 30% of body-adjusted range


def _parts(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).where(lambda s: s > 0)
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    return body, rng, upper, lower


@feature("candle_doji", Family.STRUCTURE, inputs=("open", "high", "low", "close"))
def candle_doji(df: pd.DataFrame) -> pd.Series:
    """Doji: negligible body relative to the range."""
    body, rng, _, _ = _parts(df)
    return (body / rng <= _BODY_DOJI_MAX).astype(float).where(rng.notna())


@feature("candle_hammer", Family.STRUCTURE, inputs=("open", "high", "low", "close"))
def candle_hammer(df: pd.DataFrame) -> pd.Series:
    """Hammer: long lower wick, small upper wick."""
    body, rng, upper, lower = _parts(df)
    ok = (lower >= _WICK_HAMMER_MIN * body) & (upper <= _WICK_OTHER_MAX * rng) & (body > 0)
    return ok.astype(float).where(rng.notna())


@feature("candle_shooting_star", Family.STRUCTURE, inputs=("open", "high", "low", "close"))
def candle_shooting_star(df: pd.DataFrame) -> pd.Series:
    """Shooting star: long upper wick, small lower wick."""
    body, rng, upper, lower = _parts(df)
    ok = (upper >= _WICK_HAMMER_MIN * body) & (lower <= _WICK_OTHER_MAX * rng) & (body > 0)
    return ok.astype(float).where(rng.notna())


@feature("candle_bull_engulf", Family.STRUCTURE, inputs=("open", "close"))
def candle_bull_engulf(df: pd.DataFrame) -> pd.Series:
    """Bullish engulfing: up body engulfs previous down body."""
    prev_open, prev_close = df["open"].shift(), df["close"].shift()
    ok = (
        (prev_close < prev_open)
        & (df["close"] > df["open"])
        & (df["close"] >= prev_open)
        & (df["open"] <= prev_close)
    )
    return ok.astype(float).where(prev_open.notna())


@feature("candle_bear_engulf", Family.STRUCTURE, inputs=("open", "close"))
def candle_bear_engulf(df: pd.DataFrame) -> pd.Series:
    """Bearish engulfing: down body engulfs previous up body."""
    prev_open, prev_close = df["open"].shift(), df["close"].shift()
    ok = (
        (prev_close > prev_open)
        & (df["close"] < df["open"])
        & (df["close"] <= prev_open)
        & (df["open"] >= prev_close)
    )
    return ok.astype(float).where(prev_open.notna())


@feature("candle_inside_bar", Family.STRUCTURE, inputs=("high", "low"))
def candle_inside_bar(df: pd.DataFrame) -> pd.Series:
    """Inside bar: today's range contained within yesterday's."""
    ok = (df["high"] < df["high"].shift()) & (df["low"] > df["low"].shift())
    return ok.astype(float).where(df["high"].shift().notna())


@feature("candle_outside_bar", Family.STRUCTURE, inputs=("high", "low"))
def candle_outside_bar(df: pd.DataFrame) -> pd.Series:
    """Outside bar: today's range engulfs yesterday's."""
    ok = (df["high"] > df["high"].shift()) & (df["low"] < df["low"].shift())
    return ok.astype(float).where(df["high"].shift().notna())


@feature("candle_nr7", Family.VOLATILITY, inputs=("high", "low"), min_history=7)
def candle_nr7(df: pd.DataFrame) -> pd.Series:
    """NR7: narrowest high-low range of the last 7 sessions."""
    rng = df["high"] - df["low"]
    return (rng <= rng.rolling(7).min()).astype(float).where(rng.rolling(7).min().notna())
