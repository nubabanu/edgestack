"""Momentum and oscillator features."""

from __future__ import annotations

import pandas as pd

from edgestack.features.registry import feature
from edgestack.types import Family


def wilder_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.where(avg_loss > 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # All-gain windows have avg_loss == 0 -> RSI 100.
    return rsi.fillna(100.0).where(avg_gain.notna())


@feature("rsi_14", Family.MEAN_REVERSION, min_history=15)
def rsi_14(df: pd.DataFrame) -> pd.Series:
    """Wilder RSI over 14 sessions (0..100)."""
    return wilder_rsi(df["close"], 14)


@feature("rsi_14_pctile", Family.MEAN_REVERSION, min_history=140)
def rsi_14_pctile(df: pd.DataFrame) -> pd.Series:
    """Percentile of today's RSI within its own trailing 252 sessions."""
    rsi = wilder_rsi(df["close"], 14)
    return rsi.rolling(252, min_periods=126).rank(pct=True)


@feature("macd_hist", Family.MOMENTUM, min_history=35)
def macd_hist(df: pd.DataFrame) -> pd.Series:
    """MACD histogram (12/26/9) normalized by price."""
    fast = df["close"].ewm(span=12, min_periods=12).mean()
    slow = df["close"].ewm(span=26, min_periods=26).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, min_periods=9).mean()
    return (macd - signal) / df["close"]


@feature("macd_state", Family.MOMENTUM, min_history=35)
def macd_state(df: pd.DataFrame) -> pd.Series:
    """1.0 while MACD is above its signal line, else 0.0."""
    fast = df["close"].ewm(span=12, min_periods=12).mean()
    slow = df["close"].ewm(span=26, min_periods=26).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, min_periods=9).mean()
    return (macd > signal).astype(float).where(signal.notna())


@feature("roc_10", Family.MOMENTUM, min_history=11)
def roc_10(df: pd.DataFrame) -> pd.Series:
    """10-session rate of change."""
    return df["close"].pct_change(10)


@feature("mom_60", Family.MOMENTUM, min_history=61)
def mom_60(df: pd.DataFrame) -> pd.Series:
    """60-session momentum."""
    return df["close"].pct_change(60)


@feature("mom_120", Family.MOMENTUM, min_history=121)
def mom_120(df: pd.DataFrame) -> pd.Series:
    """120-session momentum."""
    return df["close"].pct_change(120)


@feature("mom_12_1", Family.MOMENTUM, min_history=253)
def mom_12_1(df: pd.DataFrame) -> pd.Series:
    """Classic 12-minus-1-month momentum (skip the most recent month)."""
    return df["close"].shift(20) / df["close"].shift(252) - 1.0


@feature("stoch_k_14", Family.MEAN_REVERSION, inputs=("high", "low", "close"), min_history=14)
def stoch_k_14(df: pd.DataFrame) -> pd.Series:
    """Stochastic %K over 14 sessions (0..100)."""
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    rng = (high14 - low14).where(lambda s: s > 0)
    return (100.0 * (df["close"] - low14) / rng).clip(0.0, 100.0)
