"""Trend features: moving averages, slopes, trend states, directional movement."""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgestack.features.registry import feature
from edgestack.types import Family


def _sma_ratio(df: pd.DataFrame, window: int) -> pd.Series:
    return df["close"] / df["close"].rolling(window).mean() - 1.0


@feature("close_vs_sma20", Family.TREND, min_history=20)
def close_vs_sma20(df: pd.DataFrame) -> pd.Series:
    """Close relative to its 20-session SMA."""
    return _sma_ratio(df, 20)


@feature("close_vs_sma50", Family.TREND, min_history=50)
def close_vs_sma50(df: pd.DataFrame) -> pd.Series:
    """Close relative to its 50-session SMA."""
    return _sma_ratio(df, 50)


@feature("close_vs_sma100", Family.TREND, min_history=100)
def close_vs_sma100(df: pd.DataFrame) -> pd.Series:
    """Close relative to its 100-session SMA."""
    return _sma_ratio(df, 100)


@feature("close_vs_sma200", Family.TREND, min_history=200)
def close_vs_sma200(df: pd.DataFrame) -> pd.Series:
    """Close relative to its 200-session SMA."""
    return _sma_ratio(df, 200)


@feature("above_sma200", Family.TREND, min_history=200)
def above_sma200(df: pd.DataFrame) -> pd.Series:
    """1.0 when the close is above its 200-session SMA."""
    sma = df["close"].rolling(200).mean()
    return (df["close"] > sma).astype(float).where(sma.notna())


@feature("sma50_slope_20", Family.TREND, min_history=70)
def sma50_slope_20(df: pd.DataFrame) -> pd.Series:
    """20-session change of the 50-session SMA (trend slope proxy)."""
    return df["close"].rolling(50).mean().pct_change(20)


@feature("golden_state", Family.TREND, min_history=200)
def golden_state(df: pd.DataFrame) -> pd.Series:
    """1.0 while the 50-SMA is above the 200-SMA."""
    fast = df["close"].rolling(50).mean()
    slow = df["close"].rolling(200).mean()
    return (fast > slow).astype(float).where(slow.notna())


@feature("trend_consistency_60", Family.TREND, min_history=61)
def trend_consistency_60(df: pd.DataFrame) -> pd.Series:
    """Fraction of up sessions over the trailing 60."""
    up = (df["close"].diff() > 0).astype(float)
    return up.rolling(60).mean()


@feature("trend_slope_60", Family.TREND, min_history=61)
def trend_slope_60(df: pd.DataFrame) -> pd.Series:
    """Annualized 60-session log-price slope."""
    values = np.log(df["close"] / df["close"].shift(60)) / 60.0 * 252.0
    return pd.Series(values, index=df.index)


@feature("di_diff_14", Family.TREND, inputs=("high", "low", "close"), min_history=28)
def di_diff_14(df: pd.DataFrame) -> pd.Series:
    """+DI minus -DI (Wilder directional movement, 14 sessions), in [-1, 1]."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    alpha = 1.0 / 14
    atr = tr.ewm(alpha=alpha, min_periods=14).mean()
    plus_di = plus_dm.ewm(alpha=alpha, min_periods=14).mean() / atr
    minus_di = minus_dm.ewm(alpha=alpha, min_periods=14).mean() / atr
    return plus_di - minus_di
