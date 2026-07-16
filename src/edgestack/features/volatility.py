"""Volatility features: ATR, Bollinger, squeeze, Parkinson, vol regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgestack.features.registry import feature
from edgestack.types import Family

ANN = np.sqrt(252.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1.0 / window, min_periods=window).mean()


@feature("natr_14", Family.VOLATILITY, inputs=("high", "low", "close"), min_history=15)
def natr_14(df: pd.DataFrame) -> pd.Series:
    """ATR(14) normalized by the close."""
    return atr(df, 14) / df["close"]


@feature("bb_pctb_20", Family.MEAN_REVERSION, min_history=20)
def bb_pctb_20(df: pd.DataFrame) -> pd.Series:
    """Bollinger %B (20, 2): position of close within the bands."""
    mid = df["close"].rolling(20).mean()
    sd = df["close"].rolling(20).std()
    width = (4.0 * sd).where(sd > 0)
    return (df["close"] - (mid - 2.0 * sd)) / width


@feature("bb_width_20", Family.VOLATILITY, min_history=20)
def bb_width_20(df: pd.DataFrame) -> pd.Series:
    """Bollinger bandwidth (20, 2) relative to the middle band."""
    mid = df["close"].rolling(20).mean()
    sd = df["close"].rolling(20).std()
    return 4.0 * sd / mid


@feature("squeeze_on", Family.VOLATILITY, min_history=120)
def squeeze_on(df: pd.DataFrame) -> pd.Series:
    """1.0 when bandwidth sits in the bottom quintile of its trailing 100 sessions."""
    mid = df["close"].rolling(20).mean()
    sd = df["close"].rolling(20).std()
    width = 4.0 * sd / mid
    threshold = width.rolling(100).quantile(0.2)
    return (width < threshold).astype(float).where(threshold.notna())


@feature("vol_pctile_252", Family.VOLATILITY, min_history=150)
def vol_pctile_252(df: pd.DataFrame) -> pd.Series:
    """Percentile of current realized vol within its trailing year."""
    log_return = pd.Series(np.log(df["close"] / df["close"].shift()), index=df.index)
    rv = log_return.rolling(20).std() * ANN
    return rv.rolling(252, min_periods=126).rank(pct=True)


@feature("parkinson_20", Family.VOLATILITY, inputs=("high", "low"), min_history=21)
def parkinson_20(df: pd.DataFrame) -> pd.Series:
    """Annualized 20-session Parkinson (high-low) volatility."""
    hl = pd.Series(np.log(df["high"] / df["low"]) ** 2, index=df.index)
    return np.sqrt(hl.rolling(20).mean() / (4.0 * np.log(2.0))) * ANN


@feature("vol_ratio_5_60", Family.VOLATILITY, min_history=61)
def vol_ratio_5_60(df: pd.DataFrame) -> pd.Series:
    """Short-term vs medium-term realized vol (expansion > 1, compression < 1)."""
    r = pd.Series(np.log(df["close"] / df["close"].shift()), index=df.index)
    return r.rolling(5).std() / r.rolling(60).std().where(lambda s: s > 0)
