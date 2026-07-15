"""Basic price features: returns, gaps, ranges, drawdowns, realized moments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgestack.features.registry import feature
from edgestack.types import Family

ANN = np.sqrt(252.0)


def _log_ret(df: pd.DataFrame) -> pd.Series:
    return np.log(df["close"] / df["close"].shift())


@feature("ret_1d", Family.MOMENTUM, description="1-session close-to-close return")
def ret_1d(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change()


@feature("ret_5d", Family.MOMENTUM, min_history=6)
def ret_5d(df: pd.DataFrame) -> pd.Series:
    """5-session close-to-close return."""
    return df["close"].pct_change(5)


@feature("ret_20d", Family.MOMENTUM, min_history=21)
def ret_20d(df: pd.DataFrame) -> pd.Series:
    """20-session close-to-close return."""
    return df["close"].pct_change(20)


@feature("overnight_gap", Family.STRUCTURE, inputs=("open", "close"))
def overnight_gap(df: pd.DataFrame) -> pd.Series:
    """Close-to-open gap: today's open vs yesterday's close."""
    return df["open"] / df["close"].shift() - 1.0


@feature("intraday_ret", Family.MOMENTUM, inputs=("open", "close"))
def intraday_ret(df: pd.DataFrame) -> pd.Series:
    """Open-to-close return of the session."""
    return df["close"] / df["open"] - 1.0


@feature("range_pct", Family.VOLATILITY, inputs=("high", "low", "close"))
def range_pct(df: pd.DataFrame) -> pd.Series:
    """High-low range as a fraction of the close."""
    return (df["high"] - df["low"]) / df["close"]


@feature("close_loc", Family.STRUCTURE, inputs=("high", "low", "close"))
def close_loc(df: pd.DataFrame) -> pd.Series:
    """Where the close sits inside the day's range (0=low, 1=high)."""
    rng = df["high"] - df["low"]
    return ((df["close"] - df["low"]) / rng.where(rng > 0)).fillna(0.5)


@feature("dist_from_high_252", Family.STRUCTURE, min_history=252)
def dist_from_high_252(df: pd.DataFrame) -> pd.Series:
    """Close relative to the trailing 52-week high (<= 0)."""
    return df["close"] / df["close"].rolling(252).max() - 1.0


@feature("dist_from_low_252", Family.STRUCTURE, min_history=252)
def dist_from_low_252(df: pd.DataFrame) -> pd.Series:
    """Close relative to the trailing 52-week low (>= 0)."""
    return df["close"] / df["close"].rolling(252).min() - 1.0


@feature("realized_vol_20", Family.VOLATILITY, min_history=21)
def realized_vol_20(df: pd.DataFrame) -> pd.Series:
    """Annualized 20-session realized volatility of log returns."""
    return _log_ret(df).rolling(20).std() * ANN


@feature("downside_vol_20", Family.VOLATILITY, min_history=21)
def downside_vol_20(df: pd.DataFrame) -> pd.Series:
    """Annualized 20-session downside deviation."""
    r = _log_ret(df)
    return r.where(r < 0, 0.0).rolling(20).std() * ANN


@feature("skew_60", Family.VOLATILITY, min_history=61)
def skew_60(df: pd.DataFrame) -> pd.Series:
    """60-session rolling skewness of log returns."""
    return _log_ret(df).rolling(60).skew()


@feature("kurt_60", Family.VOLATILITY, min_history=61)
def kurt_60(df: pd.DataFrame) -> pd.Series:
    """60-session rolling excess kurtosis of log returns."""
    return _log_ret(df).rolling(60).kurt()


@feature("vol_of_vol_60", Family.VOLATILITY, min_history=81)
def vol_of_vol_60(df: pd.DataFrame) -> pd.Series:
    """60-session std of the realized-vol series (volatility of volatility)."""
    rv = _log_ret(df).rolling(20).std() * ANN
    return rv.rolling(60).std()
