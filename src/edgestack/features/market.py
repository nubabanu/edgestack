"""Market/benchmark context and cross-sectional features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgestack.features.registry import feature
from edgestack.types import Family

ANN = np.sqrt(252.0)


@feature("bench_trend_200", Family.REGIME, inputs=("bench_close",), min_history=200)
def bench_trend_200(df: pd.DataFrame) -> pd.Series:
    """Benchmark close relative to its 200-session SMA (market trend proxy)."""
    return df["bench_close"] / df["bench_close"].rolling(200).mean() - 1.0


@feature("bench_mom_60", Family.REGIME, inputs=("bench_close",), min_history=61)
def bench_mom_60(df: pd.DataFrame) -> pd.Series:
    """60-session benchmark momentum."""
    return df["bench_close"].pct_change(60)


@feature("bench_vol_20", Family.REGIME, inputs=("bench_close",), min_history=21)
def bench_vol_20(df: pd.DataFrame) -> pd.Series:
    """Annualized 20-session benchmark volatility (market vol proxy)."""
    return np.log(df["bench_close"] / df["bench_close"].shift()).rolling(20).std() * ANN


@feature("bench_drawdown_252", Family.REGIME, inputs=("bench_close",), min_history=252)
def bench_drawdown_252(df: pd.DataFrame) -> pd.Series:
    """Benchmark drawdown from its trailing 52-week high."""
    return df["bench_close"] / df["bench_close"].rolling(252).max() - 1.0


@feature("rel_strength_60", Family.SECTOR, inputs=("close", "bench_close"), min_history=61)
def rel_strength_60(df: pd.DataFrame) -> pd.Series:
    """Stock 60-session momentum minus benchmark momentum."""
    return df["close"].pct_change(60) - df["bench_close"].pct_change(60)


@feature("beta_120", Family.REGIME, inputs=("close", "bench_close"), min_history=121)
def beta_120(df: pd.DataFrame) -> pd.Series:
    """Rolling 120-session beta to the benchmark."""
    r = df["close"].pct_change()
    b = df["bench_close"].pct_change()
    cov = r.rolling(120).cov(b)
    var = b.rolling(120).var()
    return cov / var.where(var > 0)


@feature("corr_120", Family.REGIME, inputs=("close", "bench_close"), min_history=121)
def corr_120(df: pd.DataFrame) -> pd.Series:
    """Rolling 120-session correlation with the benchmark."""
    return df["close"].pct_change().rolling(120).corr(df["bench_close"].pct_change())


# -- cross-sectional (engine feeds one date-slice at a time) -----------------


@feature("cs_rank_mom_60", Family.MOMENTUM, inputs=("mom_60",), cross_sectional=True)
def cs_rank_mom_60(day: pd.DataFrame) -> pd.Series:
    """Cross-sectional percentile rank of 60-session momentum on this date."""
    return day["mom_60"].rank(pct=True)


@feature("cs_rank_vol", Family.VOLATILITY, inputs=("realized_vol_20",), cross_sectional=True)
def cs_rank_vol(day: pd.DataFrame) -> pd.Series:
    """Cross-sectional percentile rank of realized volatility on this date."""
    return day["realized_vol_20"].rank(pct=True)


@feature("cs_rank_rel_volume", Family.VOLUME, inputs=("rel_volume_20",), cross_sectional=True)
def cs_rank_rel_volume(day: pd.DataFrame) -> pd.Series:
    """Cross-sectional percentile rank of relative volume on this date."""
    return day["rel_volume_20"].rank(pct=True)


@feature("breadth_above_sma200", Family.REGIME, inputs=("above_sma200",), cross_sectional=True)
def breadth_above_sma200(day: pd.DataFrame) -> pd.Series:
    """Fraction of the universe above its 200-SMA (breadth), broadcast to all rows."""
    value = day["above_sma200"].mean()
    return pd.Series(value, index=day.index)
