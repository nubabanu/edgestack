"""Volume and liquidity features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgestack.features.registry import feature
from edgestack.types import Family


@feature("rel_volume_20", Family.VOLUME, inputs=("volume",), min_history=21)
def rel_volume_20(df: pd.DataFrame) -> pd.Series:
    """Volume relative to its trailing 20-session mean."""
    base = df["volume"].rolling(20).mean()
    return df["volume"] / base.where(base > 0)


@feature("volume_z_60", Family.VOLUME, inputs=("volume",), min_history=61)
def volume_z_60(df: pd.DataFrame) -> pd.Series:
    """Volume z-score against its trailing 60 sessions."""
    mean = df["volume"].rolling(60).mean()
    std = df["volume"].rolling(60).std()
    return (df["volume"] - mean) / std.where(std > 0)


@feature("dollar_vol_log_20", Family.VOLUME, inputs=("close", "volume"), min_history=21)
def dollar_vol_log_20(df: pd.DataFrame) -> pd.Series:
    """log10 of average 20-session dollar volume (liquidity proxy)."""
    dv = (df["close"] * df["volume"]).rolling(20).mean()
    return np.log10(dv.where(dv > 0))


@feature("obv_slope_20", Family.VOLUME, inputs=("close", "volume"), min_history=41)
def obv_slope_20(df: pd.DataFrame) -> pd.Series:
    """20-session on-balance-volume change, normalized by average volume."""
    direction = pd.Series(np.sign(df["close"].diff()), index=df.index).fillna(0.0)
    obv = (direction * df["volume"]).cumsum()
    base = df["volume"].rolling(20).mean()
    return (obv - obv.shift(20)) / (20.0 * base.where(base > 0))


@feature("amihud_20", Family.VOLUME, inputs=("close", "volume"), min_history=21)
def amihud_20(df: pd.DataFrame) -> pd.Series:
    """Amihud illiquidity proxy: mean(|ret| / dollar volume), scaled by 1e6."""
    ret = df["close"].pct_change().abs()
    dv = (df["close"] * df["volume"]).where(lambda s: s > 0)
    return (ret / dv).rolling(20).mean() * 1e6


@feature("zero_volume_5", Family.VOLUME, inputs=("volume",), min_history=5)
def zero_volume_5(df: pd.DataFrame) -> pd.Series:
    """1.0 if any of the last 5 sessions traded zero volume (staleness flag)."""
    return (df["volume"] == 0).rolling(5).max().astype(float)
