"""Deterministic intraday aggregation anchored to the New York open."""

from __future__ import annotations

import pandas as pd

from edgestack.data.schemas import validate_intraday_bars
from edgestack.exceptions import ValidationError


def resample_intraday_bars(frame: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """Derive coarser bars from one source resolution without mixing vendors."""
    source = validate_intraday_bars(frame, context="resample_intraday.source")
    source_intervals = set(source["interval_minutes"].astype(int).unique())
    if len(source_intervals) != 1:
        raise ValidationError("intraday resampling requires one source interval")
    source_interval = next(iter(source_intervals))
    if interval_minutes <= source_interval or interval_minutes % source_interval:
        raise ValidationError("target interval must be a larger multiple of the source interval")

    local = source["timestamp"].dt.tz_convert("America/New_York")
    sessions = local.dt.normalize()
    origin = sessions + pd.Timedelta(hours=9, minutes=30)
    elapsed = (local - origin).dt.total_seconds() // (interval_minutes * 60)
    bucket_local = origin + pd.to_timedelta(elapsed * interval_minutes, unit="m")
    working = source.assign(_bucket=bucket_local.dt.tz_convert("UTC"))
    output = (
        working.groupby(["symbol", "_bucket"], sort=True, observed=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .rename(columns={"_bucket": "timestamp"})
    )
    output.insert(2, "interval_minutes", interval_minutes)
    return validate_intraday_bars(output, context="resample_intraday.output")
