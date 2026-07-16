"""Canonical dataframe schemas and structural validation.

The canonical daily-bar panel has one row per (symbol, date):

    symbol      str                 upper-case ticker
    date        datetime64[ns]      session date, tz-naive, normalized
    open/high/low/close  float64    raw (as-traded) prices
    volume      float64             shares
    adj_close   float64             split+dividend adjusted close (NaN allowed)

Raw and adjusted prices are kept separately on purpose: fills are computed on
raw prices, total-return calculations on adjusted ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from edgestack.exceptions import SchemaError

BAR_COLUMNS: tuple[str, ...] = (
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
)

BAR_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("date", pa.timestamp("ns"), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.float64(), nullable=False),
        pa.field("adj_close", pa.float64(), nullable=True),
    ]
)


def validate_bars(df: pd.DataFrame, *, context: str = "bars") -> pd.DataFrame:
    """Structurally validate and normalize a daily-bar frame.

    Returns a normalized copy (canonical column order, sorted by symbol/date).
    Raises :class:`SchemaError` listing every violation found. Deeper,
    content-level checks (gaps, price cliffs, staleness) live in
    :mod:`edgestack.data.quality`.
    """
    problems: list[str] = []

    missing = [c for c in BAR_COLUMNS if c not in df.columns and c != "adj_close"]
    if missing:
        raise SchemaError(f"{context}: missing required columns {missing}")

    out = df.copy()
    if "adj_close" not in out.columns:
        out["adj_close"] = np.nan
    out = out[list(BAR_COLUMNS)]

    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    if getattr(out["date"].dt, "tz", None) is not None:
        problems.append("date column must be tz-naive")

    for col in ("open", "high", "low", "close", "volume", "adj_close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    core = ["open", "high", "low", "close", "volume"]
    n_null = int(out[core].isna().any(axis=1).sum())
    if n_null:
        problems.append(f"{n_null} rows with null price/volume fields")

    dupes = out.duplicated(subset=["symbol", "date"]).sum()
    if dupes:
        problems.append(f"{int(dupes)} duplicate (symbol, date) rows")

    valid = ~out[core].isna().any(axis=1)
    sub = out.loc[valid]
    nonpos = int((sub[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    if nonpos:
        problems.append(f"{nonpos} rows with non-positive prices")
    neg_vol = int((sub["volume"] < 0).sum())
    if neg_vol:
        problems.append(f"{neg_vol} rows with negative volume")

    bad_high = int((sub["high"] < sub[["open", "close", "low"]].max(axis=1)).sum())
    bad_low = int((sub["low"] > sub[["open", "close", "high"]].min(axis=1)).sum())
    if bad_high:
        problems.append(f"{bad_high} rows where high < max(open, close, low)")
    if bad_low:
        problems.append(f"{bad_low} rows where low > min(open, close, high)")

    if problems:
        raise SchemaError(f"{context}: " + "; ".join(problems))

    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def to_arrow(df: pd.DataFrame) -> pa.Table:
    """Convert a validated bar frame to an Arrow table with the canonical schema."""
    return pa.Table.from_pandas(df[list(BAR_COLUMNS)], schema=BAR_SCHEMA, preserve_index=False)
