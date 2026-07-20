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

CORPORATE_ACTION_COLUMNS: tuple[str, ...] = ("symbol", "date", "action_type", "value")
INTRADAY_BAR_COLUMNS: tuple[str, ...] = (
    "symbol",
    "timestamp",
    "interval_minutes",
    "open",
    "high",
    "low",
    "close",
    "volume",
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


def validate_corporate_actions(
    df: pd.DataFrame, *, context: str = "corporate_actions"
) -> pd.DataFrame:
    missing = [column for column in CORPORATE_ACTION_COLUMNS if column not in df.columns]
    if missing:
        raise SchemaError(f"{context}: missing required columns {missing}")
    output = df.loc[:, list(CORPORATE_ACTION_COLUMNS)].copy()
    output["symbol"] = output["symbol"].astype(str).str.upper()
    output["date"] = pd.to_datetime(output["date"]).dt.normalize()
    output["action_type"] = output["action_type"].astype(str).str.lower()
    output["value"] = pd.to_numeric(output["value"], errors="coerce")
    problems: list[str] = []
    if (~output["action_type"].isin({"dividend", "split"})).any():
        problems.append("action_type must be dividend or split")
    if output["value"].isna().any() or (output["value"] <= 0).any():
        problems.append("action values must be finite and positive")
    if output.duplicated(["symbol", "date", "action_type"]).any():
        problems.append("duplicate (symbol, date, action_type) rows")
    if problems:
        raise SchemaError(f"{context}: " + "; ".join(problems))
    return output.sort_values(["symbol", "date", "action_type"]).reset_index(drop=True)


def validate_intraday_bars(df: pd.DataFrame, *, context: str = "intraday_bars") -> pd.DataFrame:
    """Validate hourly/minute OHLCV with timezone-aware timestamps normalized to UTC."""
    missing = [column for column in INTRADAY_BAR_COLUMNS if column not in df.columns]
    if missing:
        raise SchemaError(f"{context}: missing required columns {missing}")
    output = df.loc[:, list(INTRADAY_BAR_COLUMNS)].copy()
    output["symbol"] = output["symbol"].astype(str).str.upper()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    output["interval_minutes"] = pd.to_numeric(output["interval_minutes"], errors="coerce").astype(
        "Int64"
    )
    for column in ("open", "high", "low", "close", "volume"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    problems: list[str] = []
    if (
        output[["timestamp", "interval_minutes", "open", "high", "low", "close", "volume"]]
        .isna()
        .any()
        .any()
    ):
        problems.append("null timestamp, price, or volume")
    if output.duplicated(["symbol", "timestamp", "interval_minutes"]).any():
        problems.append("duplicate (symbol, timestamp, interval_minutes) rows")
    valid = output.dropna()
    if (valid[["open", "high", "low", "close"]] <= 0).any().any():
        problems.append("non-positive prices")
    if (valid["volume"] < 0).any():
        problems.append("negative volume")
    if (~valid["interval_minutes"].isin({1, 5, 15, 60})).any():
        problems.append("interval_minutes must be 1, 5, 15, or 60")
    if (valid["high"] < valid[["open", "close", "low"]].max(axis=1)).any():
        problems.append("high below another price")
    if (valid["low"] > valid[["open", "close", "high"]].min(axis=1)).any():
        problems.append("low above another price")
    if problems:
        raise SchemaError(f"{context}: " + "; ".join(problems))
    return output.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
