"""Bar-schema validation tests."""

from __future__ import annotations

import pandas as pd
import pytest

from edgestack.data.schemas import BAR_COLUMNS, to_arrow, validate_bars
from edgestack.exceptions import SchemaError


def _good_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["aapl", "aapl"],
            "date": ["2023-01-03", "2023-01-04"],
            "open": [130.28, 126.89],
            "high": [130.90, 128.66],
            "low": [124.17, 125.08],
            "close": [125.07, 126.36],
            "volume": [112117500.0, 89113600.0],
        }
    )


def test_validate_normalizes_and_orders() -> None:
    out = validate_bars(_good_frame())
    assert tuple(out.columns) == BAR_COLUMNS
    assert out["symbol"].tolist() == ["AAPL", "AAPL"]
    assert out["date"].dt.tz is None
    assert out["adj_close"].isna().all()  # optional column added as NaN
    table = to_arrow(out)
    assert table.num_rows == 2


def test_rejects_missing_column() -> None:
    with pytest.raises(SchemaError, match="missing required columns"):
        validate_bars(_good_frame().drop(columns=["close"]))


def test_rejects_duplicate_symbol_date() -> None:
    df = pd.concat([_good_frame(), _good_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate"):
        validate_bars(df)


def test_rejects_ohlc_violations() -> None:
    df = _good_frame()
    df.loc[0, "high"] = 1.0  # high below open/close
    with pytest.raises(SchemaError, match="high"):
        validate_bars(df)


def test_rejects_negative_prices_and_volume() -> None:
    df = _good_frame()
    df.loc[0, "low"] = -5.0
    df.loc[1, "volume"] = -1.0
    with pytest.raises(SchemaError):
        validate_bars(df)
