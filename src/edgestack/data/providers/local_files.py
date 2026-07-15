"""Local CSV/Parquet price provider.

Reads user-supplied files from ``<data_dir>/raw/local/<SYMBOL>.csv`` or
``.parquet``. Column names are matched case-insensitively against the
canonical schema plus common vendor spellings (``Adj Close``, ``Date`` ...).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.providers.base import PriceDataProvider, ProviderMetadata
from edgestack.data.providers.registry import register_price_provider
from edgestack.data.schemas import validate_bars
from edgestack.exceptions import ProviderError
from edgestack.logging import get_logger, log_event

log = get_logger("provider.local")

_ALIASES = {
    "date": "date", "open": "open", "high": "high", "low": "low", "close": "close",
    "volume": "volume", "vol": "volume", "adj close": "adj_close",
    "adj_close": "adj_close", "adjclose": "adj_close", "adjusted close": "adj_close",
}


class LocalFilesProvider(PriceDataProvider):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.metadata = ProviderMetadata(
            name="local",
            kind="price",
            supported_fields=("open", "high", "low", "close", "volume", "adj_close"),
            adjustment="unknown",
            limitations=(
                "adjustment state of user-supplied files is unknown",
                "no delisted-security coverage unless the user supplies it",
            ),
        )

    def fetch_daily_bars(
        self, symbols: tuple[str, ...], start: date, end: date
    ) -> pd.DataFrame:
        frames = []
        for symbol in symbols:
            path = self._find(symbol)
            if path is None:
                log_event(log, 30, "no local file for symbol", symbol=symbol)
                continue
            frames.append(self._read(symbol, path))
        if not frames:
            raise ProviderError(f"no local files found under {self.root} for {symbols!r}")
        bars = validate_bars(pd.concat(frames, ignore_index=True), context="local files")
        mask = (bars["date"] >= pd.Timestamp(start)) & (bars["date"] <= pd.Timestamp(end))
        return bars.loc[mask].reset_index(drop=True)

    def _find(self, symbol: str) -> Path | None:
        for ext in (".csv", ".parquet"):
            path = self.root / f"{symbol.upper()}{ext}"
            if path.exists():
                return path
        return None

    def _read(self, symbol: str, path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            raw = pd.read_parquet(path)
        else:
            raw = pd.read_csv(path)
        renames = {}
        for col in raw.columns:
            key = str(col).strip().lower()
            if key in _ALIASES:
                renames[col] = _ALIASES[key]
        raw = raw.rename(columns=renames)
        missing = {"date", "open", "high", "low", "close", "volume"} - set(raw.columns)
        if missing:
            raise ProviderError(f"{path}: missing columns {sorted(missing)}")
        raw["symbol"] = symbol.upper()
        return raw


@register_price_provider("local")
def _make_local(cfg: EdgeStackConfig) -> PriceDataProvider:
    return LocalFilesProvider(Path(cfg.paths.data_dir) / "raw" / "local")
