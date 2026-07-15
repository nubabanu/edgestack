"""Stooq end-of-day price provider (free, no API key).

Endpoint: ``https://stooq.com/q/d/l/?s=<symbol>.us&d1=YYYYMMDD&d2=YYYYMMDD&i=d``
returning CSV ``Date,Open,High,Low,Close,Volume``.

Honest limitations (surfaced via ProviderMetadata and report warnings):
- prices appear split-adjusted but NOT dividend-adjusted; adj_close is left NaN;
- no delisted securities: survivorship-biased by construction;
- no corporate-action or earnings metadata;
- unofficial feed: throttling and occasional gaps are expected.
"""

from __future__ import annotations

import io
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from edgestack.config import EdgeStackConfig
from edgestack.data.providers.base import PriceDataProvider, ProviderMetadata
from edgestack.data.providers.registry import register_price_provider
from edgestack.data.schemas import validate_bars
from edgestack.exceptions import ProviderError
from edgestack.logging import get_logger, log_event

log = get_logger("provider.stooq")

_BASE_URL = "https://stooq.com/q/d/l/"
_POLITE_DELAY_S = 0.4


def parse_stooq_csv(symbol: str, text: str) -> pd.DataFrame:
    """Parse one Stooq CSV response into (unvalidated) canonical columns."""
    body = text.strip()
    if not body or body.lower().startswith("no data") or "<html" in body[:200].lower():
        raise ProviderError(f"stooq returned no data for {symbol}")
    raw = pd.read_csv(io.StringIO(body))
    expected = {"Date", "Open", "High", "Low", "Close"}
    if not expected.issubset(raw.columns):
        missing = expected - set(raw.columns)
        raise ProviderError(f"stooq response for {symbol} missing columns {missing}")
    if "Volume" not in raw.columns:  # very illiquid series omit volume
        raw["Volume"] = 0.0
    out = raw.rename(
        columns={"Date": "date", "Open": "open", "High": "high", "Low": "low",
                 "Close": "close", "Volume": "volume"}
    )[["date", "open", "high", "low", "close", "volume"]]
    out["symbol"] = symbol.upper()
    return out


class StooqProvider(PriceDataProvider):
    def __init__(self, cache_dir: Path, *, timeout: float, max_retries: int,
                 cache_ttl_days: int) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_ttl_s = cache_ttl_days * 86400
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "edgestack-research/0.1"
        self.metadata = ProviderMetadata(
            name="stooq",
            kind="price",
            supported_fields=("open", "high", "low", "close", "volume"),
            adjustment="split",
            limitations=(
                "no dividend adjustment (adj_close unavailable)",
                "no delisted securities: survivorship-biased symbol coverage",
                "no point-in-time universe or corporate-action metadata",
                "unofficial free feed; gaps and throttling possible",
            ),
        )

    def fetch_daily_bars(
        self, symbols: tuple[str, ...], start: date, end: date
    ) -> pd.DataFrame:
        frames = []
        for i, symbol in enumerate(symbols):
            if i:
                time.sleep(_POLITE_DELAY_S)
            try:
                frames.append(self._fetch_symbol(symbol, start, end))
            except ProviderError as exc:
                log_event(log, 30, "symbol skipped", symbol=symbol, error=str(exc))
        if not frames:
            raise ProviderError(f"stooq returned no data for any of {symbols!r}")
        bars = validate_bars(pd.concat(frames, ignore_index=True), context="stooq")
        # Stooq is split-adjusted but not dividend-adjusted; be explicit that
        # a total-return adjusted close is NOT available from this feed.
        bars["adj_close"] = float("nan")
        return bars

    def _fetch_symbol(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        text = self._cached_download(symbol, start, end)
        out = parse_stooq_csv(symbol, text)
        out["date"] = pd.to_datetime(out["date"])
        mask = (out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))
        return out.loc[mask]

    def _cached_download(self, symbol: str, start: date, end: date) -> str:
        cache_file = self.cache_dir / f"{symbol.upper()}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < self.cache_ttl_s:
            return cache_file.read_text(encoding="utf-8")
        params = {
            "s": f"{symbol.lower()}.us",
            "d1": f"{start:%Y%m%d}",
            "d2": f"{end:%Y%m%d}",
            "i": "d",
        }
        text = self._request_with_retries(symbol, params)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text, encoding="utf-8")
        return text

    def _request_with_retries(self, symbol: str, params: dict[str, str]) -> str:
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(_BASE_URL, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    raise ProviderError("rate limited (HTTP 429)")
                resp.raise_for_status()
                return resp.text
            except (requests.RequestException, ProviderError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    log_event(log, 30, "retrying stooq request", symbol=symbol,
                              attempt=attempt + 1, error=str(exc))
                    time.sleep(delay)
                    delay *= 2
        raise ProviderError(f"stooq request failed for {symbol}: {last_error}")


@register_price_provider("stooq")
def _make_stooq(cfg: EdgeStackConfig) -> PriceDataProvider:
    return StooqProvider(
        Path(cfg.paths.data_dir) / "cache" / "stooq",
        timeout=cfg.data.request_timeout_seconds,
        max_retries=cfg.data.max_retries,
        cache_ttl_days=cfg.data.cache_ttl_days,
    )
