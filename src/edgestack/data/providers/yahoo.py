"""Yahoo Finance chart-API price provider (free, unofficial, no API key).

Uses the public JSON chart endpoint directly (no yfinance dependency):
``https://query1.finance.yahoo.com/v8/finance/chart/<symbol>``.

Honest limitations:
- unofficial endpoint: may be throttled, changed or blocked at any time;
- OHLC series is split-adjusted but not dividend-adjusted; the separate
  ``adjclose`` series (split+dividend adjusted) is mapped to ``adj_close``;
- no delisted securities: survivorship-biased coverage;
- history is restated after splits, so stored raw history can shift.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from edgestack.config import EdgeStackConfig
from edgestack.data.providers.base import PriceDataProvider, ProviderMetadata
from edgestack.data.providers.registry import register_price_provider
from edgestack.data.schemas import validate_bars
from edgestack.exceptions import ProviderError
from edgestack.logging import get_logger, log_event

log = get_logger("provider.yahoo")

_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
_POLITE_DELAY_S = 0.3


def parse_chart_payload(symbol: str, payload: dict[str, Any]) -> pd.DataFrame:
    """Parse one Yahoo chart JSON document into canonical bar columns."""
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise ProviderError(f"yahoo error for {symbol}: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise ProviderError(f"yahoo returned no data for {symbol}")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adj = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    if not timestamps or not quote.get("close"):
        raise ProviderError(f"yahoo returned an empty series for {symbol}")

    tz = result.get("meta", {}).get("exchangeTimezoneName", "America/New_York")
    dates = (
        pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    )
    df = pd.DataFrame(
        {
            "symbol": symbol.upper(),
            "date": dates,
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
            "adj_close": adj if adj is not None else float("nan"),
        }
    )
    # Yahoo pads halted/unavailable sessions with nulls: drop them.
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0.0)
    # Occasionally the feed ships a malformed bar (high below close, zero
    # price). Drop those rows rather than poisoning the whole batch.
    ok = (
        (df[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (df["high"] >= df[["open", "close", "low"]].max(axis=1))
        & (df["low"] <= df[["open", "close", "high"]].min(axis=1))
    )
    if (~ok).any():
        log_event(log, 30, "malformed bars dropped", symbol=symbol, count=int((~ok).sum()))
    return df.loc[ok]


def parse_corporate_actions(symbol: str, payload: dict[str, Any]) -> pd.DataFrame:
    results = (payload.get("chart") or {}).get("result") or []
    columns = ["symbol", "date", "action_type", "value"]
    if not results:
        return pd.DataFrame(columns=columns)
    result = results[0]
    timezone = result.get("meta", {}).get("exchangeTimezoneName", "America/New_York")
    events = result.get("events") or {}
    rows: list[dict[str, Any]] = []
    for raw_timestamp, item in (events.get("dividends") or {}).items():
        timestamp = int(item.get("date", raw_timestamp))
        rows.append(
            {
                "symbol": symbol.upper(),
                "date": _event_date(timestamp, timezone),
                "action_type": "dividend",
                "value": float(item["amount"]),
            }
        )
    for raw_timestamp, item in (events.get("splits") or {}).items():
        timestamp = int(item.get("date", raw_timestamp))
        numerator = float(item.get("numerator", 0))
        denominator = float(item.get("denominator", 0))
        ratio = numerator / denominator if numerator > 0 and denominator > 0 else 0.0
        if ratio <= 0 and ":" in str(item.get("splitRatio", "")):
            left, right = str(item["splitRatio"]).split(":", maxsplit=1)
            ratio = float(left) / float(right)
        if ratio > 0:
            rows.append(
                {
                    "symbol": symbol.upper(),
                    "date": _event_date(timestamp, timezone),
                    "action_type": "split",
                    "value": ratio,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _event_date(timestamp: int, timezone: str) -> pd.Timestamp:
    return (
        pd.to_datetime(timestamp, unit="s", utc=True)
        .tz_convert(timezone)
        .tz_localize(None)
        .normalize()
    )


class YahooProvider(PriceDataProvider):
    def __init__(
        self, cache_dir: Path, *, timeout: float, max_retries: int, cache_ttl_days: int
    ) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_ttl_s = cache_ttl_days * 86400
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0 (research; edgestack/0.1)"
        self.last_corporate_actions = pd.DataFrame(
            columns=["symbol", "date", "action_type", "value"]
        )
        self.metadata = ProviderMetadata(
            name="yahoo",
            kind="price",
            supported_fields=("open", "high", "low", "close", "volume", "adj_close"),
            adjustment="split",
            limitations=(
                "unofficial endpoint; may be throttled or changed without notice",
                "OHLC split-adjusted only; adj_close is split+dividend adjusted",
                "no delisted securities: survivorship-biased symbol coverage",
                "no point-in-time universe membership",
            ),
        )

    def fetch_daily_bars(self, symbols: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
        frames = []
        action_frames = []
        for i, symbol in enumerate(symbols):
            if i:
                time.sleep(_POLITE_DELAY_S)
            try:
                bars, actions = self._fetch_symbol(symbol, start, end)
                frames.append(bars)
                action_frames.append(actions)
            except ProviderError as exc:
                log_event(log, 30, "symbol skipped", symbol=symbol, error=str(exc))
        if not frames:
            raise ProviderError(f"yahoo returned no data for any of {symbols!r}")
        self.last_corporate_actions = (
            pd.concat(action_frames, ignore_index=True)
            if action_frames
            else self.last_corporate_actions.iloc[0:0].copy()
        )
        raw = pd.concat(frames, ignore_index=True)
        # Re-apply the malformed-bar guard: cached responses may predate it.
        ok = (
            (raw[["open", "high", "low", "close"]] > 0).all(axis=1)
            & (raw["high"] >= raw[["open", "close", "low"]].max(axis=1))
            & (raw["low"] <= raw[["open", "close", "high"]].min(axis=1))
        )
        if (~ok).any():
            log_event(log, 30, "malformed cached bars dropped", count=int((~ok).sum()))
        bars = validate_bars(raw.loc[ok], context="yahoo")
        mask = (bars["date"] >= pd.Timestamp(start)) & (bars["date"] <= pd.Timestamp(end))
        return bars.loc[mask].reset_index(drop=True)

    def _fetch_symbol(
        self, symbol: str, start: date, end: date
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        cache_file = self.cache_dir / f"{symbol.upper()}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
        actions_file = cache_file.with_name(f"{cache_file.stem}_actions.parquet")
        if (
            cache_file.exists()
            and actions_file.exists()
            and (time.time() - cache_file.stat().st_mtime) < self.cache_ttl_s
        ):
            return pd.read_parquet(cache_file), pd.read_parquet(actions_file)
        params = {
            "period1": str(int(pd.Timestamp(start, tz="UTC").timestamp())),
            # +1 day: period2 is exclusive of the final session otherwise.
            "period2": str(int(pd.Timestamp(end + timedelta(days=1), tz="UTC").timestamp())),
            "interval": "1d",
            "events": "div,splits",
        }
        payload = self._request_with_retries(symbol, params)
        df = parse_chart_payload(symbol, payload)
        actions = parse_corporate_actions(symbol, payload)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)
        actions.to_parquet(actions_file, index=False)
        return df, actions

    def _request_with_retries(self, symbol: str, params: dict[str, str]) -> dict[str, Any]:
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(
                    _BASE_URL + symbol.upper(), params=params, timeout=self.timeout
                )
                if resp.status_code == 429:
                    raise ProviderError("rate limited (HTTP 429)")
                if resp.status_code == 404:
                    raise ProviderError(f"unknown symbol {symbol}")
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError, ProviderError) as exc:
                last_error = exc
                if isinstance(exc, ProviderError) and "unknown symbol" in str(exc):
                    break
                if attempt < self.max_retries:
                    log_event(
                        log,
                        30,
                        "retrying yahoo request",
                        symbol=symbol,
                        attempt=attempt + 1,
                        error=str(exc),
                    )
                    time.sleep(delay)
                    delay *= 2
        raise ProviderError(f"yahoo request failed for {symbol}: {last_error}")


@register_price_provider("yahoo")
def _make_yahoo(cfg: EdgeStackConfig) -> PriceDataProvider:
    return YahooProvider(
        Path(cfg.paths.data_dir) / "cache" / "yahoo",
        timeout=cfg.data.request_timeout_seconds,
        max_retries=cfg.data.max_retries,
        cache_ttl_days=cfg.data.cache_ttl_days,
    )
