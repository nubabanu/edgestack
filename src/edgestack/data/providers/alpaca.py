"""Authenticated Alpaca historical bars with delayed free-tier SIP access.

The provider deliberately refuses a range that reaches inside its configured
safety delay. Raw response pages are stored by content hash before parsing;
credentials are used only in headers and never enter artifacts or exceptions.
"""

from __future__ import annotations

import hashlib
import json
import time as time_module
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import atomic_write_bytes
from edgestack.data.provider_credentials import ResearchProviderCredentials
from edgestack.data.providers.base import IntradayDataProvider, PriceDataProvider, ProviderMetadata
from edgestack.data.providers.registry import register_price_provider
from edgestack.data.schemas import validate_bars, validate_intraday_bars
from edgestack.exceptions import ProviderError

_URL = "https://data.alpaca.markets/v2/stocks/bars"
_NY = ZoneInfo("America/New_York")
_TIMEFRAMES = {"1m": ("1Min", 1), "5m": ("5Min", 5), "15m": ("15Min", 15), "60m": ("1Hour", 60)}


class AlpacaHistoricalProvider(PriceDataProvider, IntradayDataProvider):
    def __init__(
        self,
        api_key_id: str | None,
        api_secret_key: str | None,
        *,
        raw_dir: Path,
        timeout: float = 30.0,
        max_retries: int = 3,
        safety_delay_minutes: int = 16,
        session: requests.Session | None = None,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._api_key_id = api_key_id
        self._api_secret_key = api_secret_key
        self.raw_dir = raw_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.safety_delay_minutes = safety_delay_minutes
        self.session = session or requests.Session()
        self._now = now or (lambda: datetime.now(UTC))
        self._sleep = sleep or time_module.sleep
        self.last_raw_hashes: tuple[str, ...] = ()
        self.metadata = ProviderMetadata(
            name="alpaca",
            kind="price",
            supported_fields=("open", "high", "low", "close", "volume", "trade_count", "vwap"),
            frequencies=("1d", "1m", "5m", "15m", "60m"),
            is_point_in_time=True,
            adjustment="split",
            publication_delay=f"{safety_delay_minutes} minutes",
            limitations=(
                "authentication is required",
                "free recent SIP history is delayed; requests stop before the safety cutoff",
                "bars contain eligible trades, not auction imbalance or queue priority",
                "split adjustment can be restated after a later corporate action",
            ),
        )

    @property
    def configured(self) -> bool:
        return bool(self._api_key_id and self._api_secret_key)

    def _headers(self) -> dict[str, str]:
        if not self.configured:
            raise ProviderError(
                "alpaca historical data is unconfigured; set the Alpaca key id and secret"
            )
        return {
            "APCA-API-KEY-ID": str(self._api_key_id),
            "APCA-API-SECRET-KEY": str(self._api_secret_key),
            "Accept": "application/json",
            "User-Agent": "edgestack/0.1 research-paper-only",
        }

    def _range(self, start: date, end: date) -> tuple[datetime, datetime]:
        if end < start:
            raise ProviderError("alpaca end date must not precede start date")
        start_at = datetime.combine(start, time.min, tzinfo=_NY).astimezone(UTC)
        requested_end = datetime.combine(end + timedelta(days=1), time.min, tzinfo=_NY).astimezone(
            UTC
        )
        cutoff = self._now().astimezone(UTC) - timedelta(minutes=self.safety_delay_minutes)
        end_at = min(requested_end, cutoff)
        if end_at <= start_at:
            raise ProviderError(
                f"alpaca range reaches inside the {self.safety_delay_minutes}-minute safety delay"
            )
        return start_at, end_at

    def _request_pages(self, params: dict[str, str]) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        raw_hashes: list[str] = []
        page_token: str | None = None
        while True:
            query = dict(params)
            if page_token:
                query["page_token"] = page_token
            payload = self._request(query)
            pages.append(payload)
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            digest = hashlib.sha256(raw).hexdigest()
            atomic_write_bytes(self.raw_dir / f"{digest}.json", raw)
            raw_hashes.append(digest)
            value = payload.get("next_page_token")
            page_token = str(value) if value else None
            if not page_token:
                self.last_raw_hashes = tuple(raw_hashes)
                return pages

    def _request(self, params: dict[str, str]) -> dict[str, Any]:
        last_category = "unknown"
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    _URL,
                    params=params,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
            except requests.RequestException:
                last_category = "network"
            else:
                if response.status_code in {401, 403}:
                    raise ProviderError("alpaca historical authentication failed")
                if response.status_code == 429:
                    last_category = "rate_limit"
                elif response.status_code >= 500:
                    last_category = "upstream"
                elif response.status_code >= 400:
                    raise ProviderError(
                        f"alpaca historical request failed (HTTP {response.status_code})"
                    )
                else:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise ProviderError("alpaca historical response was malformed") from exc
                    if not isinstance(payload, dict):
                        raise ProviderError("alpaca historical response was not an object")
                    return payload
            if attempt == self.max_retries:
                break
            self._sleep(min(30.0, float(2**attempt)))
        raise ProviderError(f"alpaca historical request failed ({last_category})")

    def _common_params(
        self,
        symbols: tuple[str, ...],
        start: date,
        end: date,
        timeframe: str,
    ) -> dict[str, str]:
        start_at, end_at = self._range(start, end)
        return {
            "symbols": ",".join(dict.fromkeys(symbol.upper() for symbol in symbols)),
            "timeframe": timeframe,
            "start": start_at.isoformat().replace("+00:00", "Z"),
            "end": end_at.isoformat().replace("+00:00", "Z"),
            "limit": "10000",
            "adjustment": "split",
            "asof": end.isoformat(),
            "feed": "sip",
            "sort": "asc",
        }

    @staticmethod
    def _rows(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for page in pages:
            bars = page.get("bars") or {}
            if isinstance(bars, list):
                symbol = str(page.get("symbol", "")).upper()
                rows.extend({"symbol": symbol, **item} for item in bars if isinstance(item, dict))
                continue
            if not isinstance(bars, dict):
                continue
            for symbol, values in bars.items():
                if isinstance(values, list):
                    rows.extend(
                        {"symbol": str(symbol).upper(), **item}
                        for item in values
                        if isinstance(item, dict)
                    )
        return rows

    def fetch_intraday_bars(
        self,
        symbols: tuple[str, ...],
        start: date,
        end: date,
        *,
        interval: str = "1m",
        include_prepost: bool = False,
    ) -> pd.DataFrame:
        if interval not in _TIMEFRAMES:
            raise ProviderError("Alpaca interval must be 1m, 5m, 15m, or 60m")
        timeframe, minutes = _TIMEFRAMES[interval]
        pages = self._request_pages(self._common_params(symbols, start, end, timeframe))
        rows = self._rows(pages)
        if not rows:
            raise ProviderError("alpaca returned no intraday bars")
        frame = pd.DataFrame(rows)
        output = pd.DataFrame(
            {
                "symbol": frame["symbol"],
                "timestamp": pd.to_datetime(frame["t"], utc=True),
                "interval_minutes": minutes,
                "open": frame["o"],
                "high": frame["h"],
                "low": frame["l"],
                "close": frame["c"],
                "volume": frame["v"],
            }
        )
        if not include_prepost:
            local = output["timestamp"].dt.tz_convert(_NY)
            minute = local.dt.hour * 60 + local.dt.minute
            output = output.loc[(minute >= 570) & (minute < 960)]
        return validate_intraday_bars(output.reset_index(drop=True), context="alpaca_intraday")

    def fetch_daily_bars(self, symbols: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
        pages = self._request_pages(self._common_params(symbols, start, end, "1Day"))
        rows = self._rows(pages)
        if not rows:
            raise ProviderError("alpaca returned no daily bars")
        frame = pd.DataFrame(rows)
        session_dates = (
            pd.to_datetime(frame["t"], utc=True)
            .dt.tz_convert(_NY)
            .dt.tz_localize(None)
            .dt.normalize()
        )
        output = pd.DataFrame(
            {
                "symbol": frame["symbol"],
                "date": session_dates,
                "open": frame["o"],
                "high": frame["h"],
                "low": frame["l"],
                "close": frame["c"],
                "volume": frame["v"],
                "adj_close": frame["c"],
            }
        )
        return validate_bars(output, context="alpaca_daily")


@register_price_provider("alpaca")
def _make_alpaca(cfg: EdgeStackConfig) -> PriceDataProvider:
    credentials = ResearchProviderCredentials()
    return AlpacaHistoricalProvider(
        credentials.alpaca_key_id,
        credentials.alpaca_secret_key,
        raw_dir=Path(cfg.paths.data_dir) / "raw" / "alpaca",
        timeout=cfg.data.request_timeout_seconds,
        max_retries=cfg.data.max_retries,
        safety_delay_minutes=cfg.research.alpaca_safety_delay_minutes,
    )
