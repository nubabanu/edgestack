"""REST snapshot providers for the indicative live-quote gateway."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import requests
from pydantic import SecretStr, ValidationError

from edgestack.data.live_quotes import MarketSession, ProviderQuote, QuoteLatency
from edgestack.data.providers.base import LiveQuoteProvider


class QuoteProviderError(Exception):
    """Sanitized provider failure safe for logs and health responses."""

    def __init__(self, provider: str, category: str) -> None:
        self.provider = provider
        self.category = category
        super().__init__(f"{provider} quote request failed ({category})")


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in (float("inf"), float("-inf")) else None


def _positive(value: Any) -> float | None:
    parsed = _float(value)
    return parsed if parsed is not None and parsed > 0 else None


def _spread(bid_value: Any, ask_value: Any) -> tuple[float | None, float | None]:
    bid = _positive(bid_value)
    ask = _positive(ask_value)
    return (bid, ask) if bid is None or ask is None or ask >= bid else (None, None)


def _timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) or str(value).replace(".", "", 1).isdigit():
        raw = float(value)
        if raw > 10_000_000_000:
            raw /= 1000.0
        try:
            parsed = datetime.fromtimestamp(raw, tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None
        return parsed if parsed.year >= 2000 else None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    return parsed if parsed.year >= 2000 else None


def _change(price: float, previous_close: float | None) -> tuple[float | None, float | None]:
    if previous_close is None or previous_close <= 0:
        return None, None
    value = price - previous_close
    return value, value / previous_close * 100.0


class _RestProvider(LiveQuoteProvider):
    def __init__(self, *, name: str, configured: bool, timeout: float) -> None:
        self.name = name
        self.configured = configured
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "edgestack/0.1 research-paper-only"

    def _json(self, url: str, **kwargs: Any) -> Any:
        if not self.configured:
            raise QuoteProviderError(self.name, "unconfigured")
        try:
            response = self.session.get(url, timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise QuoteProviderError(self.name, "network") from exc
        if response.status_code in (401, 403):
            raise QuoteProviderError(self.name, "authentication")
        if response.status_code == 429:
            raise QuoteProviderError(self.name, "rate_limit")
        if response.status_code >= 500:
            raise QuoteProviderError(self.name, "upstream")
        if response.status_code >= 400:
            raise QuoteProviderError(self.name, "request")
        try:
            return response.json()
        except ValueError as exc:
            raise QuoteProviderError(self.name, "malformed_response") from exc


class FinnhubQuoteProvider(_RestProvider):
    def __init__(self, api_key: SecretStr | None, *, timeout: float) -> None:
        super().__init__(name="finnhub", configured=bool(api_key), timeout=timeout)
        self._api_key = api_key

    def fetch_quotes(self, symbols: tuple[str, ...]) -> dict[str, ProviderQuote]:
        headers = {"X-Finnhub-Token": self._api_key.get_secret_value()} if self._api_key else {}
        output: dict[str, ProviderQuote] = {}
        for symbol in symbols:
            payload = self._json(
                "https://finnhub.io/api/v1/quote", params={"symbol": symbol}, headers=headers
            )
            price = _float(payload.get("c")) if isinstance(payload, dict) else None
            observed = _timestamp(payload.get("t")) if isinstance(payload, dict) else None
            if price is None or price <= 0 or observed is None:
                continue
            previous = _positive(payload.get("pc"))
            change, percent = _change(price, previous)
            try:
                output[symbol] = ProviderQuote(
                    symbol=symbol,
                    price=price,
                    previous_close=previous,
                    change=_float(payload.get("d")) if payload.get("d") is not None else change,
                    change_percent=(
                        _float(payload.get("dp")) if payload.get("dp") is not None else percent
                    ),
                    observed_at=observed,
                    provider=self.name,
                    feed="FINNHUB_US_EQUITIES",
                    latency=QuoteLatency.REALTIME,
                    warnings=("Vendor feed coverage varies by exchange and entitlement.",),
                )
            except ValidationError:
                continue
        return output


class TwelveDataQuoteProvider(_RestProvider):
    def __init__(self, api_key: SecretStr | None, *, timeout: float) -> None:
        super().__init__(name="twelvedata", configured=bool(api_key), timeout=timeout)
        self._api_key = api_key

    def fetch_quotes(self, symbols: tuple[str, ...]) -> dict[str, ProviderQuote]:
        headers = (
            {"Authorization": f"apikey {self._api_key.get_secret_value()}"} if self._api_key else {}
        )
        output: dict[str, ProviderQuote] = {}
        for symbol in symbols:
            payload = self._json(
                "https://api.twelvedata.com/quote", params={"symbol": symbol}, headers=headers
            )
            if not isinstance(payload, dict) or payload.get("status") == "error":
                continue
            price = _float(payload.get("close"))
            observed = _timestamp(payload.get("timestamp") or payload.get("datetime"))
            if price is None or price <= 0 or observed is None:
                continue
            previous = _positive(payload.get("previous_close"))
            change, percent = _change(price, previous)
            session = (
                MarketSession.REGULAR
                if payload.get("is_market_open") is True
                else MarketSession.UNKNOWN
            )
            supplied_change = _float(payload.get("change"))
            supplied_percent = _float(payload.get("percent_change"))
            bid, ask = _spread(payload.get("bid"), payload.get("ask"))
            try:
                output[symbol] = ProviderQuote(
                    symbol=symbol,
                    price=price,
                    bid=bid,
                    ask=ask,
                    previous_close=previous,
                    change=supplied_change if supplied_change is not None else change,
                    change_percent=supplied_percent if supplied_percent is not None else percent,
                    observed_at=observed,
                    market_session=session,
                    provider=self.name,
                    feed="TWELVE_DATA_BASIC_US",
                    latency=QuoteLatency.REALTIME,
                    warnings=("Free Basic coverage excludes real-time extended-hours data.",),
                )
            except ValidationError:
                continue
        return output


class AlpacaQuoteProvider(_RestProvider):
    def __init__(
        self, api_key_id: SecretStr | None, api_secret_key: SecretStr | None, *, timeout: float
    ) -> None:
        configured = bool(api_key_id and api_secret_key)
        super().__init__(name="alpaca", configured=configured, timeout=timeout)
        self._api_key_id = api_key_id
        self._api_secret_key = api_secret_key

    def fetch_quotes(self, symbols: tuple[str, ...]) -> dict[str, ProviderQuote]:
        headers = {}
        if self._api_key_id and self._api_secret_key:
            headers = {
                "APCA-API-KEY-ID": self._api_key_id.get_secret_value(),
                "APCA-API-SECRET-KEY": self._api_secret_key.get_secret_value(),
            }
        payload = self._json(
            "https://data.alpaca.markets/v2/stocks/snapshots",
            params={"symbols": ",".join(symbols), "feed": "iex"},
            headers=headers,
        )
        if not isinstance(payload, dict):
            return {}
        output: dict[str, ProviderQuote] = {}
        for symbol in symbols:
            item = payload.get(symbol) or payload.get("snapshots", {}).get(symbol)
            if not isinstance(item, dict):
                continue
            trade = item.get("latestTrade") or item.get("latest_trade") or {}
            quote = item.get("latestQuote") or item.get("latest_quote") or {}
            previous_bar = item.get("prevDailyBar") or item.get("previousDailyBar") or {}
            price = _float(trade.get("p"))
            observed = _timestamp(trade.get("t"))
            if price is None or price <= 0 or observed is None:
                continue
            previous = _positive(previous_bar.get("c"))
            change, percent = _change(price, previous)
            bid, ask = _spread(quote.get("bp"), quote.get("ap"))
            try:
                output[symbol] = ProviderQuote(
                    symbol=symbol,
                    price=price,
                    bid=bid,
                    ask=ask,
                    previous_close=previous,
                    change=change,
                    change_percent=percent,
                    observed_at=observed,
                    provider=self.name,
                    feed="IEX_ONLY",
                    latency=QuoteLatency.REALTIME,
                    warnings=(
                        "Free Alpaca market data covers IEX rather than the consolidated SIP.",
                    ),
                )
            except ValidationError:
                continue
        return output


class YahooQuoteProvider(_RestProvider):
    def __init__(self, *, timeout: float) -> None:
        super().__init__(name="yahoo", configured=True, timeout=timeout)

    def fetch_quotes(self, symbols: tuple[str, ...]) -> dict[str, ProviderQuote]:
        payload = self._json(
            "https://query1.finance.yahoo.com/v8/finance/spark",
            params={"symbols": ",".join(symbols), "range": "1d", "interval": "1m"},
        )
        if isinstance(payload, dict) and "spark" not in payload:
            return self._parse_direct_spark(symbols, payload)
        results = (
            ((payload.get("spark") or {}).get("result") or []) if isinstance(payload, dict) else []
        )
        output: dict[str, ProviderQuote] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).upper()
            if symbol not in symbols:
                continue
            response = (item.get("response") or [{}])[0]
            meta = response.get("meta") or {}
            timestamps = response.get("timestamp") or []
            prices = (
                (((response.get("indicators") or {}).get("quote") or [{}])[0]).get("close") or []
            )
            pairs = [
                (ts, price) for ts, price in zip(timestamps, prices) if _float(price) is not None
            ]
            last_ts, last_price = pairs[-1] if pairs else (meta.get("regularMarketTime"), None)
            price = _float(meta.get("regularMarketPrice")) or _float(last_price)
            observed = _timestamp(meta.get("regularMarketTime") or last_ts)
            if price is None or price <= 0 or observed is None:
                continue
            previous = _positive(meta.get("chartPreviousClose") or meta.get("previousClose"))
            change, percent = _change(price, previous)
            state = str(meta.get("marketState", "UNKNOWN")).upper()
            market_session = {
                "PRE": MarketSession.PRE,
                "PREPRE": MarketSession.PRE,
                "REGULAR": MarketSession.REGULAR,
                "POST": MarketSession.POST,
                "POSTPOST": MarketSession.POST,
                "CLOSED": MarketSession.CLOSED,
            }.get(state, MarketSession.UNKNOWN)
            try:
                output[symbol] = ProviderQuote(
                    symbol=symbol,
                    price=price,
                    previous_close=previous,
                    change=change,
                    change_percent=percent,
                    observed_at=observed,
                    market_session=market_session,
                    provider=self.name,
                    feed="UNOFFICIAL_YAHOO",
                    latency=QuoteLatency.UNKNOWN,
                    warnings=(
                        "Unofficial fallback; quotes may be delayed, throttled, or unavailable.",
                    ),
                )
            except ValidationError:
                continue
        return output

    def _parse_direct_spark(
        self, symbols: tuple[str, ...], payload: dict[str, Any]
    ) -> dict[str, ProviderQuote]:
        """Parse Yahoo's current symbol-keyed v8 spark response."""
        output: dict[str, ProviderQuote] = {}
        for symbol in symbols:
            item = payload.get(symbol)
            if not isinstance(item, dict):
                continue
            timestamps = item.get("timestamp") or []
            prices = item.get("close") or []
            pairs = [
                (ts, price) for ts, price in zip(timestamps, prices) if _float(price) is not None
            ]
            if not pairs:
                continue
            last_ts, last_price = pairs[-1]
            price = _float(last_price)
            observed = _timestamp(last_ts)
            previous = _positive(item.get("chartPreviousClose") or item.get("previousClose"))
            if price is None or price <= 0 or observed is None:
                continue
            change, percent = _change(price, previous)
            try:
                output[symbol] = ProviderQuote(
                    symbol=symbol,
                    price=price,
                    previous_close=previous,
                    change=change,
                    change_percent=percent,
                    observed_at=observed,
                    provider=self.name,
                    feed="UNOFFICIAL_YAHOO",
                    latency=QuoteLatency.UNKNOWN,
                    warnings=(
                        "Unofficial fallback; quotes may be delayed, throttled, or unavailable.",
                    ),
                )
            except ValidationError:
                continue
        return output
