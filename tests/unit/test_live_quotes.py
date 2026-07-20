from __future__ import annotations

from argparse import Namespace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import SecretStr
from scripts import agent_toolkit

from edgestack.data.live_quotes import (
    FreshnessStatus,
    MarketSession,
    ProviderQuote,
    QuoteLatency,
    normalize_symbols,
)
from edgestack.data.providers.live import (
    AlpacaQuoteProvider,
    FinnhubQuoteProvider,
    QuoteProviderError,
    TwelveDataQuoteProvider,
    YahooQuoteProvider,
)
from edgestack.data.quote_service import LiveQuoteService, LiveQuoteSettings

NOW = datetime(2026, 7, 20, 14, 0, tzinfo=UTC)  # 10:00 ET, an XNYS session.


class _Response:
    def __init__(self, payload: Any, *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        if isinstance(self.payload, ValueError):
            raise self.payload
        return self.payload


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.headers: dict[str, str] = {}
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class _StubProvider:
    def __init__(
        self,
        name: str,
        quotes: dict[str, ProviderQuote] | None = None,
        *,
        configured: bool = True,
        error: QuoteProviderError | None = None,
    ) -> None:
        self.name = name
        self.configured = configured
        self.quotes = quotes or {}
        self.error = error
        self.calls: list[tuple[str, ...]] = []

    def fetch_quotes(self, symbols: tuple[str, ...]) -> dict[str, ProviderQuote]:
        self.calls.append(symbols)
        if self.error:
            raise self.error
        return {symbol: self.quotes[symbol] for symbol in symbols if symbol in self.quotes}


def _quote(
    symbol: str,
    *,
    provider: str,
    observed_at: datetime = NOW - timedelta(seconds=10),
    session: MarketSession = MarketSession.UNKNOWN,
    price: float = 101.0,
) -> ProviderQuote:
    return ProviderQuote(
        symbol=symbol,
        price=price,
        previous_close=100.0,
        observed_at=observed_at,
        market_session=session,
        provider=provider,
        feed=f"{provider.upper()}_TEST",
        latency=QuoteLatency.REALTIME,
    )


def _settings(**overrides: Any) -> LiveQuoteSettings:
    values = {
        "providers": "finnhub,twelvedata,alpaca,yahoo",
        "cache_ttl_seconds": 5.0,
        "stale_after_seconds": 120.0,
        "request_timeout_seconds": 1.0,
        "cooldown_seconds": 60.0,
    }
    values.update(overrides)
    return LiveQuoteSettings(**values)


def test_normalize_symbols_uppercases_deduplicates_and_limits() -> None:
    assert normalize_symbols(" spy,ACN,spy ") == ("SPY", "ACN")
    with pytest.raises(ValueError, match="invalid symbols"):
        normalize_symbols("SPY,$BAD")
    with pytest.raises(ValueError, match="at most 1"):
        normalize_symbols("SPY,ACN", limit=1)


def test_runtime_settings_load_exact_environment_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGESTACK_LIVE_QUOTE_PROVIDERS", "alpaca,yahoo")
    monkeypatch.setenv("EDGESTACK_FINNHUB_API_KEY", "hidden")
    settings = LiveQuoteSettings(_env_file=None)

    assert settings.provider_order == ("alpaca", "yahoo")
    assert settings.finnhub_api_key is not None
    assert settings.finnhub_api_key.get_secret_value() == "hidden"
    assert "hidden" not in repr(settings)


def test_finnhub_parser_uses_header_and_normalizes_quote() -> None:
    provider = FinnhubQuoteProvider(SecretStr("top-secret"), timeout=1)
    session = _Session(
        [_Response({"c": 101.0, "pc": 100.0, "d": 1.0, "dp": 1.0, "t": 1_752_500_000})]
    )
    provider.session = session  # type: ignore[assignment]

    quote = provider.fetch_quotes(("SPY",))["SPY"]

    assert quote.price == 101.0
    assert quote.feed == "FINNHUB_US_EQUITIES"
    assert session.calls[0][1]["headers"] == {"X-Finnhub-Token": "top-secret"}
    assert "top-secret" not in str(session.calls[0][1]["params"])


def test_twelve_data_parser_keeps_zero_change_and_authorization_header() -> None:
    provider = TwelveDataQuoteProvider(SecretStr("twelve-secret"), timeout=1)
    session = _Session(
        [
            _Response(
                {
                    "symbol": "SPY",
                    "close": "100.0",
                    "previous_close": "100.0",
                    "change": "0.0",
                    "percent_change": "0.0",
                    "timestamp": 1_752_500_000,
                    "is_market_open": True,
                }
            )
        ]
    )
    provider.session = session  # type: ignore[assignment]

    quote = provider.fetch_quotes(("SPY",))["SPY"]

    assert quote.change == 0.0
    assert quote.change_percent == 0.0
    assert quote.market_session is MarketSession.REGULAR
    assert session.calls[0][1]["headers"] == {"Authorization": "apikey twelve-secret"}


def test_alpaca_parser_labels_iex_and_uses_batched_snapshot() -> None:
    provider = AlpacaQuoteProvider(SecretStr("key-id"), SecretStr("secret"), timeout=1)
    session = _Session(
        [
            _Response(
                {
                    "SPY": {
                        "latestTrade": {"p": 101.0, "t": "2026-07-20T13:59:55Z"},
                        "latestQuote": {"bp": 100.99, "ap": 101.01},
                        "prevDailyBar": {"c": 100.0},
                    }
                }
            )
        ]
    )
    provider.session = session  # type: ignore[assignment]

    quote = provider.fetch_quotes(("SPY",))["SPY"]

    assert quote.feed == "IEX_ONLY"
    assert quote.bid == 100.99
    assert quote.ask == 101.01
    assert session.calls[0][1]["params"] == {"symbols": "SPY", "feed": "iex"}


def test_yahoo_parser_is_an_unofficial_no_key_fallback() -> None:
    provider = YahooQuoteProvider(timeout=1)
    session = _Session(
        [
            _Response(
                {
                    "spark": {
                        "result": [
                            {
                                "symbol": "SPY",
                                "response": [
                                    {
                                        "meta": {
                                            "regularMarketPrice": 101.0,
                                            "chartPreviousClose": 100.0,
                                            "regularMarketTime": 1_752_500_000,
                                            "marketState": "REGULAR",
                                        },
                                        "timestamp": [1_752_500_000],
                                        "indicators": {"quote": [{"close": [101.0]}]},
                                    }
                                ],
                            }
                        ]
                    }
                }
            )
        ]
    )
    provider.session = session  # type: ignore[assignment]

    quote = provider.fetch_quotes(("SPY",))["SPY"]

    assert provider.configured is True
    assert quote.feed == "UNOFFICIAL_YAHOO"
    assert quote.latency is QuoteLatency.UNKNOWN


def test_yahoo_parser_accepts_current_symbol_keyed_spark_shape() -> None:
    provider = YahooQuoteProvider(timeout=1)
    provider.session = _Session(  # type: ignore[assignment]
        [
            _Response(
                {
                    "SPY": {
                        "symbol": "SPY",
                        "timestamp": [1_752_500_000, 1_752_500_060],
                        "close": [100.5, 101.0],
                        "previousClose": 100.0,
                    }
                }
            )
        ]
    )

    quote = provider.fetch_quotes(("SPY",))["SPY"]

    assert quote.price == 101.0
    assert quote.previous_close == 100.0
    assert quote.observed_at == datetime.fromtimestamp(1_752_500_060, tz=UTC)


@pytest.mark.parametrize(
    ("status_code", "category"),
    [(401, "authentication"), (403, "authentication"), (429, "rate_limit"), (503, "upstream")],
)
def test_http_failures_are_sanitized(status_code: int, category: str) -> None:
    provider = FinnhubQuoteProvider(SecretStr("never-leak"), timeout=1)
    provider.session = _Session([_Response({}, status_code=status_code)])  # type: ignore[assignment]

    with pytest.raises(QuoteProviderError) as caught:
        provider.fetch_quotes(("SPY",))

    assert caught.value.category == category
    assert "never-leak" not in str(caught.value)


def test_malformed_json_and_missing_quote_fields_fail_safely() -> None:
    malformed = FinnhubQuoteProvider(SecretStr("never-leak"), timeout=1)
    malformed.session = _Session([_Response(ValueError("bad JSON"))])  # type: ignore[assignment]
    with pytest.raises(QuoteProviderError, match="malformed_response"):
        malformed.fetch_quotes(("SPY",))

    missing = FinnhubQuoteProvider(SecretStr("never-leak"), timeout=1)
    missing.session = _Session([_Response({"c": 0, "t": None})])  # type: ignore[assignment]
    assert missing.fetch_quotes(("SPY",)) == {}

    invalid_time = FinnhubQuoteProvider(SecretStr("never-leak"), timeout=1)
    invalid_time.session = _Session(  # type: ignore[assignment]
        [_Response({"c": 100.0, "pc": 99.0, "t": 0})]
    )
    assert invalid_time.fetch_quotes(("SPY",)) == {}


def test_service_skips_unconfigured_and_fills_symbols_from_later_providers() -> None:
    finnhub = _StubProvider("finnhub", configured=False)
    twelve = _StubProvider("twelvedata", {"SPY": _quote("SPY", provider="twelvedata")})
    alpaca = _StubProvider("alpaca", {"ACN": _quote("ACN", provider="alpaca")})
    yahoo = _StubProvider("yahoo")
    service = LiveQuoteService(
        {"finnhub": finnhub, "twelvedata": twelve, "alpaca": alpaca, "yahoo": yahoo},
        _settings(),
        now=lambda: NOW,
    )

    result = service.fetch("spy,acn")

    assert [(quote.symbol, quote.provider) for quote in result.quotes] == [
        ("SPY", "twelvedata"),
        ("ACN", "alpaca"),
    ]
    assert result.providers_attempted == ("twelvedata", "alpaca")
    assert finnhub.calls == []


def test_service_falls_through_stale_quote_but_returns_it_if_no_fresh_source() -> None:
    stale = _quote("SPY", provider="finnhub", observed_at=NOW - timedelta(minutes=10))
    fresh = _quote("SPY", provider="twelvedata")
    service = LiveQuoteService(
        {
            "finnhub": _StubProvider("finnhub", {"SPY": stale}),
            "twelvedata": _StubProvider("twelvedata", {"SPY": fresh}),
            "alpaca": _StubProvider("alpaca"),
            "yahoo": _StubProvider("yahoo"),
        },
        _settings(),
        now=lambda: NOW,
    )
    assert service.fetch("SPY").quotes[0].provider == "twelvedata"

    stale_only = LiveQuoteService(
        {"finnhub": _StubProvider("finnhub", {"SPY": stale})},
        _settings(providers="finnhub"),
        now=lambda: NOW,
    ).fetch("SPY")
    assert stale_only.quotes[0].freshness_status is FreshnessStatus.STALE


def test_service_cache_and_health_are_observable_without_secrets() -> None:
    provider = _StubProvider("finnhub", {"SPY": _quote("SPY", provider="finnhub")})
    service = LiveQuoteService(
        {"finnhub": provider}, _settings(providers="finnhub"), now=lambda: NOW
    )

    assert service.fetch("SPY").quotes[0].from_cache is False
    assert service.fetch("SPY").quotes[0].from_cache is True
    health = service.health()

    assert provider.calls == [("SPY",)]
    assert health.providers[0].status == "healthy"
    assert health.providers[0].cache_hits == 1
    assert "secret" not in health.model_dump_json().lower()


def test_service_refetches_after_cache_expiry() -> None:
    clock = [NOW]
    provider = _StubProvider("finnhub", {"SPY": _quote("SPY", provider="finnhub")})
    service = LiveQuoteService(
        {"finnhub": provider},
        _settings(providers="finnhub", cache_ttl_seconds=5),
        now=lambda: clock[0],
    )

    service.fetch("SPY")
    clock[0] += timedelta(seconds=6)
    service.fetch("SPY")

    assert provider.calls == [("SPY",), ("SPY",)]


def test_closed_market_quote_is_not_rejected_for_wall_clock_age() -> None:
    sunday = datetime(2026, 7, 19, 16, tzinfo=UTC)
    old = _quote(
        "SPY",
        provider="finnhub",
        observed_at=sunday - timedelta(hours=48),
        session=MarketSession.UNKNOWN,
    )
    later = _StubProvider("twelvedata", {"SPY": _quote("SPY", provider="twelvedata")})
    service = LiveQuoteService(
        {
            "finnhub": _StubProvider("finnhub", {"SPY": old}),
            "twelvedata": later,
        },
        _settings(providers="finnhub,twelvedata"),
        now=lambda: sunday,
    )

    result = service.fetch("SPY")

    assert result.quotes[0].freshness_status is FreshnessStatus.MARKET_CLOSED
    assert later.calls == []


def test_rate_limit_opens_sanitized_cooldown_and_falls_through() -> None:
    limited = _StubProvider("finnhub", error=QuoteProviderError("finnhub", "rate_limit"))
    fallback = _StubProvider("yahoo", {"SPY": _quote("SPY", provider="yahoo")})
    service = LiveQuoteService(
        {"finnhub": limited, "yahoo": fallback},
        _settings(providers="finnhub,yahoo"),
        now=lambda: NOW,
    )

    result = service.fetch("SPY")
    health = service.health()

    assert result.quotes[0].provider == "yahoo"
    assert health.providers[0].status == "cooldown"
    assert health.providers[0].last_error_category == "rate_limit"


def test_agent_toolkit_quote_forwards_forced_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quote = LiveQuoteService(
        {"alpaca": _StubProvider("alpaca", {"SPY": _quote("SPY", provider="alpaca")})},
        _settings(providers="alpaca"),
        now=lambda: NOW,
    )
    monkeypatch.setattr("edgestack.data.quote_service.build_live_quote_service", lambda: quote)

    result = agent_toolkit.cmd_quote(Namespace(symbols="spy", provider="alpaca"))

    assert result["quotes"][0]["symbol"] == "SPY"
    assert result["quotes"][0]["provider"] == "alpaca"
    assert result["providers_attempted"] == ["alpaca"]


def test_agent_toolkit_quote_reports_forced_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = LiveQuoteService(
        {"alpaca": _StubProvider("alpaca")},
        _settings(providers="alpaca"),
        now=lambda: NOW,
    )
    monkeypatch.setattr("edgestack.data.quote_service.build_live_quote_service", lambda: service)

    with pytest.raises(RuntimeError, match="attempted: alpaca"):
        agent_toolkit.cmd_quote(Namespace(symbols="SPY", provider="alpaca"))
