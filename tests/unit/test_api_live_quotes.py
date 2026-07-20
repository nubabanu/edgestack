from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from edgestack.api.app import create_app
from edgestack.config import EdgeStackConfig
from edgestack.data.live_quotes import (
    FreshnessStatus,
    LiveQuote,
    MarketSession,
    ProviderHealth,
    ProviderHealthReport,
    QuoteBatch,
    QuoteLatency,
    normalize_symbols,
)

NOW = datetime(2026, 7, 20, 14, 0, tzinfo=UTC)


def _live_quote(symbol: str) -> LiveQuote:
    return LiveQuote(
        symbol=symbol,
        price=101.0,
        previous_close=100.0,
        change=1.0,
        change_percent=1.0,
        observed_at=NOW,
        market_session=MarketSession.REGULAR,
        freshness_status=FreshnessStatus.FRESH,
        age_seconds=1.0,
        stale_after_seconds=120.0,
        provider="finnhub",
        feed="FINNHUB_US_EQUITIES",
        latency=QuoteLatency.REALTIME,
    )


class _ApiQuoteService:
    def __init__(self, available: tuple[str, ...]) -> None:
        self.available = available

    def fetch(self, symbols: str) -> QuoteBatch:
        requested = normalize_symbols(symbols)
        quotes = tuple(_live_quote(symbol) for symbol in requested if symbol in self.available)
        return QuoteBatch(
            requested_at=NOW,
            quotes=quotes,
            missing_symbols=tuple(symbol for symbol in requested if symbol not in self.available),
            providers_attempted=("finnhub", "yahoo"),
        )

    def health(self) -> ProviderHealthReport:
        return ProviderHealthReport(
            checked_at=NOW,
            provider_order=("finnhub", "yahoo"),
            providers=(
                ProviderHealth(name="finnhub", configured=True, status="healthy"),
                ProviderHealth(name="yahoo", configured=True, status="ready"),
            ),
            cache_entries=1,
            cache_ttl_seconds=5,
            stale_after_seconds=120,
        )


def test_market_quotes_returns_partial_batch_and_preserves_health_contract() -> None:
    client = TestClient(
        create_app(EdgeStackConfig(), quote_service=_ApiQuoteService(("SPY",)))  # type: ignore[arg-type]
    )

    response = client.get("/market/quotes", params={"symbols": "spy,ACN,spy"})

    assert response.status_code == 200
    assert [quote["symbol"] for quote in response.json()["quotes"]] == ["SPY"]
    assert response.json()["missing_symbols"] == ["ACN"]
    assert response.json()["quotes"][0]["provider"] == "finnhub"
    assert client.get("/health").json() == {"status": "ok"}


def test_market_quotes_rejects_invalid_or_excessive_symbols() -> None:
    client = TestClient(
        create_app(EdgeStackConfig(), quote_service=_ApiQuoteService(("SPY",)))  # type: ignore[arg-type]
    )
    assert client.get("/market/quotes", params={"symbols": "$BAD"}).status_code == 422
    too_many = ",".join(f"S{index}" for index in range(51))
    response = client.get("/market/quotes", params={"symbols": too_many})
    assert response.status_code == 422
    assert "at most 50" in response.json()["detail"]


def test_market_quotes_returns_503_only_when_all_symbols_fail() -> None:
    client = TestClient(
        create_app(EdgeStackConfig(), quote_service=_ApiQuoteService(()))  # type: ignore[arg-type]
    )

    response = client.get("/market/quotes", params={"symbols": "SPY,ACN"})

    assert response.status_code == 503
    assert response.json()["detail"]["missing_symbols"] == ["SPY", "ACN"]


def test_provider_health_is_sanitized_and_does_not_probe_network() -> None:
    client = TestClient(
        create_app(EdgeStackConfig(), quote_service=_ApiQuoteService(("SPY",)))  # type: ignore[arg-type]
    )

    payload = client.get("/market/providers/health").json()

    assert payload["provider_order"] == ["finnhub", "yahoo"]
    assert payload["providers"][0]["status"] == "healthy"
    assert "key" not in str(payload).lower()
    assert "secret" not in str(payload).lower()
