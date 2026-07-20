"""Provider selection, cache, freshness and health for indicative live quotes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from edgestack.data.live_quotes import (
    FreshnessStatus,
    LiveQuote,
    MarketSession,
    ProviderHealth,
    ProviderHealthReport,
    ProviderQuote,
    QuoteBatch,
    normalize_symbols,
)
from edgestack.data.providers.base import LiveQuoteProvider
from edgestack.data.providers.live import (
    AlpacaQuoteProvider,
    FinnhubQuoteProvider,
    QuoteProviderError,
    TwelveDataQuoteProvider,
    YahooQuoteProvider,
)

_PROVIDER_NAMES = ("finnhub", "twelvedata", "alpaca", "yahoo")


class LiveQuoteSettings(BaseSettings):
    """Runtime-only settings. Secret fields never enter EdgeStackConfig/config hashes."""

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", frozen=True, populate_by_name=True
    )

    providers: str = Field(
        default="finnhub,twelvedata,alpaca,yahoo",
        validation_alias="EDGESTACK_LIVE_QUOTE_PROVIDERS",
    )
    cache_ttl_seconds: float = Field(
        default=5.0, gt=0, validation_alias="EDGESTACK_LIVE_QUOTE_CACHE_TTL_SECONDS"
    )
    stale_after_seconds: float = Field(
        default=120.0, gt=0, validation_alias="EDGESTACK_LIVE_QUOTE_STALE_AFTER_SECONDS"
    )
    request_timeout_seconds: float = Field(
        default=5.0, gt=0, validation_alias="EDGESTACK_LIVE_QUOTE_TIMEOUT_SECONDS"
    )
    cooldown_seconds: float = Field(
        default=60.0, gt=0, validation_alias="EDGESTACK_LIVE_QUOTE_COOLDOWN_SECONDS"
    )
    max_symbols: int = Field(
        default=50, ge=1, le=200, validation_alias="EDGESTACK_LIVE_QUOTE_MAX_SYMBOLS"
    )
    finnhub_api_key: SecretStr | None = Field(
        default=None, validation_alias="EDGESTACK_FINNHUB_API_KEY"
    )
    twelve_data_api_key: SecretStr | None = Field(
        default=None, validation_alias="EDGESTACK_TWELVE_DATA_API_KEY"
    )
    alpaca_api_key_id: SecretStr | None = Field(
        default=None, validation_alias="EDGESTACK_ALPACA_API_KEY_ID"
    )
    alpaca_api_secret_key: SecretStr | None = Field(
        default=None, validation_alias="EDGESTACK_ALPACA_API_SECRET_KEY"
    )

    @field_validator("providers")
    @classmethod
    def _valid_providers(cls, value: str) -> str:
        names = tuple(
            dict.fromkeys(part.strip().lower() for part in value.split(",") if part.strip())
        )
        if not names:
            raise ValueError("at least one live quote provider is required")
        unknown = [name for name in names if name not in _PROVIDER_NAMES]
        if unknown:
            raise ValueError(f"unknown live quote providers: {', '.join(unknown)}")
        return ",".join(names)

    @property
    def provider_order(self) -> tuple[str, ...]:
        return tuple(self.providers.split(","))


@dataclass
class _ProviderState:
    last_success_at: datetime | None = None
    last_error_category: str | None = None
    cooldown_until: datetime | None = None
    requests: int = 0
    successful_quotes: int = 0
    cache_hits: int = 0


@dataclass(frozen=True)
class _CacheEntry:
    quote: ProviderQuote
    cached_at: datetime


class LiveQuoteService:
    def __init__(
        self,
        providers: dict[str, LiveQuoteProvider],
        settings: LiveQuoteSettings,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.providers = providers
        self.settings = settings
        self._now = now or (lambda: datetime.now(UTC))
        self._states = {name: _ProviderState() for name in providers}
        self._cache: dict[tuple[str, str], _CacheEntry] = {}
        self._calendar = xcals.get_calendar("XNYS")

    def fetch(
        self, symbols: str | tuple[str, ...] | list[str], *, provider: str = "auto"
    ) -> QuoteBatch:
        normalized = normalize_symbols(symbols, limit=self.settings.max_symbols)
        requested_at = self._utc_now()
        order = self.settings.provider_order if provider == "auto" else (provider.lower(),)
        unknown = [name for name in order if name not in self.providers]
        if unknown:
            raise ValueError(f"unknown quote provider: {unknown[0]}")

        accepted: dict[str, LiveQuote] = {}
        stale_candidates: dict[str, LiveQuote] = {}
        attempted: list[str] = []
        for name in order:
            source = self.providers[name]
            state = self._states[name]
            if not source.configured or self._in_cooldown(state, requested_at):
                continue
            unresolved = tuple(symbol for symbol in normalized if symbol not in accepted)
            if not unresolved:
                break

            pending: list[str] = []
            for symbol in unresolved:
                cached = self._cached(name, symbol, requested_at)
                if cached is None:
                    pending.append(symbol)
                    continue
                state.cache_hits += 1
                quote = self._decorate(cached, requested_at, from_cache=True)
                self._accept_or_remember(quote, accepted, stale_candidates)

            pending = [symbol for symbol in pending if symbol not in accepted]
            if not pending:
                continue
            attempted.append(name)
            state.requests += 1
            try:
                results = source.fetch_quotes(tuple(pending))
            except QuoteProviderError as exc:
                self._record_failure(state, exc.category, requested_at)
                continue
            except Exception:
                self._record_failure(state, "internal", requested_at)
                continue

            state.last_error_category = None
            state.cooldown_until = None
            if results:
                state.last_success_at = requested_at
                state.successful_quotes += len(results)
            else:
                state.last_error_category = "empty_response"
            for symbol, provider_quote in results.items():
                if symbol not in pending:
                    continue
                self._cache[(name, symbol)] = _CacheEntry(provider_quote, requested_at)
                quote = self._decorate(provider_quote, requested_at, from_cache=False)
                self._accept_or_remember(quote, accepted, stale_candidates)

        for symbol in normalized:
            if symbol not in accepted and symbol in stale_candidates:
                accepted[symbol] = stale_candidates[symbol]
        return QuoteBatch(
            requested_at=requested_at,
            quotes=tuple(accepted[symbol] for symbol in normalized if symbol in accepted),
            missing_symbols=tuple(symbol for symbol in normalized if symbol not in accepted),
            providers_attempted=tuple(attempted),
        )

    def health(self) -> ProviderHealthReport:
        now = self._utc_now()
        rows: list[ProviderHealth] = []
        for name, provider in self.providers.items():
            state = self._states[name]
            in_cooldown = self._in_cooldown(state, now)
            enabled = name in self.settings.provider_order
            if not enabled:
                status = "disabled"
            elif not provider.configured:
                status = "unconfigured"
            elif in_cooldown:
                status = "cooldown"
            elif state.last_error_category:
                status = "error"
            elif state.last_success_at:
                status = "healthy"
            else:
                status = "ready"
            rows.append(
                ProviderHealth(
                    name=name,
                    configured=provider.configured,
                    enabled=enabled,
                    status=status,
                    last_success_at=state.last_success_at,
                    last_error_category=state.last_error_category,
                    cooldown_until=state.cooldown_until if in_cooldown else None,
                    requests=state.requests,
                    successful_quotes=state.successful_quotes,
                    cache_hits=state.cache_hits,
                )
            )
        self._prune_cache(now)
        return ProviderHealthReport(
            checked_at=now,
            provider_order=self.settings.provider_order,
            providers=tuple(rows),
            cache_entries=len(self._cache),
            cache_ttl_seconds=self.settings.cache_ttl_seconds,
            stale_after_seconds=self.settings.stale_after_seconds,
        )

    def _decorate(self, quote: ProviderQuote, now: datetime, *, from_cache: bool) -> LiveQuote:
        age = max((now - quote.observed_at).total_seconds(), 0.0)
        future_skew = (quote.observed_at - now).total_seconds()
        # The exchange calendar is authoritative for the current session. A
        # provider's stale `CLOSED` flag must not make an old quote look valid
        # during a live XNYS session.
        session = self._market_session(now)
        if future_skew > 300:
            freshness = FreshnessStatus.STALE
        elif session is MarketSession.CLOSED:
            freshness = FreshnessStatus.MARKET_CLOSED
        elif age > self.settings.stale_after_seconds:
            freshness = FreshnessStatus.STALE
        else:
            freshness = FreshnessStatus.FRESH
        change = quote.change
        percent = quote.change_percent
        if quote.previous_close and change is None:
            change = quote.price - quote.previous_close
        if quote.previous_close and percent is None:
            percent = (quote.price - quote.previous_close) / quote.previous_close * 100.0
        warnings = quote.warnings
        if future_skew > 300:
            warnings += ("Provider timestamp is more than five minutes ahead of server time.",)
        return LiveQuote(
            **quote.model_dump(exclude={"change", "change_percent", "market_session", "warnings"}),
            change=change,
            change_percent=percent,
            market_session=session,
            freshness_status=freshness,
            age_seconds=round(age, 3),
            stale_after_seconds=self.settings.stale_after_seconds,
            from_cache=from_cache,
            warnings=warnings,
        )

    @staticmethod
    def _accept_or_remember(
        quote: LiveQuote,
        accepted: dict[str, LiveQuote],
        stale_candidates: dict[str, LiveQuote],
    ) -> None:
        if quote.freshness_status is not FreshnessStatus.STALE:
            accepted[quote.symbol] = quote
            return
        previous = stale_candidates.get(quote.symbol)
        if previous is None or quote.observed_at > previous.observed_at:
            stale_candidates[quote.symbol] = quote

    def _cached(self, provider: str, symbol: str, now: datetime) -> ProviderQuote | None:
        entry = self._cache.get((provider, symbol))
        if entry is None:
            return None
        if (now - entry.cached_at).total_seconds() >= self.settings.cache_ttl_seconds:
            self._cache.pop((provider, symbol), None)
            return None
        return entry.quote

    def _prune_cache(self, now: datetime) -> None:
        expired = [
            key
            for key, entry in self._cache.items()
            if (now - entry.cached_at).total_seconds() >= self.settings.cache_ttl_seconds
        ]
        for key in expired:
            self._cache.pop(key, None)

    def _record_failure(self, state: _ProviderState, category: str, now: datetime) -> None:
        state.last_error_category = category
        if category in {"rate_limit", "network", "upstream", "internal"}:
            state.cooldown_until = now + timedelta(seconds=self.settings.cooldown_seconds)

    @staticmethod
    def _in_cooldown(state: _ProviderState, now: datetime) -> bool:
        return state.cooldown_until is not None and state.cooldown_until > now

    def _market_session(self, now: datetime) -> MarketSession:
        eastern = now.astimezone(ZoneInfo("America/New_York"))
        session_label = pd.Timestamp(eastern.date())
        if not self._calendar.is_session(session_label):
            return MarketSession.CLOSED
        market_open = self._calendar.session_open(session_label).to_pydatetime()
        market_close = self._calendar.session_close(session_label).to_pydatetime()
        if market_open <= now <= market_close:
            return MarketSession.REGULAR
        local_time = eastern.timetz().replace(tzinfo=None)
        if time(4) <= local_time < market_open.astimezone(eastern.tzinfo).time():
            return MarketSession.PRE
        if market_close < now and local_time < time(20):
            return MarketSession.POST
        return MarketSession.CLOSED

    def _utc_now(self) -> datetime:
        value = self._now()
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def build_live_quote_service(
    settings: LiveQuoteSettings | None = None,
    *,
    now: Callable[[], datetime] | None = None,
) -> LiveQuoteService:
    resolved = settings or LiveQuoteSettings()
    timeout = resolved.request_timeout_seconds
    providers: dict[str, LiveQuoteProvider] = {
        "finnhub": FinnhubQuoteProvider(resolved.finnhub_api_key, timeout=timeout),
        "twelvedata": TwelveDataQuoteProvider(resolved.twelve_data_api_key, timeout=timeout),
        "alpaca": AlpacaQuoteProvider(
            resolved.alpaca_api_key_id, resolved.alpaca_api_secret_key, timeout=timeout
        ),
        "yahoo": YahooQuoteProvider(timeout=timeout),
    }
    return LiveQuoteService(providers, resolved, now=now)
