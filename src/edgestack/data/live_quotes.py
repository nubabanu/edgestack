"""Normalized, explicitly indicative live-quote contracts.

These snapshots are for display and paper-only alerts.  They are deliberately
separate from canonical bars, recommendation publication and fill simulation.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,14}$")


class MarketSession(StrEnum):
    PRE = "PRE"
    REGULAR = "REGULAR"
    POST = "POST"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


class FreshnessStatus(StrEnum):
    FRESH = "FRESH"
    STALE = "STALE"
    MARKET_CLOSED = "MARKET_CLOSED"


class QuoteLatency(StrEnum):
    REALTIME = "REALTIME"
    DELAYED = "DELAYED"
    UNKNOWN = "UNKNOWN"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProviderQuote(_FrozenModel):
    """Provider-normalized snapshot before gateway freshness/cache decoration."""

    symbol: str
    price: float = Field(gt=0)
    bid: float | None = Field(default=None, gt=0)
    ask: float | None = Field(default=None, gt=0)
    previous_close: float | None = Field(default=None, gt=0)
    change: float | None = None
    change_percent: float | None = None
    observed_at: datetime
    market_session: MarketSession = MarketSession.UNKNOWN
    provider: str
    feed: str
    latency: QuoteLatency
    warnings: tuple[str, ...] = ()

    @field_validator("symbol")
    @classmethod
    def _valid_symbol(cls, value: str) -> str:
        symbol = value.strip().upper()
        if not _SYMBOL_RE.fullmatch(symbol):
            raise ValueError(f"invalid symbol: {value!r}")
        return symbol

    @field_validator("price", "bid", "ask", "previous_close", "change", "change_percent")
    @classmethod
    def _finite_number(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("quote numbers must be finite")
        return value

    @field_validator("observed_at")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _valid_spread(self) -> ProviderQuote:
        if self.bid is not None and self.ask is not None and self.ask < self.bid:
            raise ValueError("ask cannot be below bid")
        return self


class LiveQuote(_FrozenModel):
    symbol: str
    price: float
    bid: float | None = None
    ask: float | None = None
    previous_close: float | None = None
    change: float | None = None
    change_percent: float | None = None
    observed_at: datetime
    market_session: MarketSession
    freshness_status: FreshnessStatus
    age_seconds: float = Field(ge=0)
    stale_after_seconds: float = Field(gt=0)
    provider: str
    feed: str
    latency: QuoteLatency
    from_cache: bool = False
    warnings: tuple[str, ...] = ()


class QuoteBatch(_FrozenModel):
    requested_at: datetime
    quotes: tuple[LiveQuote, ...]
    missing_symbols: tuple[str, ...] = ()
    providers_attempted: tuple[str, ...] = ()
    note: str = "Indicative quotes only; canonical signals use official closes."


class ProviderHealth(_FrozenModel):
    name: str
    configured: bool
    enabled: bool = True
    status: str
    last_success_at: datetime | None = None
    last_error_category: str | None = None
    cooldown_until: datetime | None = None
    requests: int = 0
    successful_quotes: int = 0
    cache_hits: int = 0


class ProviderHealthReport(_FrozenModel):
    checked_at: datetime
    provider_order: tuple[str, ...]
    providers: tuple[ProviderHealth, ...]
    cache_entries: int
    cache_ttl_seconds: float
    stale_after_seconds: float


def normalize_symbols(
    raw: str | tuple[str, ...] | list[str], *, limit: int = 50
) -> tuple[str, ...]:
    values = raw.split(",") if isinstance(raw, str) else raw
    symbols = tuple(
        dict.fromkeys(str(value).strip().upper() for value in values if str(value).strip())
    )
    if not symbols:
        raise ValueError("at least one symbol is required")
    if len(symbols) > limit:
        raise ValueError(f"at most {limit} symbols are allowed")
    invalid = [symbol for symbol in symbols if not _SYMBOL_RE.fullmatch(symbol)]
    if invalid:
        raise ValueError(f"invalid symbols: {', '.join(invalid)}")
    return symbols
