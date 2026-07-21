"""Abstract provider interfaces.

Every provider declares :class:`ProviderMetadata` so consumers can see —
without reading vendor docs — what the data can and cannot support:
point-in-time capability, adjustment behavior, publication delay and known
limitations. Free/prototype feeds must be honest about their gaps.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import pandas as pd

from edgestack.types import UniverseSnapshot

if TYPE_CHECKING:
    from edgestack.data.live_quotes import ProviderQuote

# 16:00 ET is 20:00 UTC in summer, 21:00 UTC in winter; the later bound is
# safe year-round.
_US_CLOSE_UTC_HOUR = 21


def range_cache_fresh(
    mtime_epoch_s: float, end: date, ttl_s: float, *, now_s: float | None = None
) -> bool:
    """True when a cached (start, end) daily-bar response is still trustworthy.

    TTL alone is not enough: a response fetched before session ``end``'s close
    is permanently missing the final bar, and serving it for the whole TTL
    masks fresh data from every later fetch of the same range (observed
    2026-07-20: a pre-close catch-up run poisoned the nightly refresh and
    blocked publication). The cache must postdate the end-date US close
    (~21:00 UTC) AND be within TTL.
    """
    now = time.time() if now_s is None else now_s
    if now - mtime_epoch_s >= ttl_s:
        return False
    close_utc = datetime(end.year, end.month, end.day, _US_CLOSE_UTC_HOUR, tzinfo=UTC)
    return mtime_epoch_s >= close_utc.timestamp()


@dataclass(frozen=True)
class ProviderMetadata:
    name: str
    kind: str
    supported_fields: tuple[str, ...]
    frequencies: tuple[str, ...] = ("1d",)
    is_point_in_time: bool = False
    adjustment: str = "unknown"  # raw | split | split_dividend | unknown
    timezone: str = "America/New_York"
    publication_delay: str = "end_of_day"
    limitations: tuple[str, ...] = field(default_factory=tuple)


class PriceDataProvider(abc.ABC):
    """Daily OHLCV bars."""

    metadata: ProviderMetadata

    @abc.abstractmethod
    def fetch_daily_bars(self, symbols: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
        """Return a canonical bar frame (see :mod:`edgestack.data.schemas`).

        Symbols with no data are simply absent from the result; callers decide
        whether that is an error.
        """


class IntradayDataProvider(abc.ABC):
    """Optional timezone-aware intraday OHLCV capability."""

    metadata: ProviderMetadata

    @abc.abstractmethod
    def fetch_intraday_bars(
        self,
        symbols: tuple[str, ...],
        start: date,
        end: date,
        *,
        interval: str = "60m",
        include_prepost: bool = False,
    ) -> pd.DataFrame:
        """Return symbol/timestamp/OHLCV bars with timestamps convertible to UTC.

        ``include_prepost`` requests extended-hours trades when the provider can
        supply them. Callers must never infer a premarket series when it is false.
        """


class LiveQuoteProvider(abc.ABC):
    """Latest indicative market snapshots; never an execution or signal feed."""

    name: str
    configured: bool

    @abc.abstractmethod
    def fetch_quotes(self, symbols: tuple[str, ...]) -> dict[str, ProviderQuote]:
        """Return the successfully resolved subset keyed by uppercase symbol."""


class UniverseProvider(abc.ABC):
    metadata: ProviderMetadata

    @abc.abstractmethod
    def universe(self, as_of: date) -> UniverseSnapshot: ...


class CorporateActionsProvider(abc.ABC):
    """Splits and dividends. Frame: symbol, date, action_type, value."""

    metadata: ProviderMetadata

    @abc.abstractmethod
    def fetch_actions(self, symbols: tuple[str, ...], start: date, end: date) -> pd.DataFrame: ...


class _UnavailableProvider(abc.ABC):
    """Base for interfaces that ship without a bundled implementation.

    They exist so downstream modules can be written against a stable surface;
    instantiating a concrete provider is a later, opt-in step.
    """

    metadata: ProviderMetadata


class FundamentalDataProvider(_UnavailableProvider):
    @abc.abstractmethod
    def fetch_fundamentals(
        self, symbols: tuple[str, ...], start: date, end: date
    ) -> pd.DataFrame: ...


class EarningsDataProvider(_UnavailableProvider):
    @abc.abstractmethod
    def fetch_earnings_events(
        self, symbols: tuple[str, ...], start: date, end: date
    ) -> pd.DataFrame: ...


class MacroDataProvider(_UnavailableProvider):
    @abc.abstractmethod
    def fetch_series(self, series_ids: tuple[str, ...], start: date, end: date) -> pd.DataFrame: ...


class ShortDataProvider(_UnavailableProvider):
    @abc.abstractmethod
    def fetch_borrow(self, symbols: tuple[str, ...], as_of: date) -> pd.DataFrame: ...


class OptionsDataProvider(_UnavailableProvider):
    @abc.abstractmethod
    def fetch_options_metrics(
        self, symbols: tuple[str, ...], start: date, end: date
    ) -> pd.DataFrame: ...


class SentimentDataProvider(_UnavailableProvider):
    @abc.abstractmethod
    def fetch_sentiment(self, symbols: tuple[str, ...], start: date, end: date) -> pd.DataFrame: ...


class NewsDataProvider(_UnavailableProvider):
    @abc.abstractmethod
    def fetch_news(self, symbols: tuple[str, ...], start: date, end: date) -> pd.DataFrame: ...
