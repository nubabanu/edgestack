"""Abstract provider interfaces.

Every provider declares :class:`ProviderMetadata` so consumers can see —
without reading vendor docs — what the data can and cannot support:
point-in-time capability, adjustment behavior, publication delay and known
limitations. Free/prototype feeds must be honest about their gaps.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from edgestack.types import UniverseSnapshot


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
        self, symbols: tuple[str, ...], start: date, end: date, *, interval: str = "60m"
    ) -> pd.DataFrame:
        """Return symbol/timestamp/OHLCV bars with timestamps convertible to UTC."""


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
