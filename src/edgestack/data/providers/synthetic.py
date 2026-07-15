"""Synthetic market generator.

Lives in ``src`` (not tests) because it powers both the statistical acceptance
suite and the offline demo. Design is compositional: a base process (GBM)
plus a list of :class:`Effect` perturbations on the log-return series —
momentum, mean reversion, calendar effects (optionally time-boxed to create
"train-only" edges), and decaying edges.

Everything is deterministic per (seed, symbol): each symbol derives its own
``np.random.Generator`` from a ``SeedSequence`` so adding a symbol never
changes another symbol's path.
"""

from __future__ import annotations

import abc
import zlib
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.calendar import TradingCalendar
from edgestack.data.providers.base import PriceDataProvider, ProviderMetadata
from edgestack.data.providers.registry import register_price_provider
from edgestack.data.schemas import validate_bars

SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class GBM:
    """Geometric Brownian motion base process (annualized parameters)."""

    mu: float = 0.06
    sigma: float = 0.20


class Effect(abc.ABC):
    """A perturbation applied to the per-session log-return drift.

    ``drift`` receives the session index ``t``, the calendar facts row for the
    session, and the history of realized log returns ``returns[:t]``; it
    returns an additive adjustment to the session's expected log return.
    """

    @abc.abstractmethod
    def drift(self, t: int, facts: pd.Series, returns: np.ndarray) -> float: ...


@dataclass(frozen=True)
class Momentum(Effect):
    """Positive autocorrelation: recent returns continue."""

    rho: float = 0.15
    lookback: int = 20

    def drift(self, t: int, facts: pd.Series, returns: np.ndarray) -> float:
        if t < 1:
            return 0.0
        window = returns[max(0, t - self.lookback) : t]
        return self.rho * float(window.mean())


@dataclass(frozen=True)
class MeanReversion(Effect):
    """Ornstein-Uhlenbeck-style pull of log price toward its trend line."""

    kappa: float = 0.05
    mu: float = 0.06  # trend the price reverts toward (annualized)

    def drift(self, t: int, facts: pd.Series, returns: np.ndarray) -> float:
        if t < 1:
            return 0.0
        log_price = float(returns[:t].sum())
        trend = (self.mu / SESSIONS_PER_YEAR) * t
        return -self.kappa * (log_price - trend)


@dataclass(frozen=True)
class CalendarEffect(Effect):
    """Extra drift on sessions where a calendar-facts column is truthy.

    ``active_start``/``active_end`` time-box the effect, which is how the test
    suite injects edges that exist only inside (or only outside) the training
    period.
    """

    facts_column: str = "is_turn_of_month"
    drift_bps: float = 15.0
    active_start: date | None = None
    active_end: date | None = None

    def drift(self, t: int, facts: pd.Series, returns: np.ndarray) -> float:
        session: pd.Timestamp = facts.name  # facts row is indexed by session date
        if self.active_start is not None and session.date() < self.active_start:
            return 0.0
        if self.active_end is not None and session.date() > self.active_end:
            return 0.0
        return self.drift_bps / 1e4 if bool(facts[self.facts_column]) else 0.0


@dataclass(frozen=True)
class DecayingEdge(Effect):
    """A calendar effect whose magnitude halves every ``half_life_sessions``."""

    facts_column: str = "is_monday"
    initial_bps: float = 20.0
    half_life_sessions: int = 250

    def drift(self, t: int, facts: pd.Series, returns: np.ndarray) -> float:
        if not bool(facts[self.facts_column]):
            return 0.0
        return (self.initial_bps / 1e4) * 0.5 ** (t / self.half_life_sessions)


@dataclass(frozen=True)
class SyntheticMarket:
    """Deterministic synthetic daily-bar market on the real exchange calendar."""

    seed: int = 42
    calendar_name: str = "XNYS"
    base: GBM = field(default_factory=GBM)
    effects: tuple[Effect, ...] = ()
    initial_price: float = 100.0
    base_volume: float = 1_000_000.0

    def _symbol_rng(self, symbol: str) -> np.random.Generator:
        # Stable per-symbol stream: crc32 keeps it deterministic across runs
        # and platforms (unlike hash()).
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, zlib.crc32(symbol.encode())])
        )

    def generate(self, symbols: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
        cal = TradingCalendar(self.calendar_name)
        facts = cal.session_facts(start, end)
        frames = [self._generate_symbol(sym, facts) for sym in symbols]
        return validate_bars(pd.concat(frames, ignore_index=True), context="synthetic")

    def _generate_symbol(self, symbol: str, facts: pd.DataFrame) -> pd.DataFrame:
        rng = self._symbol_rng(symbol)
        n = len(facts)
        daily_mu = (self.base.mu - 0.5 * self.base.sigma**2) / SESSIONS_PER_YEAR
        daily_sigma = self.base.sigma / np.sqrt(SESSIONS_PER_YEAR)

        shocks = rng.normal(0.0, daily_sigma, size=n)
        returns = np.empty(n)
        for t in range(n):
            drift = daily_mu
            row = facts.iloc[t]
            for effect in self.effects:
                drift += effect.drift(t, row, returns)
            returns[t] = drift + shocks[t]

        closes = self.initial_price * np.exp(np.cumsum(returns))
        prev_closes = np.concatenate(([self.initial_price], closes[:-1]))

        # Overnight gap and intraday range around the close path.
        gap = rng.normal(0.0, 0.35 * daily_sigma, size=n)
        opens = prev_closes * np.exp(gap)
        hi_ext = np.abs(rng.normal(0.0, 0.5 * daily_sigma, size=n))
        lo_ext = np.abs(rng.normal(0.0, 0.5 * daily_sigma, size=n))
        highs = np.maximum(opens, closes) * np.exp(hi_ext)
        lows = np.minimum(opens, closes) * np.exp(-lo_ext)

        # Volume correlated with absolute return size.
        rel_move = np.abs(returns) / max(daily_sigma, 1e-12)
        volume = self.base_volume * np.exp(rng.normal(0.0, 0.3, size=n)) * (1.0 + rel_move)

        return pd.DataFrame(
            {
                "symbol": symbol,
                "date": facts.index,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": np.round(volume),
                # Synthetic market has no corporate actions: adjusted == raw.
                "adj_close": closes,
            }
        )


class SyntheticPriceProvider(PriceDataProvider):
    """PriceDataProvider facade over :class:`SyntheticMarket`."""

    def __init__(self, market: SyntheticMarket) -> None:
        self.market = market
        self.metadata = ProviderMetadata(
            name="synthetic",
            kind="price",
            supported_fields=("open", "high", "low", "close", "volume", "adj_close"),
            is_point_in_time=True,
            adjustment="split_dividend",
            limitations=("synthetic data: for tests and demos only",),
        )

    def fetch_daily_bars(
        self, symbols: tuple[str, ...], start: date, end: date
    ) -> pd.DataFrame:
        return self.market.generate(symbols, start, end)


@register_price_provider("synthetic")
def _make_synthetic(cfg: EdgeStackConfig) -> PriceDataProvider:
    return SyntheticPriceProvider(SyntheticMarket(seed=cfg.project.random_seed))
