"""Failover daily price provider: primary feed with per-symbol fallback.

Registered as ``yahoo_stooq`` (Yahoo primary, Stooq fallback). Symbols the
primary cannot serve — or all of them when the primary fails outright — are
re-fetched from the secondary so one flaky feed no longer stalls the nightly
catalog refresh.

Honest limitations (surfaced via ProviderMetadata):
- fallback rows carry ``adj_close = close``: split-adjusted only, missing any
  dividend adjustment inside the fetch window. Consumers would otherwise drop
  NaN-adj_close rows, and the incoming-wins catalog merge would clobber good
  history. The next successful primary fetch restates the window with true
  dividend-adjusted values;
- corporate actions come from the primary only (Stooq has none);
- both underlying feeds are unofficial and survivorship-biased.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.providers.base import PriceDataProvider, ProviderMetadata
from edgestack.data.providers.registry import get_price_provider, register_price_provider
from edgestack.data.schemas import validate_bars
from edgestack.exceptions import ProviderError
from edgestack.logging import get_logger, log_event

log = get_logger("provider.failover")

_EMPTY_ACTIONS = ("symbol", "date", "action_type", "value")


class FailoverPriceProvider(PriceDataProvider):
    """Serve bars from ``primary``, filling per-symbol gaps from ``secondary``."""

    def __init__(self, primary: PriceDataProvider, secondary: PriceDataProvider) -> None:
        self.primary = primary
        self.secondary = secondary
        self.last_corporate_actions = pd.DataFrame(columns=list(_EMPTY_ACTIONS))
        self.last_provider_by_symbol: dict[str, str] = {}
        self.metadata = ProviderMetadata(
            name=f"{primary.metadata.name}+{secondary.metadata.name}",
            kind="price",
            supported_fields=primary.metadata.supported_fields,
            adjustment=primary.metadata.adjustment,
            limitations=(
                f"fallback ({secondary.metadata.name}) rows carry adj_close=close "
                "(split-adjusted only, no dividend adjustment) until the primary restates them",
                f"corporate actions from {primary.metadata.name} only",
                *primary.metadata.limitations,
            ),
        )

    def fetch_daily_bars(self, symbols: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
        primary_name = self.primary.metadata.name
        secondary_name = self.secondary.metadata.name
        self.last_provider_by_symbol = {}
        frames: list[pd.DataFrame] = []

        try:
            primary_bars = self.primary.fetch_daily_bars(symbols, start, end)
        except ProviderError as exc:
            log_event(log, 30, "primary provider failed", provider=primary_name, error=str(exc))
            primary_bars = None
        actions = getattr(self.primary, "last_corporate_actions", None)
        self.last_corporate_actions = (
            actions if actions is not None else pd.DataFrame(columns=list(_EMPTY_ACTIONS))
        )
        if primary_bars is not None and not primary_bars.empty:
            frames.append(primary_bars)
            for symbol in primary_bars["symbol"].astype(str).unique():
                self.last_provider_by_symbol[symbol.upper()] = primary_name

        missing = tuple(s for s in symbols if s.upper() not in self.last_provider_by_symbol)
        if missing:
            try:
                fallback = self.secondary.fetch_daily_bars(missing, start, end)
            except ProviderError as exc:
                log_event(
                    log, 30, "secondary provider failed", provider=secondary_name, error=str(exc)
                )
                fallback = None
            if fallback is not None and not fallback.empty:
                fallback = fallback.copy()
                # NaN adj_close rows would be dropped by recommendation consumers
                # and would overwrite good history in the incoming-wins merge.
                nan_adj = fallback["adj_close"].isna()
                fallback.loc[nan_adj, "adj_close"] = fallback.loc[nan_adj, "close"]
                frames.append(fallback)
                for symbol in fallback["symbol"].astype(str).unique():
                    self.last_provider_by_symbol[symbol.upper()] = secondary_name
                    log_event(
                        log, 30, "fallback provider used", symbol=symbol, provider=secondary_name
                    )

        if not frames:
            raise ProviderError(
                f"neither {primary_name} nor {secondary_name} returned data for any of {symbols!r}"
            )
        return validate_bars(pd.concat(frames, ignore_index=True), context="failover")


@register_price_provider("yahoo_stooq")
def _make_yahoo_stooq(cfg: EdgeStackConfig) -> PriceDataProvider:
    return FailoverPriceProvider(
        get_price_provider("yahoo", cfg),
        get_price_provider("stooq", cfg),
    )
