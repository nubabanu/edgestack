"""Failover provider tests (no network: stub providers only)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from edgestack.config import EdgeStackConfig
from edgestack.data.providers.base import PriceDataProvider, ProviderMetadata
from edgestack.data.providers.failover import FailoverPriceProvider
from edgestack.data.providers.registry import get_price_provider
from edgestack.exceptions import ProviderError

START = date(2023, 1, 2)
END = date(2023, 1, 6)


def _bars(symbol: str, *, adj_close: float | None = 10.4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": [pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04")],
            "open": [10.0, 10.5],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [1000.0, 1500.0],
            "adj_close": [adj_close, adj_close] if adj_close is not None else float("nan"),
        }
    )


class StubProvider(PriceDataProvider):
    def __init__(
        self,
        name: str,
        serves: dict[str, pd.DataFrame],
        *,
        error: bool = False,
        actions: pd.DataFrame | None = None,
    ) -> None:
        self.serves = serves
        self.error = error
        self.calls: list[tuple[str, ...]] = []
        self.metadata = ProviderMetadata(
            name=name, kind="price", supported_fields=("open", "high", "low", "close", "volume")
        )
        if actions is not None:
            self.last_corporate_actions = actions

    def fetch_daily_bars(self, symbols: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
        self.calls.append(symbols)
        if self.error:
            raise ProviderError(f"{self.metadata.name} is down")
        frames = [self.serves[s] for s in symbols if s in self.serves]
        if not frames:
            raise ProviderError(f"{self.metadata.name} returned no data for any of {symbols!r}")
        return pd.concat(frames, ignore_index=True)


def test_primary_serves_all_secondary_untouched() -> None:
    primary = StubProvider("prim", {"AAA": _bars("AAA"), "BBB": _bars("BBB")})
    secondary = StubProvider("sec", {"AAA": _bars("AAA")})
    provider = FailoverPriceProvider(primary, secondary)
    bars = provider.fetch_daily_bars(("AAA", "BBB"), START, END)
    assert set(bars["symbol"]) == {"AAA", "BBB"}
    assert secondary.calls == []
    assert provider.last_provider_by_symbol == {"AAA": "prim", "BBB": "prim"}


def test_partial_miss_filled_by_secondary_with_adj_close_repair() -> None:
    primary = StubProvider("prim", {"AAA": _bars("AAA")})
    secondary = StubProvider("sec", {"BBB": _bars("BBB", adj_close=None)})
    provider = FailoverPriceProvider(primary, secondary)
    bars = provider.fetch_daily_bars(("AAA", "BBB"), START, END)
    assert set(bars["symbol"]) == {"AAA", "BBB"}
    assert secondary.calls == [("BBB",)]
    fallback = bars[bars["symbol"] == "BBB"]
    assert fallback["adj_close"].tolist() == fallback["close"].tolist()
    # Primary rows keep their true dividend-adjusted values.
    assert bars[bars["symbol"] == "AAA"]["adj_close"].tolist() == [10.4, 10.4]
    assert provider.last_provider_by_symbol == {"AAA": "prim", "BBB": "sec"}


def test_primary_failure_falls_back_entirely() -> None:
    primary = StubProvider("prim", {}, error=True)
    secondary = StubProvider("sec", {"AAA": _bars("AAA", adj_close=None)})
    provider = FailoverPriceProvider(primary, secondary)
    bars = provider.fetch_daily_bars(("AAA",), START, END)
    assert set(bars["symbol"]) == {"AAA"}
    assert not bars["adj_close"].isna().any()
    assert provider.last_provider_by_symbol == {"AAA": "sec"}


def test_both_fail_raises_provider_error() -> None:
    provider = FailoverPriceProvider(
        StubProvider("prim", {}, error=True), StubProvider("sec", {}, error=True)
    )
    with pytest.raises(ProviderError, match="neither prim nor sec"):
        provider.fetch_daily_bars(("AAA",), START, END)


def test_corporate_actions_come_from_primary_only() -> None:
    actions = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "date": pd.Timestamp("2023-01-03"),
                "action_type": "dividend",
                "value": 0.5,
            }
        ]
    )
    primary = StubProvider("prim", {"AAA": _bars("AAA")}, actions=actions)
    secondary = StubProvider("sec", {"BBB": _bars("BBB", adj_close=None)})
    provider = FailoverPriceProvider(primary, secondary)
    provider.fetch_daily_bars(("AAA", "BBB"), START, END)
    assert provider.last_corporate_actions.equals(actions)

    # A primary without the attribute yields an empty frame, never an error.
    bare = FailoverPriceProvider(StubProvider("prim", {"AAA": _bars("AAA")}), secondary)
    bare.fetch_daily_bars(("AAA",), START, END)
    assert bare.last_corporate_actions.empty


def test_registry_resolves_yahoo_stooq(cfg: EdgeStackConfig) -> None:
    provider = get_price_provider("yahoo_stooq", cfg)
    assert isinstance(provider, FailoverPriceProvider)
    assert provider.metadata.name == "yahoo+stooq"
    assert any("adj_close=close" in item for item in provider.metadata.limitations)
