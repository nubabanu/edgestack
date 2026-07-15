"""Provider registry: resolve provider names from config to instances."""

from __future__ import annotations

from collections.abc import Callable

from edgestack.config import EdgeStackConfig
from edgestack.data.providers.base import PriceDataProvider
from edgestack.exceptions import ProviderError

_PRICE_FACTORIES: dict[str, Callable[[EdgeStackConfig], PriceDataProvider]] = {}


def register_price_provider(name: str) -> Callable:
    def deco(factory: Callable[[EdgeStackConfig], PriceDataProvider]) -> Callable:
        if name in _PRICE_FACTORIES:
            raise ProviderError(f"price provider already registered: {name}")
        _PRICE_FACTORIES[name] = factory
        return factory

    return deco


def get_price_provider(name: str, cfg: EdgeStackConfig) -> PriceDataProvider:
    # Import implementations lazily so registration happens on first use
    # without import cycles.
    import importlib

    for module in ("synthetic", "local_files", "stooq"):
        try:
            importlib.import_module(f"edgestack.data.providers.{module}")
        except ImportError:  # provider module not built yet / optional deps missing
            continue

    try:
        factory = _PRICE_FACTORIES[name]
    except KeyError:
        raise ProviderError(
            f"unknown price provider {name!r}; available: {sorted(_PRICE_FACTORIES)}"
        ) from None
    return factory(cfg)
