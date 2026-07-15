# Adding a data provider

Price providers implement `PriceDataProvider`
(`src/edgestack/data/providers/base.py`) and register a factory:

```python
from edgestack.data.providers.base import PriceDataProvider, ProviderMetadata
from edgestack.data.providers.registry import register_price_provider


class MyProvider(PriceDataProvider):
    def __init__(self, ...):
        self.metadata = ProviderMetadata(
            name="myfeed", kind="price",
            supported_fields=("open", "high", "low", "close", "volume"),
            is_point_in_time=False,
            adjustment="split",              # raw | split | split_dividend
            publication_delay="end_of_day",
            limitations=(
                "state EVERY known gap honestly: missing delisted symbols, "
                "no dividend adjustment, throttling, ...",
            ),
        )

    def fetch_daily_bars(self, symbols, start, end) -> pd.DataFrame:
        ...  # return canonical columns; validate with schemas.validate_bars


@register_price_provider("myfeed")
def _make(cfg):  # add "myfeed" to the module list in registry.get_price_provider
    return MyProvider(...)
```

Also add the name to `UniverseConfig.source`'s Literal in `config.py`.

## Requirements

1. **Honest metadata.** The `limitations` tuple is printed at download time
   and drives report warnings. A feed without delisted symbols must say
   "survivorship-biased"; a feed without dividend adjustment must say so.
2. **Return canonical bars** and pass them through
   `edgestack.data.schemas.validate_bars` — structural violations should fail
   in the provider, not corrupt the catalog.
3. **Network discipline**: timeouts from `cfg.data.request_timeout_seconds`,
   bounded retries with exponential backoff, HTTP 429 treated as retryable,
   an on-disk cache under `data/cache/<name>/` with
   `cfg.data.cache_ttl_days`, and a polite inter-request delay.
4. **Refuse garbage**: if the endpoint returns HTML, an error document or an
   empty body, raise `ProviderError` — never parse a guess (see the Stooq
   provider detecting the anti-bot page).
5. **No credentials in code or logs.** Read keys from environment variables;
   the structured logger redacts common secret patterns, but do not rely on
   it.
6. **Tests stay offline**: pin a real response excerpt as a fixture and unit-
   test the parser; put any live call behind `@pytest.mark.network`
   (excluded by default).

Non-price interfaces (earnings, macro, short/borrow, options, sentiment,
news, fundamentals, corporate actions) follow the same pattern from the ABCs
in `base.py`; point-in-time discipline (publication timestamps, vintage data)
is mandatory for anything feeding features.
