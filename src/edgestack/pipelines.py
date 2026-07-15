"""Pipeline entry points wired to the CLI.

Each function is replaced with a real implementation as its milestone lands;
unimplemented stages raise ``NotImplementedError`` so the CLI exits cleanly
with code 2 instead of pretending to work.
"""

from __future__ import annotations

from datetime import date

from edgestack.config import EdgeStackConfig


def run_data_download(cfg: EdgeStackConfig, start: date, end: date, *,
                      provider: str | None = None,
                      symbols: tuple[str, ...] | None = None) -> None:
    from edgestack.data.catalog import DataCatalog
    from edgestack.data.providers.registry import get_price_provider
    from edgestack.data.universe import static_universe

    provider_name = provider or cfg.universe.source
    price_provider = get_price_provider(provider_name, cfg)
    universe = static_universe(cfg, end)
    wanted = symbols or universe.symbols
    print(f"downloading {len(wanted)} symbols {start}..{end} from {provider_name}")
    for limitation in price_provider.metadata.limitations:
        print(f"  provider limitation: {limitation}")
    for limitation in universe.limitations:
        print(f"  universe limitation: {limitation}")

    bars = price_provider.fetch_daily_bars(tuple(wanted), start, end)
    catalog = DataCatalog(cfg)
    written = catalog.write_bars(bars, provider=provider_name)
    catalog.audit("data_download", reason=provider_name,
                  symbols=len(written), rows=len(bars))
    got = set(written)
    for symbol in wanted:
        status = f"{written[symbol]} rows" if symbol in got else "NO DATA"
        print(f"  {symbol}: {status}")
    print(f"catalog now holds {len(catalog.list_symbols())} symbols")


def run_data_validate(cfg: EdgeStackConfig) -> None:
    from edgestack.data.calendar import TradingCalendar
    from edgestack.data.catalog import DataCatalog
    from edgestack.data.quality import assess_panel

    catalog = DataCatalog(cfg)
    # Quality assessment may inspect the full history including the guarded
    # test period: it validates data plumbing, not trading hypotheses.
    with catalog.guard.unlock(reason="data quality validation") as key:
        panel = catalog.load_panel(unlock_key=key)
    report = assess_panel(panel, TradingCalendar(cfg.data.calendar))
    print(report.summary())
    if report.quarantined:
        print("some symbols failed quality gates; fix or exclude them before research")


def run_features_build(cfg: EdgeStackConfig) -> None:
    raise NotImplementedError("feature build (milestone 4)")


def run_edges_discover(cfg: EdgeStackConfig) -> None:
    raise NotImplementedError("edge discovery (milestone 7)")


def run_edges_validate(cfg: EdgeStackConfig, *, batch_id: str | None = None) -> None:
    raise NotImplementedError("edge validation (milestone 7)")


def run_models_train(cfg: EdgeStackConfig) -> None:
    raise NotImplementedError("model training (milestone 8)")


def run_signals_generate(cfg: EdgeStackConfig, *, as_of: date | None = None) -> None:
    raise NotImplementedError("signal generation (milestone 9)")


def run_signals_rank(cfg: EdgeStackConfig, *, top: int = 20, as_of: date | None = None) -> None:
    raise NotImplementedError("signal ranking (milestone 9)")


def run_backtest(cfg: EdgeStackConfig, *, scenario: str | None = None) -> None:
    raise NotImplementedError("backtest (milestone 10)")


def run_monitor(cfg: EdgeStackConfig) -> None:
    raise NotImplementedError("monitoring (milestone 11)")


def run_report_backtest(cfg: EdgeStackConfig, *, run_id: str | None = None) -> None:
    raise NotImplementedError("backtest report (milestone 10)")


def run_paper_session(cfg: EdgeStackConfig, *, as_of: date | None = None) -> None:
    raise NotImplementedError("paper trading (milestone 12)")


def run_api_serve(cfg: EdgeStackConfig, *, host: str, port: int) -> None:
    raise NotImplementedError("API server (milestone 12)")
