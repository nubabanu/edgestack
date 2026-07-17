"""Fail-fast nightly orchestrator for the canonical V2 publication."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from edgestack.config import EdgeStackConfig, load_config
from edgestack.data.calendar import TradingCalendar
from edgestack.data.catalog import DataCatalog
from edgestack.data.providers.registry import get_price_provider
from edgestack.data.quality import assess_panel
from edgestack.exceptions import DataError
from edgestack.logging import configure
from edgestack.pipelines import run_features_build
from edgestack.recommendation.instrument_schemas import NewsEvidenceV2
from edgestack.recommendation.news import gather_news_evidence
from edgestack.recommendation.nightly import (
    build_and_publish_canonical_baseline,
    load_legacy_watchlist,
)
from edgestack.recommendation.policy import load_baseline_policy

# Catalog symbols with no bar for this many days are treated as delisted and
# skipped by the nightly fetch. Their history remains in the catalog —
# deleting it would introduce survivorship bias into point-in-time research.
INACTIVE_AFTER_DAYS = 60


def _split_active(
    catalog: DataCatalog, symbols: tuple[str, ...], run_date: date, required: set[str]
) -> tuple[list[str], list[str]]:
    active: list[str] = []
    inactive: list[str] = []
    for symbol in symbols:
        if symbol in required:
            active.append(symbol)
            continue
        path = catalog.prices_dir / f"{symbol}.parquet"
        try:
            last = pd.read_parquet(path, columns=["date"])["date"].max()
        except (OSError, ValueError, KeyError):
            active.append(symbol)
            continue
        if pd.isna(last) or (run_date - pd.Timestamp(last).date()).days > INACTIVE_AFTER_DAYS:
            inactive.append(symbol)
        else:
            active.append(symbol)
    return active, inactive


def update_data(cfg: EdgeStackConfig, run_date: date) -> None:
    catalog = DataCatalog(cfg)
    policy_symbols = {weight.symbol for weight in load_baseline_policy().weights}
    required = set(cfg.universe.symbols) | policy_symbols
    all_symbols = tuple(sorted(set(catalog.list_symbols()) | required))
    if not all_symbols:
        raise RuntimeError("nightly data update has no configured or catalog symbols")
    active, inactive = _split_active(catalog, all_symbols, run_date, required)
    if inactive:
        print(
            f"nightly skipping {len(inactive)} inactive symbols "
            f"(no bar in {INACTIVE_AFTER_DAYS} days): {', '.join(inactive[:10])}"
            + ("…" if len(inactive) > 10 else "")
        )
    symbols = tuple(active)
    provider = get_price_provider(cfg.universe.source, cfg)
    start = run_date - timedelta(days=45)
    for offset in range(0, len(symbols), 25):
        batch = symbols[offset : offset + 25]
        bars = provider.fetch_daily_bars(batch, start, run_date)
        written = catalog.write_bars(bars, provider=cfg.universe.source)
        actions = getattr(provider, "last_corporate_actions", None)
        if actions is not None:
            catalog.write_corporate_actions(actions, provider=cfg.universe.source)
        missing = sorted(set(batch) - set(written))
        # Fail-fast only on symbols the canonical policy/universe needs today.
        # Historical PIT catalog members can be delisted (empty series) without
        # invalidating the publication.
        missing_required = sorted(set(missing) & required)
        if missing_required:
            raise RuntimeError(f"nightly provider returned no rows for {missing_required}")
        if missing:
            print(f"nightly skipped {len(missing)} inactive catalog symbols: {missing}")


def validate_required(cfg: EdgeStackConfig) -> None:
    """Quality-gate only the symbols tonight's publication consumes.

    The full research catalog contains delisted PIT members whose stale
    histories fail gates without affecting the baseline; those are still
    covered by the on-demand `edgestack data validate`. The publication
    itself re-checks its own panel fail-fast in
    build_and_publish_canonical_baseline.
    """
    catalog = DataCatalog(cfg)
    policy_symbols = {weight.symbol for weight in load_baseline_policy().weights}
    required = tuple(sorted(policy_symbols | set(cfg.universe.symbols)))
    with catalog.guard.unlock(reason="nightly required-symbol validation") as key:
        panel = catalog.load_panel(symbols=required, unlock_key=key)
    report = assess_panel(panel, TradingCalendar(cfg.data.calendar))
    print(report.summary())
    if report.quarantined:
        raise DataError("required symbols failed quality gates; publication blocked")


def _news_symbols(cfg: EdgeStackConfig) -> tuple[str, ...]:
    """Policy plus watchlist symbols — the set the published bundle can serve."""
    symbols = {weight.symbol for weight in load_baseline_policy().weights}
    symbols |= {entry.symbol for entry in load_legacy_watchlist(Path(cfg.paths.artifacts_dir))}
    return tuple(sorted(symbols))


def run_nightly(
    cfg: EdgeStackConfig,
    *,
    run_date: date,
    update_prices: bool = True,
    build_features: bool = True,
    fetch_news: bool = True,
) -> str:
    if update_prices:
        update_data(cfg, run_date)
    validate_required(cfg)
    if build_features:
        run_features_build(cfg)
    news: tuple[NewsEvidenceV2, ...] = ()
    if fetch_news:
        news = gather_news_evidence(
            _news_symbols(cfg),
            Path(cfg.paths.data_dir) / "cache" / "news",
            timeout_seconds=cfg.data.request_timeout_seconds,
        )
        print(f"nightly news context: {len(news)} items")
    publication = build_and_publish_canonical_baseline(cfg, run_date=run_date, news_evidence=news)
    return publication.run_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/live.yaml")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--skip-data-update", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-news", action="store_true")
    args = parser.parse_args()
    configure()
    cfg = load_config(args.config)
    run_id = run_nightly(
        cfg,
        run_date=date.fromisoformat(args.date),
        update_prices=not args.skip_data_update,
        build_features=not args.skip_features,
        fetch_news=not args.skip_news,
    )
    print(f"published canonical run {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
