"""Fail-fast nightly orchestrator for the canonical V2 publication."""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from edgestack.config import EdgeStackConfig, load_config
from edgestack.data.catalog import DataCatalog
from edgestack.data.providers.registry import get_price_provider
from edgestack.logging import configure
from edgestack.pipelines import run_data_validate, run_features_build
from edgestack.recommendation.nightly import build_and_publish_canonical_baseline
from edgestack.recommendation.policy import load_baseline_policy


def update_data(cfg: EdgeStackConfig, run_date: date) -> None:
    catalog = DataCatalog(cfg)
    policy_symbols = {weight.symbol for weight in load_baseline_policy().weights}
    symbols = tuple(
        sorted(set(catalog.list_symbols()) | set(cfg.universe.symbols) | policy_symbols)
    )
    if not symbols:
        raise RuntimeError("nightly data update has no configured or catalog symbols")
    provider = get_price_provider(cfg.universe.source, cfg)
    start = run_date - timedelta(days=45)
    for offset in range(0, len(symbols), 25):
        batch = symbols[offset : offset + 25]
        bars = provider.fetch_daily_bars(batch, start, run_date)
        written = catalog.write_bars(bars, provider=cfg.universe.source)
        missing = sorted(set(batch) - set(written))
        if missing:
            raise RuntimeError(f"nightly provider returned no rows for {missing}")


def run_nightly(
    cfg: EdgeStackConfig,
    *,
    run_date: date,
    update_prices: bool = True,
    build_features: bool = True,
) -> str:
    if update_prices:
        update_data(cfg, run_date)
    run_data_validate(cfg)
    if build_features:
        run_features_build(cfg)
    publication = build_and_publish_canonical_baseline(cfg, run_date=run_date)
    return publication.run_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/live.yaml")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--skip-data-update", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    args = parser.parse_args()
    configure()
    cfg = load_config(args.config)
    run_id = run_nightly(
        cfg,
        run_date=date.fromisoformat(args.date),
        update_prices=not args.skip_data_update,
        build_features=not args.skip_features,
    )
    print(f"published canonical run {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
