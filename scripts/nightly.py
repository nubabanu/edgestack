"""Fail-fast nightly orchestrator for the canonical V2 publication."""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from edgestack.config import EdgeStackConfig, load_config
from edgestack.data.calendar import TradingCalendar
from edgestack.data.catalog import DataCatalog
from edgestack.data.providers.registry import get_price_provider
from edgestack.data.quality import assess_panel
from edgestack.exceptions import DataError
from edgestack.logging import configure
from edgestack.pipelines import run_features_build
from edgestack.recommendation.nightly import build_and_publish_canonical_baseline
from edgestack.recommendation.policy import load_baseline_policy


def update_data(cfg: EdgeStackConfig, run_date: date) -> None:
    catalog = DataCatalog(cfg)
    policy_symbols = {weight.symbol for weight in load_baseline_policy().weights}
    required = set(cfg.universe.symbols) | policy_symbols
    symbols = tuple(sorted(set(catalog.list_symbols()) | required))
    if not symbols:
        raise RuntimeError("nightly data update has no configured or catalog symbols")
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


def run_nightly(
    cfg: EdgeStackConfig,
    *,
    run_date: date,
    update_prices: bool = True,
    build_features: bool = True,
) -> str:
    if update_prices:
        update_data(cfg, run_date)
    validate_required(cfg)
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
