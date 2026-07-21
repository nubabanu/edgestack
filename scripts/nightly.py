"""Fail-fast nightly orchestrator for the canonical V2 publication."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from edgestack.config import EdgeStackConfig, load_config
from edgestack.data.calendar import TradingCalendar
from edgestack.data.catalog import DataCatalog
from edgestack.data.fetch_worker import fetch_and_store
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
from edgestack.research.locks import nightly_lock

try:
    from scripts import nightly_status
except ImportError:  # running as `python scripts/nightly.py`
    import nightly_status  # type: ignore[no-redef]

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


def _fetch_batch_isolated(
    config_path: str | Path,
    batch: tuple[str, ...],
    start: date,
    end: date,
    artifacts_dir: Path,
    *,
    timeout_s: float = 600,
) -> dict[str, Any] | None:
    """Run one fetch batch via ``edgestack.data.fetch_worker`` in a child process.

    A native crash (the intermittent PyEval_SaveThread abort) surfaces as a
    nonzero exit with no result file; each batch gets one fresh-process retry.
    Returns the worker's result dict, or None when both attempts failed.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fd, out_name = tempfile.mkstemp(dir=artifacts_dir, prefix=".fetch_batch_", suffix=".json")
    os.close(fd)
    out_path = Path(out_name)
    cmd = [
        sys.executable,
        "-m",
        "edgestack.data.fetch_worker",
        "--config",
        str(config_path),
        "--symbols",
        ",".join(batch),
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
        "--out",
        str(out_path),
    ]
    try:
        result: dict[str, Any] | None = None
        for attempt in (1, 2):
            reason = "worker crashed (no result file)"
            try:
                proc = subprocess.run(cmd, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                reason = f"worker timed out after {timeout_s:.0f}s"
            else:
                try:
                    result = json.loads(out_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    result = None
                if result is not None and proc.returncode == 0 and "error" not in result:
                    return result
                if result is not None:
                    reason = str(result.get("error", f"worker exit code {proc.returncode}"))
            print(f"nightly fetch batch attempt {attempt} failed ({reason}): {list(batch)}")
        return result
    finally:
        out_path.unlink(missing_ok=True)


def update_data(
    cfg: EdgeStackConfig,
    run_date: date,
    *,
    config_path: str | Path | None = None,
    isolate_fetch: bool = True,
) -> dict[str, Any]:
    """Refresh the price catalog; returns a summary for the run-status artifact."""
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
    start = run_date - timedelta(days=45)
    isolated = isolate_fetch and config_path is not None
    provider_counts: Counter[str] = Counter()
    failed_batches: list[list[str]] = []
    for offset in range(0, len(symbols), 25):
        batch = symbols[offset : offset + 25]
        if isolated:
            assert config_path is not None
            result = _fetch_batch_isolated(
                config_path, batch, start, run_date, Path(cfg.paths.artifacts_dir)
            )
            if result is None or "error" in result:
                # A crashed batch only blocks publication when it starves the
                # symbols tonight's policy/universe actually needs.
                failed_required = sorted(set(batch) & required)
                if failed_required:
                    reason = (result or {}).get("error", "worker crashed twice")
                    raise RuntimeError(
                        f"nightly fetch failed for required symbols {failed_required}: {reason}"
                    )
                failed_batches.append(list(batch))
                continue
            summary = result
        else:
            summary = fetch_and_store(cfg, batch, start, run_date)
        provider_counts.update(summary["provider_by_symbol"].values())
        missing = list(summary["missing"])
        # Fail-fast only on symbols the canonical policy/universe needs today.
        # Historical PIT catalog members can be delisted (empty series) without
        # invalidating the publication.
        missing_required = sorted(set(missing) & required)
        if missing_required:
            raise RuntimeError(f"nightly provider returned no rows for {missing_required}")
        if missing:
            print(f"nightly skipped {len(missing)} inactive catalog symbols: {missing}")
    return {
        "provider_by_symbol_counts": dict(provider_counts),
        "failed_batches": failed_batches,
    }


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


def _stage(name: str, fn: Callable[[], Any]) -> Any:
    """Run one internal stage, recording its outcome in the status ledger.

    Fail-fast semantics are unchanged: a failed stage is recorded, then the
    exception propagates and aborts the run.
    """
    started = time.monotonic()
    try:
        result = fn()
    except Exception:
        nightly_status.record_stage(name, "failed", duration_s=time.monotonic() - started)
        raise
    detail = result if isinstance(result, dict) else None
    nightly_status.record_stage(name, "ok", duration_s=time.monotonic() - started, detail=detail)
    return result


def run_nightly(
    cfg: EdgeStackConfig,
    *,
    run_date: date,
    update_prices: bool = True,
    build_features: bool = True,
    fetch_news: bool = True,
    config_path: str | Path | None = None,
    isolate_fetch: bool = True,
) -> str:
    with nightly_lock(Path(cfg.paths.artifacts_dir)):
        if update_prices:
            _stage(
                "data_update",
                lambda: update_data(
                    cfg, run_date, config_path=config_path, isolate_fetch=isolate_fetch
                ),
            )
        _stage("quality_gates", lambda: validate_required(cfg))
        if build_features:
            _stage("features", lambda: run_features_build(cfg))
        news: tuple[NewsEvidenceV2, ...] = ()
        if fetch_news:
            news = _stage(
                "news",
                lambda: gather_news_evidence(
                    _news_symbols(cfg),
                    Path(cfg.paths.data_dir) / "cache" / "news",
                    timeout_seconds=cfg.data.request_timeout_seconds,
                ),
            )
            print(f"nightly news context: {len(news)} items")
        publication = _stage(
            "publish",
            lambda: build_and_publish_canonical_baseline(
                cfg, run_date=run_date, news_evidence=news
            ),
        )
        return publication.run_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/live.yaml")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--skip-data-update", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument(
        "--no-fetch-isolation",
        action="store_true",
        help="fetch in-process instead of per-batch subprocesses (debugging)",
    )
    args = parser.parse_args()
    configure()
    cfg = load_config(args.config)
    run_id = run_nightly(
        cfg,
        run_date=date.fromisoformat(args.date),
        update_prices=not args.skip_data_update,
        build_features=not args.skip_features,
        fetch_news=not args.skip_news,
        config_path=args.config,
        isolate_fetch=not args.no_fetch_isolation,
    )
    print(f"published canonical run {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
