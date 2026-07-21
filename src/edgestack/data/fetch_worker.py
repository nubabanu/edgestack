"""Bounded price-refresh worker, runnable in a short-lived child process.

The nightly Yahoo fetch intermittently dies with a fatal native
``PyEval_SaveThread`` abort somewhere in the fetch/parquet-write loop. Running
each batch through this module in a subprocess turns that abort into a nonzero
exit (with no result file) that the orchestrator can detect and retry, instead
of losing the whole publication.

``fetch_and_store`` is the single implementation used both in-process and via
the CLI:

    python -m edgestack.data.fetch_worker --config configs/live.yaml \
        --symbols SPY,TLT --start 2026-06-05 --end 2026-07-20 --out result.json

The result JSON is ``{"written": {sym: rows}, "missing": [...],
"provider_by_symbol": {sym: name}}``; handled errors write ``{"error": ...}``
and exit 1. The worker never takes ``nightly_lock`` (the orchestrator holds
it) and never touches the TestPeriodGuard.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from edgestack.config import EdgeStackConfig, load_config
from edgestack.data.catalog import DataCatalog, atomic_write_bytes
from edgestack.data.providers.registry import get_price_provider
from edgestack.logging import configure


def fetch_and_store(
    cfg: EdgeStackConfig, symbols: tuple[str, ...], start: date, end: date
) -> dict[str, Any]:
    """Fetch daily bars for ``symbols`` and merge them into the catalog."""
    catalog = DataCatalog(cfg)
    provider = get_price_provider(cfg.universe.source, cfg)
    bars = provider.fetch_daily_bars(symbols, start, end)
    written = catalog.write_bars(bars, provider=cfg.universe.source)
    actions = getattr(provider, "last_corporate_actions", None)
    if actions is not None:
        catalog.write_corporate_actions(actions, provider=cfg.universe.source)
    per_symbol = getattr(provider, "last_provider_by_symbol", None) or {}
    return {
        "written": {str(symbol): int(rows) for symbol, rows in written.items()},
        "missing": sorted(set(symbols) - set(written)),
        "provider_by_symbol": {
            str(symbol): per_symbol.get(str(symbol), cfg.universe.source) for symbol in written
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--symbols", required=True, help="comma-separated ticker list")
    parser.add_argument("--start", required=True, help="ISO date")
    parser.add_argument("--end", required=True, help="ISO date")
    parser.add_argument("--out", required=True, help="path for the JSON result file")
    args = parser.parse_args(argv)
    configure()
    out_path = Path(args.out)
    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    try:
        cfg = load_config(args.config)
        result = fetch_and_store(
            cfg, symbols, date.fromisoformat(args.start), date.fromisoformat(args.end)
        )
    except Exception as exc:  # handled failure: structured error + exit 1
        atomic_write_bytes(out_path, json.dumps({"error": str(exc)}).encode("utf-8"))
        return 1
    atomic_write_bytes(out_path, json.dumps(result).encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
