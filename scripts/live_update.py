"""Extend the broad catalog with recent data (2024 -> today) for live scoring."""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgestack.config import load_config
from edgestack.data.catalog import DataCatalog
from edgestack.data.providers.registry import get_price_provider
from edgestack.exceptions import ProviderError
from edgestack.logging import configure


def main() -> int:
    configure()
    cfg = load_config("configs/broad.yaml")
    catalog = DataCatalog(cfg)
    symbols = catalog.list_symbols()
    provider = get_price_provider("yahoo", cfg)
    start, end = date(2024, 1, 1), date.today()
    print(f"updating {len(symbols)} symbols {start}..{end}")
    t0 = time.time()
    got, failed = 0, 0
    chunk = 25
    for i in range(0, len(symbols), chunk):
        batch = tuple(symbols[i:i + chunk])
        try:
            bars = provider.fetch_daily_bars(batch, start, end)
            written = catalog.write_bars(bars, provider="yahoo")
            got += len(written)
            failed += len(batch) - len(written)
        except ProviderError:
            failed += len(batch)
        if (i // chunk) % 5 == 4:
            print(f"  {i + len(batch)}/{len(symbols)} ({time.time()-t0:.0f}s)")
    print(f"updated {got}, unavailable {failed} ({time.time()-t0:.0f}s)")
    catalog.audit("live_update", reason=str(end), updated=got)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
