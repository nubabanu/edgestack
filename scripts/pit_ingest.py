"""Phase 1: ingest the point-in-time S&P 500 universe and MEASURE coverage.

Downloads every ticker that was ever an index member during the sample from
the free Yahoo feed, then quantifies exactly how much of the point-in-time
universe is recoverable — survivorship becomes a measured number, not a
hidden assumption.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgestack.config import load_config
from edgestack.data.catalog import DataCatalog, atomic_write_bytes
from edgestack.data.providers.registry import get_price_provider
from edgestack.data.universe_pit import PitSP500Universe
from edgestack.exceptions import ProviderError
from edgestack.logging import configure

START, END = date(2011, 1, 1), date(2023, 12, 31)
SAMPLE_START = date(2012, 1, 1)


def main() -> int:
    configure()
    cfg = load_config("configs/broad.yaml")
    catalog = DataCatalog(cfg)
    universe = PitSP500Universe(Path("data/cache/universe"), earliest=date(2011, 1, 1))

    ever = list(universe.ever_members(SAMPLE_START, END))
    wanted = sorted(set(ever) | {"SPY"})
    current = set(universe.payload["current"])
    print(f"point-in-time ever-members {SAMPLE_START}..{END}: {len(ever)} "
          f"({sum(1 for s in ever if s not in current)} later removed)")

    provider = get_price_provider("yahoo", cfg)
    already = set(catalog.list_symbols())
    todo = [s for s in wanted if s not in already]
    print(f"downloading {len(todo)} symbols ({len(already)} already cached)")

    got, failed = list(already & set(wanted)), []
    t0 = time.time()
    chunk = 25
    for i in range(0, len(todo), chunk):
        batch = tuple(todo[i:i + chunk])
        try:
            bars = provider.fetch_daily_bars(batch, START, END)
            written = catalog.write_bars(bars, provider="yahoo")
            got.extend(written)
            failed.extend(s for s in batch if s not in written)
        except ProviderError:
            failed.extend(batch)
        done = i + len(batch)
        print(f"  {done}/{len(todo)} attempted, {len(got)} with data, "
              f"{len(failed)} unavailable ({time.time()-t0:.0f}s)")

    # -- coverage measurement -------------------------------------------------
    have = set(got)
    dead = [s for s in ever if s not in current]
    alive = [s for s in ever if s in current]
    coverage = {
        "universe_version": universe.payload["fetched_at"],
        "sample": [str(SAMPLE_START), str(END)],
        "ever_members": len(ever),
        "with_data": len(have & set(ever)),
        "alive_members": len(alive),
        "alive_with_data": len(have & set(alive)),
        "removed_members": len(dead),
        "removed_with_data": len(have & set(dead)),
        "unavailable_symbols": sorted(set(ever) - have),
    }
    coverage["overall_coverage"] = coverage["with_data"] / coverage["ever_members"]
    coverage["removed_coverage"] = (
        coverage["removed_with_data"] / coverage["removed_members"]
    )
    by_year = {}
    for year in range(SAMPLE_START.year, END.year + 1):
        members = universe.members(date(year, 6, 30))
        with_data = sum(1 for s in members if s in have)
        by_year[year] = {"members": len(members), "with_data": with_data,
                         "coverage": round(with_data / len(members), 3)}
    coverage["by_year"] = by_year

    atomic_write_bytes(Path(cfg.paths.artifacts_dir) / "universe_coverage.json",
                       json.dumps(coverage, indent=2).encode())
    catalog.audit("pit_universe_ingest", reason="phase1",
                  ever=len(ever), with_data=len(have))

    print("\n=== COVERAGE (the survivorship bias, measured) ===")
    print(f"ever-members with usable data: {coverage['with_data']}/{len(ever)} "
          f"({coverage['overall_coverage']:.0%})")
    print(f"  still-listed members: {coverage['alive_with_data']}/{len(alive)} "
          f"({coverage['alive_with_data']/len(alive):.0%})")
    print(f"  REMOVED members:      {coverage['removed_with_data']}/{len(dead)} "
          f"({coverage['removed_coverage']:.0%})  <- the critical number")
    for year, row in by_year.items():
        print(f"  {year}: {row['with_data']}/{row['members']} PIT members covered "
              f"({row['coverage']:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
