"""Daily forward point-in-time universe snapshot (survivorship-free capture).

Backfilled membership history is only an approximation (public change log);
but every snapshot taken TODAY is exact by construction. This script freezes
the current S&P 500 membership (PitSP500Universe) plus the catalog's active
symbols into an append-only dated JSON. From the first snapshot onward, any
forward study can reconstruct the exact investable universe on any date with
zero survivorship bias — the bias only lives in the past.

Run nightly (chained in nightly.bat). Idempotent: one file per calendar day,
existing snapshots are never rewritten.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / "data" / "curated" / "universe_snapshots"
PRICES = ROOT / "data" / "curated" / "prices"
CACHE = ROOT / "data" / "cache" / "universe"


def active_catalog_symbols(max_stale_days: int = 7) -> list[str]:
    out = []
    cutoff = pd.Timestamp(date.today()) - pd.Timedelta(days=max_stale_days)
    for f in PRICES.glob("*.parquet"):
        try:
            last = pd.read_parquet(f, columns=["date"])["date"].max()
        except (OSError, ValueError, KeyError):
            continue
        if pd.notna(last) and pd.Timestamp(last) >= cutoff:
            out.append(f.stem)
    return sorted(out)


def main() -> int:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from edgestack.data.universe_pit import PitSP500Universe

    today = date.today()
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    out = SNAP_DIR / f"{today.isoformat()}.json"
    if out.exists():
        print(f"universe snapshot for {today} already exists — skipped")
        return 0
    try:
        members = list(PitSP500Universe(CACHE, earliest=date(2000, 1, 3)).members(today))
        membership_source = "sp500_change_log"
    except Exception as exc:
        members, membership_source = [], f"unavailable ({exc})"
    payload = {
        "date": today.isoformat(),
        "sp500_members": members,
        "membership_source": membership_source,
        "catalog_active": active_catalog_symbols(),
    }
    out.write_text(json.dumps(payload, indent=1))
    print(
        f"universe snapshot {today}: {len(members)} S&P members "
        f"({membership_source}), {len(payload['catalog_active'])} active catalog symbols -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
