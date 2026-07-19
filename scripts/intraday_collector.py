"""Forward intraday-bar collector (own-your-archive strategy).

Free intraday backfill no longer exists at universe scale; the honest path
is to collect forward. This nightly step pulls the trailing few days of
60m and 15m bars for a small configured watch set, appending to the
catalog's intraday store. In 12-24 months this yields a provenance-clean
archive for execution-window research ("is 15:45 better than 10:00?") that
no vendor can take away or retro-revise.

Vendor limits (Yahoo): 60m history <= 729 days/request, 15m <= 59 days.
Pulling a trailing 7-day window nightly stays far inside both and makes
each run idempotent (overlapping bars are deduplicated by the catalog).

Symbols: the tranche watch set + basket, the baseline policy ETFs, and
SPY/QQQ as benchmarks. Edit WATCH to extend; cost is one request per
(symbol, interval) per night.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WATCH = (
    "ACN",
    "CTSH",
    "EPAM",
    "DXC",
    "IBM",
    "IT",  # tranche program + basket
    "SPY",
    "QQQ",
    "TLT",
    "SHY",
    "GLD",  # benchmarks + baseline policy
)
LOOKBACK_DAYS = 7


def main() -> int:
    from edgestack.config import load_config
    from edgestack.pipelines import run_intraday_download

    cfg = load_config(str(ROOT / "configs" / "live.yaml"))
    end = date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    failures = 0
    for interval in ("60m", "15m"):
        try:
            run_intraday_download(cfg, start, end, symbols=WATCH, interval=interval)
            print(f"intraday collector: {interval} bars {start}..{end} for {len(WATCH)} symbols")
        except Exception as exc:
            print(f"WARN intraday collector {interval} failed: {exc}")
            failures += 1
    return 1 if failures == 2 else 0


if __name__ == "__main__":
    raise SystemExit(main())
