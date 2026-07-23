"""Forward intraday-bar collector (own-your-archive strategy).

Free intraday backfill no longer exists at universe scale; the honest path
is to collect forward. This nightly step pulls 1m/5m/15m/60m OHLCV trades,
including extended hours, for a bounded liquid watch set and appends them to
the catalog. The 1m/5m archive supports opening-sequence research; 15m is
coarse and 60m is context only.

Vendor limits (Yahoo): 1m <= 7, 5m/15m <= 59, 60m <= 729 calendar days.
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
    "IWM",
    "DIA",
    "XLK",
    "SMH",
    "XLF",
    "XLE",
    "XLY",
    "XLP",  # opening-fade index, sector, and style proxies
    "USO",
    "BNO",  # oil surge watch proxies (oil_surge_watch.py)
    "TLT",
    "SHY",
    "GLD",  # benchmarks + baseline policy
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",  # optional liquid-stock cohort; reported separately
)
LOOKBACK_CALENDAR_DAYS = 7


def main() -> int:
    from edgestack.config import load_config
    from edgestack.pipelines import run_intraday_download

    cfg = load_config(str(ROOT / "configs" / "live.yaml"))
    end = date.today()
    start = end - timedelta(days=LOOKBACK_CALENDAR_DAYS - 1)
    failures = 0
    for interval in ("60m", "15m", "5m", "1m"):
        try:
            run_intraday_download(
                cfg,
                start,
                end,
                symbols=WATCH,
                interval=interval,
                include_prepost=True,
            )
            print(f"intraday collector: {interval} bars {start}..{end} for {len(WATCH)} symbols")
        except Exception as exc:
            print(f"WARN intraday collector {interval} failed: {exc}")
            failures += 1
    try:
        from edgestack.research.opening_state import sync_opening_fade_state

        campaign = sync_opening_fade_state(cfg)
        common = campaign.metrics.get("common_archive_sessions", 0)
        print(f"Edge Lab opening archive: {common}/252 common sessions")
    except Exception as exc:
        print(f"WARN opening-state sync failed: {exc}")
        failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
