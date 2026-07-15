"""Fetch a CURRENT-DAY fundamentals/positioning snapshot for S&P members.

Yahoo quoteSummary (crumb flow). This is a point-in-time snapshot of TODAY
only — usable for ranking today's opportunities, NOT for backtesting (no
historical vintages). Every number is as-reported by the free feed; treat as
approximate.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests

from edgestack.data.catalog import atomic_write_bytes
from edgestack.data.universe_pit import PitSP500Universe
from edgestack.logging import configure

MODULES = "financialData,defaultKeyStatistics,summaryDetail,price"
FIELDS = {
    "financialData": ["returnOnEquity", "profitMargins", "operatingMargins",
                      "grossMargins", "revenueGrowth", "earningsGrowth",
                      "freeCashflow", "totalCash", "totalDebt", "debtToEquity",
                      "currentRatio", "targetMeanPrice", "recommendationMean",
                      "numberOfAnalystOpinions", "returnOnAssets"],
    "defaultKeyStatistics": ["forwardPE", "trailingEps", "forwardEps", "pegRatio",
                             "priceToBook", "enterpriseToEbitda", "beta",
                             "heldPercentInsiders", "heldPercentInstitutions",
                             "shortPercentOfFloat", "shortRatio",
                             "52WeekChange"],
    "summaryDetail": ["trailingPE", "dividendYield", "marketCap",
                      "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "averageVolume"],
    "price": ["regularMarketPrice", "shortName"],
}


def main() -> int:
    configure()
    universe = PitSP500Universe(Path("data/cache/universe"), earliest=date(2011, 1, 1))
    symbols = list(universe.members(date.today()))
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    try:
        session.get("https://fc.yahoo.com", timeout=15)
    except requests.RequestException:
        pass
    crumb = session.get("https://query1.finance.yahoo.com/v1/test/getcrumb",
                        timeout=15).text.strip()
    print(f"fetching fundamentals for {len(symbols)} current members")

    out: dict[str, dict] = {}
    failed = 0
    t0 = time.time()
    for i, symbol in enumerate(symbols):
        try:
            r = session.get(
                f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}",
                params={"modules": MODULES, "crumb": crumb}, timeout=15)
            result = r.json()["quoteSummary"]["result"]
            if not result:
                failed += 1
                continue
            node = result[0]
            row: dict = {}
            for module, keys in FIELDS.items():
                sub = node.get(module) or {}
                for key in keys:
                    value = sub.get(key)
                    if isinstance(value, dict):
                        value = value.get("raw")
                    if value is not None:
                        row[key] = value
            out[symbol] = row
        except Exception:
            failed += 1
        if i % 60 == 59:
            print(f"  {i+1}/{len(symbols)} ({failed} failed, {time.time()-t0:.0f}s)")
        time.sleep(0.25)

    payload = {"as_of": str(date.today()), "n": len(out), "failed": failed,
               "data": out}
    atomic_write_bytes(Path("artifacts") / "fundamentals_snapshot.json",
                       json.dumps(payload).encode())
    print(f"saved {len(out)} snapshots ({failed} failed, {time.time()-t0:.0f}s) "
          f"-> artifacts/fundamentals_snapshot.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
