"""Generate the next 12 months of sniper dates from the exchange calendar.

Encodes the loss-aversion-first sniper system (docs/sniper-playbook.md):
highest-hit-rate windows only, defined 1-4 session holds, veto rules attached.
Writes artifacts/sniper_calendar.json and prints the schedule.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import exchange_calendars as xcals
import pandas as pd

from edgestack.data.catalog import atomic_write_bytes

VETOES = (
    "skip if SPY < 200-DMA",
    "skip if 20d vol > 20%",
    "never leverage",
    "size = pain limit / worst-case %",
)


def main() -> int:
    today = date.today()
    end = today + timedelta(days=400)
    cal = xcals.get_calendar("XNYS", start=str(today - timedelta(days=40)), end=str(end))
    sessions = pd.DatetimeIndex(cal.sessions_in_range(str(today), str(end)))

    def month_sessions(y: int, m: int) -> pd.DatetimeIndex:
        return sessions[(sessions.year == y) & (sessions.month == m)]

    events = []
    months = sorted({(d.year, d.month) for d in sessions})
    for y, m in months:
        ms = month_sessions(y, m)
        if len(ms) < 5:
            continue
        # turn-of-month trade: buy last session close, sell 4th session of next
        nxt = [(yy, mm) for yy, mm in months if (yy, mm) > (y, m)][:1]
        if nxt:
            ns = month_sessions(*nxt[0])
            if len(ns) >= 4 and m not in (8,):  # September entry vetoed anyway
                events.append(
                    {
                        "date": str(ms[-1].date()),
                        "trade": "ToM 4-session",
                        "action": f"buy QQQ/SPY at the close; sell {ns[3].date()} close",
                        "hit": "62-64%",
                        "worst": "-7%",
                        "note": "strongest in Oct/Nov/Dec; veto rules apply",
                    }
                )
        if m == 1:  # Feb sniper
            events.append(
                {
                    "date": str(ms[-1].date()),
                    "trade": "FEB-1 SNIPER (best of year)",
                    "action": "buy XLV (optional: V) at the close; sell next close",
                    "hit": "89% (XLV, 28y)",
                    "worst": "-2.0%",
                    "note": "tightest tail in the dataset (p5 -0.3%)",
                }
            )
        if m == 8:  # September de-risk
            events.append(
                {
                    "date": str(ms[-1].date()),
                    "trade": "SEPTEMBER STAND-ASIDE",
                    "action": "reduce/skip new entries until Sep 30",
                    "hit": "-",
                    "worst": "-",
                    "note": "worst month in 12/13 US instruments and 12/12 foreign markets",
                }
            )
        if m == 11:
            # Tuesday before Thanksgiving (4th Thursday)
            d = date(y, 11, 1)
            th = 0
            while True:
                if d.weekday() == 3:
                    th += 1
                    if th == 4:
                        break
                d += timedelta(days=1)
            wed = sessions[sessions < pd.Timestamp(d)][-1]
            tue = sessions[sessions < wed][-1]
            events.append(
                {
                    "date": str(tue.date()),
                    "trade": "THANKSGIVING WED",
                    "action": "buy SPY at the close; sell Wednesday close",
                    "hit": "70% (SPY) / 81% (QQQ, fatter tail)",
                    "worst": "-2.4% (SPY)",
                    "note": "do not initiate anything new before the 4-day closure",
                }
            )
            if len(ms) >= 7:
                events.append(
                    {
                        "date": str(ms[5].date()),
                        "trade": "NOV WORST-DAY FLAT",
                        "action": f"go flat at the close; {ms[6].date()} is the worst "
                        "single day of the year",
                        "hit": "-",
                        "worst": "-",
                        "note": "SPY t=-2.7 / QQQ t=-3.6 on Nov td7, both agree",
                    }
                )
        if m == 12 and len(ms) >= 15:
            events.append(
                {
                    "date": str(ms[13].date()),
                    "trade": "PRE-CHRISTMAS",
                    "action": "buy SPY at the close; sell next close (td15)",
                    "hit": "73%",
                    "worst": "small",
                    "note": "optional; Santa window is milder (+12bps/day)",
                }
            )

    events.sort(key=lambda e: e["date"])
    payload = {
        "generated_for": str(today),
        "vetoes": list(VETOES),
        "event_trigger": {
            "name": "RED CLOSE IN CALM UPTREND (any day)",
            "rule": "SPY down on the day, above 200-DMA, vol<20% -> buy the close, sell next open",
            "hit": "60%",
            "worst_p5": "-0.8%",
            "stat": "t=5.41, the gentlest trade found",
        },
        "events": events,
    }
    atomic_write_bytes(
        Path("artifacts") / "sniper_calendar.json", json.dumps(payload, indent=1).encode()
    )
    print(f"{'date':<12}{'trade':<28}{'hit':<24}{'worst':<14}action")
    for e in events:
        print(f"{e['date']:<12}{e['trade']:<28}{e['hit']:<24}{e['worst']:<14}{e['action']}")
    print(f"\nvetoes: {'; '.join(VETOES)}")
    print("saved -> artifacts/sniper_calendar.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
