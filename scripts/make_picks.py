"""Derive the week / month / year picks from the existing research artifacts.

  week  : top frozen-board tilt (validated 10-session horizon)
  month : best board name with a CLEAN fundamentals profile (no high-short /
          extreme-leverage / negative-FCF flags, moderate beta) — the weekly
          edge plus balance-sheet screening for the longer, untested hold
  year  : top value-quality-momentum composite name (CURRENT-snapshot ranking,
          not backtestable — labeled as judgment, not validated edge)

Writes artifacts/picks.json (served at GET /picks, bundled into the app seed).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgestack.data.catalog import atomic_write_bytes

ART = Path("artifacts")


def clean_fundamentals(row: dict) -> tuple[bool, str]:
    flags = []
    if (row.get("shortPercentOfFloat") or 0) > 0.10:
        flags.append("high short interest")
    if (row.get("debtToEquity") or 0) > 300:
        flags.append("extreme leverage")
    fcf = row.get("freeCashflow")
    if fcf is not None and fcf < 0:
        flags.append("negative FCF")
    if (row.get("beta") or 1.0) > 1.8:
        flags.append("beta>1.8")
    return (not flags, ", ".join(flags) or "clean")


def main() -> int:
    board = json.loads((ART / "live_board.json").read_text(encoding="utf-8"))
    vqm = json.loads((ART / "vqm_rank.json").read_text(encoding="utf-8"))
    funda = json.loads((ART / "fundamentals_snapshot.json").read_text(encoding="utf-8"))["data"]

    rows = board["rows"]
    week = rows[0]
    week_pick = {
        "horizon": "week",
        "symbol": week["symbol"],
        "close": week["close"],
        "buy": "next session at the OPEN (validated execution); cancel if it "
        "gaps more than +1.5% above this close",
        "sell": "close of the 10th session, or stop/target first",
        "stop": week["stop"],
        "target": week["target"],
        "rationale": f"top frozen-system tilt: conviction "
        f"{week['conviction']:.0f}/100, E[net] "
        f"{week['e_net_10d']:+.2%}/10d, hit {week['hit']:.0%} "
        f"({week['n_edges']} edges, {week['families']} families)",
        "validated": True,
    }

    month_pick = None
    for r in rows:
        ok, verdict = clean_fundamentals(funda.get(r["symbol"], {}))
        if ok:
            month_pick = {
                "horizon": "month",
                "symbol": r["symbol"],
                "close": r["close"],
                "buy": "next session at the OPEN; a turn-of-month entry "
                "(last session of the month, at the close) adds the "
                "validated ToM tailwind",
                "sell": "~21 sessions, or stop first; re-check the board weekly",
                "stop": r["stop"],
                "target": r["target"],
                "rationale": f"board tilt (conviction {r['conviction']:.0f}) "
                f"with clean fundamentals ({verdict}); 1-month "
                f"hold extends the validated 10-session horizon "
                f"— extension itself is unvalidated",
                "validated": False,
            }
            break

    top = vqm["top30"][0]
    year_pick = {
        "horizon": "year",
        "symbol": top["symbol"],
        "name": top.get("shortName", ""),
        "buy": "timing is statistically irrelevant at 12 months; if choosing, "
        "buy at the close of the last session of a month (turn-of-"
        "month) — never on leverage",
        "sell": "review quarterly; exit if composite rank falls out of the "
        "top decile or SPY closes below its 200-DMA for two weeks",
        "rationale": f"#1 of 503 on the value-quality-momentum composite "
        f"(V {top['VALUE']:.2f} / Q {top['QUALITY']:.2f} / "
        f"G {top['GROWTH']:.2f} / M {top['MOMENTUM']:.2f} / "
        f"S {top['SENTIMENT']:.2f}). CURRENT-snapshot ranking — "
        f"informed judgment, NOT a backtested edge",
        "validated": False,
    }

    payload = {
        "schema_version": 1,
        "as_of": board["as_of"],
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "disclaimer": "Research output only. Not investment advice.",
        "picks": [week_pick, month_pick, year_pick],
    }
    atomic_write_bytes(ART / "picks.json", json.dumps(payload, indent=1).encode())
    for p in payload["picks"]:
        if p:
            print(f"{p['horizon']:>5}: {p['symbol']:<6} — {p['rationale'][:80]}")
    print("saved -> artifacts/picks.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
