"""Swing-zone PAPER book: pre-registered forward experiment of the dip-zone strategy.

Runs the walk-forward rule that passed swing_zone_strategy_study.py
(2026-07-23, batch in artifacts/swing_zone_strategy_study.json) as a nightly
PAPER autopilot - the live-forward simulation. No live orders ever (repo
covenant).

FROZEN RULE (theta=5%, the scale that passed dev+val+holdout at both cost
levels): point-in-time zigzag pivots; dip zone = median of last 3 confirmed
troughs in trailing 756 sessions; oscillator filter >= 4 confirmed pivots and
top/dip zone gap >= 10%. Enter when close <= dip_zone * 1.05; exit at close
>= top_zone * 0.95, close < dip_zone * 0.85, or 21 sessions. Fills at the
next session's adjusted open. EUR 500 per position, max 10 concurrent, max 3
new entries per night (deepest-in-zone first).

PRE-REGISTERED REVIEW (do not judge earlier; do not fund earlier):
- Review after >= 100 closed trades AND >= 6 months of nightly operation.
- Success criteria fixed now: mean net trade return > 0 at 20 bps/side AND
  book total return >= SPY over the same span. Anything else = retire.
- The backtest's t-stats are inflated by cross-sectional correlation
  (simultaneous dips share the same market bounce); this forward book is the
  correction for that, not a formality.

Honest caveats: question-level selection bias remains from the scan session;
edge size is small (~+0.2%/trade over random timing at convention costs);
at EUR 500/position that is ~EUR 1-3 per trade - the experiment tests the
EDGE, not a get-rich mechanism. Not investment advice.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from swing_cycle_scan import load_prices  # noqa: E402
from swing_zone_strategy_study import (  # noqa: E402
    ENTRY_TOL,
    MIN_PIVOTS_TRAILING,
    MIN_ZONE_GAP,
    STOP_TOL,
    TARGET_TOL,
    TRAILING,
    zigzag_confirmed,
)

PRICES = ROOT / "data" / "curated" / "prices"
BOOK_PATH = ROOT / "artifacts" / "paper_swing.json"
LOG_PATH = ROOT / "logs" / "swing_paper_book.log"
TOAST_SCRIPT = ROOT / "scripts" / "notify_toast.ps1"

THETA = 0.05
TIMEOUT_SESSIONS = 21
EUR_PER_POSITION = 500
MAX_OPEN = 10
MAX_NEW_PER_NIGHT = 3
STARTED = "2026-07-23"


def current_zones(px: pd.Series) -> dict | None:
    """Point-in-time zones at the LAST bar (same math as the study, causal)."""
    log_px = np.log(px.to_numpy(dtype=float))
    n = len(log_px)
    if n <= TRAILING:
        return None
    t = n - 1
    pivots = [pv for pv in zigzag_confirmed(log_px, THETA) if pv[1] <= t]
    recent = [pv for pv in pivots if pv[1] > t - TRAILING]
    troughs = [pv[0] for pv in recent if pv[2] == "T"][-3:]
    peaks = [pv[0] for pv in recent if pv[2] == "P"][-3:]
    if len(recent) < MIN_PIVOTS_TRAILING or not troughs or not peaks:
        return None
    dip_zone = float(np.exp(np.median(log_px[troughs])))
    top_zone = float(np.exp(np.median(log_px[peaks])))
    if top_zone / dip_zone < MIN_ZONE_GAP:
        return None
    return {"dip_zone": dip_zone, "top_zone": top_zone, "close": float(px.iloc[-1])}


def load_book() -> dict:
    if BOOK_PATH.exists():
        return json.loads(BOOK_PATH.read_text(encoding="utf-8"))
    return {
        "_note": (
            "Pre-registered PAPER experiment (see swing_paper_book.py docstring). "
            "Research only, no live orders ever."
        ),
        "started": STARTED,
        "params": {
            "theta": THETA,
            "timeout": TIMEOUT_SESSIONS,
            "eur": EUR_PER_POSITION,
            "max_open": MAX_OPEN,
        },
        "open": [],
        "closed": [],
    }


def adj_open_fill(df: pd.DataFrame, after_date: str) -> tuple[str, float] | None:
    """First session strictly after `after_date`: (date, adjusted open)."""
    idx = df.index[df.index > pd.Timestamp(after_date)]
    if not len(idx):
        return None
    row = df.loc[idx[0]]
    return str(idx[0].date()), float(row["open"] * row["adj_close"] / row["close"])


def notify(lines: list[str]) -> None:
    body = "; ".join(lines)[:250]
    try:
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(TOAST_SCRIPT),
                "-Title",
                "EdgeStack swing paper book",
                "-Body",
                body,
            ],
            timeout=30,
            capture_output=True,
            check=False,
        )
    except Exception as exc:
        print(f"WARN toast failed: {exc}")
    try:
        import notify_telegram

        notify_telegram.send("EdgeStack swing paper book\n" + "\n".join(lines)[:3800])
    except Exception as exc:
        print(f"WARN telegram failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="evaluate but never persist/send")
    args = parser.parse_args()

    book = load_book()
    lines: list[str] = []
    alerts: list[str] = []

    # pass 1: fill pending intents at the next session's adjusted open
    for pos in list(book["open"]):
        if pos.get("entry_fill") is None:
            df = pd.read_parquet(PRICES / f"{pos['symbol']}.parquet")
            df = df.assign(date=pd.to_datetime(df["date"])).set_index("date").sort_index()
            fill = adj_open_fill(df, pos["signal_date"])
            if fill:
                pos["entry_fill_date"], pos["entry_fill"] = fill
                pos["shares"] = round(EUR_PER_POSITION / fill[1], 4)
                lines.append(f"PAPER filled entry {pos['symbol']} @ {fill[1]:.2f} ({fill[0]})")
        elif pos.get("exit_signal_date") and pos.get("exit_fill") is None:
            df = pd.read_parquet(PRICES / f"{pos['symbol']}.parquet")
            df = df.assign(date=pd.to_datetime(df["date"])).set_index("date").sort_index()
            fill = adj_open_fill(df, pos["exit_signal_date"])
            if fill:
                pos["exit_fill_date"], pos["exit_fill"] = fill
                ret = fill[1] / pos["entry_fill"] - 1
                pos["net_return"] = round(ret, 4)
                book["closed"].append(pos)
                book["open"].remove(pos)
                lines.append(
                    f"PAPER closed {pos['symbol']} {pos['exit_reason']} "
                    f"@ {fill[1]:.2f} ({ret:+.1%})"
                )

    # pass 2: evaluate exits and new entries on the latest closes
    open_syms = {p["symbol"] for p in book["open"]}
    candidates: list[tuple[float, str, dict]] = []
    for path in sorted(PRICES.glob("*.parquet")):
        sym = path.stem
        if "=" in sym or sym.startswith("^"):
            continue  # paper book trades stocks only (the PASS group without roll caveats)
        try:
            px = load_prices(sym)
        except Exception:
            continue
        if px is None:
            continue
        zones = current_zones(px)
        if zones is None:
            continue
        last_date = str(px.index[-1].date())
        if sym in open_syms:
            pos = next(p for p in book["open"] if p["symbol"] == sym)
            if pos.get("entry_fill") is None or pos.get("exit_signal_date"):
                continue
            held = (pd.Timestamp(last_date) - pd.Timestamp(pos["entry_fill_date"])).days
            reason = None
            if zones["close"] >= zones["top_zone"] * TARGET_TOL:
                reason = "target"
            elif zones["close"] < zones["dip_zone"] * STOP_TOL:
                reason = "zone_break"
            elif held >= TIMEOUT_SESSIONS * 1.6:  # calendar-day proxy for sessions
                reason = "timeout"
            if reason:
                pos["exit_signal_date"] = last_date
                pos["exit_reason"] = reason
                alerts.append(
                    f"PAPER exit intent {sym} ({reason}) at close {zones['close']:.2f}; "
                    "fills next open"
                )
        elif zones["dip_zone"] * STOP_TOL <= zones["close"] <= zones["dip_zone"] * ENTRY_TOL:
            # inside the zone window; below STOP_TOL the regime already broke
            depth = zones["close"] / zones["dip_zone"]
            candidates.append((depth, sym, zones))

    slots = MAX_OPEN - len(book["open"])
    for _depth, sym, zones in sorted(candidates)[: max(0, min(slots, MAX_NEW_PER_NIGHT))]:
        last_date = str(load_prices(sym).index[-1].date())
        book["open"].append(
            {
                "symbol": sym,
                "signal_date": last_date,
                "signal_close": round(zones["close"], 2),
                "dip_zone": round(zones["dip_zone"], 2),
                "top_zone": round(zones["top_zone"], 2),
                "entry_fill": None,
            }
        )
        alerts.append(
            f"PAPER entry intent {sym} at close {zones['close']:.2f} "
            f"(dip zone {zones['dip_zone']:.2f}, target {zones['top_zone']:.2f}); "
            "fills next open, EUR 500"
        )

    closed_rets = [p["net_return"] for p in book["closed"] if p.get("net_return") is not None]
    if closed_rets:
        lines.append(
            f"PAPER swing book: {len(book['closed'])} closed, mean {np.mean(closed_rets):+.2%}, "
            f"{len(book['open'])} open"
        )
    else:
        lines.append(f"PAPER swing book: {len(book['open'])} open, none closed yet")

    for line in lines + alerts:
        print(line)
    if not args.dry_run:
        BOOK_PATH.parent.mkdir(exist_ok=True)
        BOOK_PATH.write_text(json.dumps(book, indent=2))
        LOG_PATH.parent.mkdir(exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"===== swing_paper_book {date.today().isoformat()} =====\n")
            for line in lines + alerts:
                fh.write(line + "\n")
        if alerts:
            notify(alerts)
    elif alerts:
        print("[dry-run] would send:\n" + "\n".join(alerts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
