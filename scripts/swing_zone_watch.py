"""Nightly display-only watcher: alerts when a swing-cycle survivor enters its dip zone.

Watches the fuzzy-zone oscillators found by scripts/swing_cycle_scan.py
(watchlist artifacts/swing_zones.json, human-editable) and sends a Telegram +
toast alert when a symbol's close first drops into its historical trough zone
(close <= trough_zone * 1.10).

Honest caveats:
- DISPLAY-ONLY, permanently informational unless a future survivor-bar study
  says otherwise: scripts/stock_range_study.py measured that buying the
  "usual dip zone" of a ranging stock earned EXACTLY the unconditional
  baseline (zero edge). These alerts are context for a human, never tickets.
- Zones come from a historical scan and go stale: rerun swing_cycle_scan
  periodically (the alert text shows the watchlist date).
- No live orders ever (repo covenant).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PRICES = ROOT / "data" / "curated" / "prices"
ZONES_PATH = ROOT / "artifacts" / "swing_zones.json"
STATE_PATH = ROOT / "artifacts" / "swing_zone_state.json"
LOG_PATH = ROOT / "logs" / "swing_zone_watch.log"
TOAST_SCRIPT = ROOT / "scripts" / "notify_toast.ps1"

from swing_zone_strategy_study import (  # noqa: E402
    ENTRY_TOL,
    STOP_TOL,
    TARGET_TOL,
)

PLAN_PATH = ROOT / "artifacts" / "tranche_plan.json"
COOLDOWN_SESSIONS = 10
DEFAULT_EUR = 500  # paper-book sizing; override per symbol via tranche_plan.json "SWING" key


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        return None


def live_zones(symbol: str, theta_pct: int) -> dict | None:
    """Point-in-time zone refresh at the watchlist entry's own swing scale."""
    import numpy as np
    from swing_cycle_scan import load_prices
    from swing_zone_strategy_study import (
        MIN_PIVOTS_TRAILING,
        MIN_ZONE_GAP,
        TRAILING,
        zigzag_confirmed,
    )

    px = load_prices(symbol)
    if px is None or len(px) <= TRAILING:
        return None
    lp = np.log(px.to_numpy(dtype=float))
    recent = [pv for pv in zigzag_confirmed(lp, theta_pct / 100) if pv[1] > len(lp) - TRAILING]
    troughs = [pv[0] for pv in recent if pv[2] == "T"][-3:]
    peaks = [pv[0] for pv in recent if pv[2] == "P"][-3:]
    if len(recent) < MIN_PIVOTS_TRAILING or not troughs or not peaks:
        return None
    dip = float(np.exp(np.median(lp[troughs])))
    top = float(np.exp(np.median(lp[peaks])))
    if top / dip < MIN_ZONE_GAP:
        return None
    return {"trough_zone": round(dip, 2), "peak_zone": round(top, 2)}


def last_close(symbol: str) -> tuple[str, float, int] | None:
    """(date, adj_close, session_count) of the latest curated bar."""
    path = PRICES / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["date", "adj_close"])
    if not len(df):
        return None
    return (
        str(pd.Timestamp(df["date"].iloc[-1]).date()),
        float(df["adj_close"].iloc[-1]),
        len(df),
    )


def _eur_size(symbol: str) -> int:
    try:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8")) if PLAN_PATH.exists() else {}
        return int(plan.get(symbol, {}).get("SWING", DEFAULT_EUR))
    except (OSError, ValueError):
        return DEFAULT_EUR


def zone_alert_text(entry: dict, close: float, updated: str) -> str:
    dip, top = entry["trough_zone"], entry["peak_zone"]
    target = top * TARGET_TOL
    upside = target / close - 1 if close > 0 else 0.0
    return (
        f"BUY WINDOW: {entry['symbol']} closed {close:.2f} inside its dip zone "
        f"(buy at <= {dip * ENTRY_TOL:.2f}). Tested plan: buy ~EUR {_eur_size(entry['symbol'])} "
        f"at next open; SELL TARGET ~{target:.2f} ({upside:+.0%}); HARD EXIT if close drops "
        f"below {dip * STOP_TOL:.2f} (zone broken); TIME EXIT after ~4 weeks either way. "
        f"Edge ~+1-3%/trade historically, unproven live (paper gate pending) - no "
        f"guarantees, only an edge; your decision. Zones as of {updated}."
    )


def evaluate(
    entries: list[dict], state: dict, closes: dict[str, tuple[str, float, int]], updated: str
) -> list[str]:
    """Pure transition logic: returns alert texts, mutates state."""
    alerts: list[str] = []
    for entry in entries:
        sym = entry["symbol"]
        info = closes.get(sym)
        if info is None:
            continue
        bar_date, close, session_no = info
        dip = entry["trough_zone"]
        in_zone = dip * STOP_TOL <= close <= dip * ENTRY_TOL  # the TESTED buy window
        sym_state = state.setdefault(sym, {"in_zone": False, "last_alert_session": -(10**9)})
        if (
            in_zone
            and not sym_state["in_zone"]
            and session_no - sym_state["last_alert_session"] >= COOLDOWN_SESSIONS
        ):
            alerts.append(zone_alert_text(entry, close, updated))
            sym_state["last_alert_session"] = session_no
        sym_state["in_zone"] = in_zone
        sym_state["last_bar"] = bar_date
    return alerts


def notify(alerts: list[str]) -> None:
    body = "; ".join(alerts)[:250]
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
                "EdgeStack swing zone",
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

        notify_telegram.send("EdgeStack swing zone\n" + "\n".join(alerts)[:3800])
    except Exception as exc:
        print(f"WARN telegram failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="evaluate but never send/persist")
    args = parser.parse_args()

    zones = load_json(ZONES_PATH)
    if not zones or not zones.get("entries"):
        print("no swing zone watchlist; run scripts/swing_cycle_scan.py first")
        return 0
    state = load_json(STATE_PATH) or {}
    closes: dict[str, tuple[str, float, int]] = {}
    for entry in zones["entries"]:
        try:
            info = last_close(entry["symbol"])
        except Exception as exc:
            print(f"WARN {entry['symbol']}: load failed ({exc})")
            continue
        if info:
            closes[entry["symbol"]] = info
            fresh = live_zones(entry["symbol"], entry.get("theta_pct", 5))
            if fresh:
                entry.update(fresh)  # levels track the market; stored zones are fallback
            buy_at = entry["trough_zone"] * ENTRY_TOL
            in_window = entry["trough_zone"] * STOP_TOL <= info[1] <= buy_at
            print(
                f"{entry['symbol']:11s} close {info[1]:>9.2f} | buy <= {buy_at:>9.2f} "
                f"| {'BUY WINDOW' if in_window else 'waiting'}"
            )

    alerts = evaluate(zones["entries"], state, closes, zones.get("updated", "?"))
    if alerts and not args.dry_run:
        notify(alerts)
        print("\n" + "\n".join(alerts))
    elif alerts:
        print("[dry-run] would send:\n" + "\n".join(alerts))
    else:
        print("no new zone entries")
    if not args.dry_run:
        STATE_PATH.parent.mkdir(exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2))
        ZONES_PATH.write_text(json.dumps(zones, indent=2))  # persist refreshed levels
        LOG_PATH.parent.mkdir(exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"===== swing_zone_watch {date.today().isoformat()} =====\n")
            for alert in alerts or ["no new zone entries"]:
                fh.write(alert + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
