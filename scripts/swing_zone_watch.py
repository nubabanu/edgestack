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

ZONE_TOL = 1.10  # in-zone when close <= trough_zone * ZONE_TOL
COOLDOWN_SESSIONS = 10


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        return None


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


def zone_alert_text(entry: dict, close: float, updated: str) -> str:
    upside = entry["peak_zone"] / close - 1 if close > 0 else 0.0
    return (
        f"SWING ZONE (DISPLAY-ONLY): {entry['symbol']} entered its historical dip zone "
        f"(close {close:.2f} vs trough zone ~{entry['trough_zone']}). Historical top zone "
        f"~{entry['peak_zone']} ({upside:+.0%} away), {entry['swings_per_year']} swings/yr "
        f"at the {entry['theta_pct']}% scale. No validated edge - the range study measured "
        f"ZERO floor-bounce edge; context only. Zones from scan of {updated}."
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
        in_zone = close <= entry["trough_zone"] * ZONE_TOL
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
            in_zone = info[1] <= entry["trough_zone"] * ZONE_TOL
            print(
                f"{entry['symbol']:11s} close {info[1]:>9.2f} vs zone "
                f"{entry['trough_zone']:>9} -> {'IN ZONE' if in_zone else 'above'}"
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
        LOG_PATH.parent.mkdir(exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"===== swing_zone_watch {date.today().isoformat()} =====\n")
            for alert in alerts or ["no new zone entries"]:
                fh.write(alert + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
