"""Pre-close heads-up: would a tranche T1 dip trigger fire at today's price?

Runs ~15 minutes before the US close (Task Scheduler, 21:45 CET/CEST on
weekdays). Fetches today's partial daily bar from Yahoo's chart endpoint and
evaluates the T1 dip conditions with the live price as a synthetic close.
If a trigger would fire, pushes a Telegram + toast heads-up so the buy
decision can be made the same evening instead of the morning after.

This is a HEADS-UP, not a new signal: the canonical signal remains the
nightly watcher on the real close (zoo timing: signal at close t, fill t+1).
A late-session rally can invalidate the provisional trigger — the message
says so. Earnings blackouts are honored (no heads-up within 3 days before a
cached print date). Deduped per (symbol, day) via artifacts/precheck_state.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from tranche_watch import EARNINGS_BLACKOUT_DAYS, SYMBOLS, notify  # noqa: E402

STATE_PATH = ROOT / "artifacts" / "precheck_state.json"
EARNINGS_CACHE = ROOT / "artifacts" / "earnings_cache.json"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_daily_with_today(symbol: str) -> pd.DataFrame:
    """Recent daily bars; during the session the last row is today's partial."""
    resp = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={"range": "1y", "interval": "1d", "includePrePost": "false"},
        headers=UA,
        timeout=20,
    )
    resp.raise_for_status()
    res = resp.json()["chart"]["result"][0]
    quote = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(res["timestamp"], unit="s", utc=True),
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
        }
    ).dropna()
    df["et_date"] = df["ts"].dt.tz_convert(ZoneInfo("America/New_York")).dt.date
    return df


def rsi2(close: pd.Series) -> float:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    series = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return float(series.iloc[-1])


def provisional_t1(symbol: str, df: pd.DataFrame) -> tuple[bool, str]:
    c = df["close"].reset_index(drop=True)
    last = df.iloc[-1]
    ibs = (
        (last["close"] - last["low"]) / (last["high"] - last["low"])
        if last["high"] > last["low"]
        else 0.5
    )
    rets = c.pct_change()
    down3 = bool((rets.iloc[-3:] < 0).all())
    r2 = rsi2(c)
    if symbol == "SPY":
        calm = bool(len(c) >= 200 and c.iloc[-1] > c.rolling(200).mean().iloc[-1])
        calm = calm and float(rets.rolling(20).std().iloc[-1]) * (252**0.5) < 0.30
        dip = r2 < 10 or down3 or ibs < 0.2
        return calm and dip, f"calm={calm} RSI2={r2:.0f} down3={down3} IBS={ibs:.2f}"
    if symbol == "ACN":
        return r2 < 10, f"RSI2={r2:.0f} (<10 fires)"
    return down3 or ibs < 0.2, f"down3={down3} IBS={ibs:.2f} (<0.2 fires)"


def in_earnings_blackout(symbol: str, today: date) -> bool:
    if not EARNINGS_CACHE.exists():
        return False
    entry = json.loads(EARNINGS_CACHE.read_text()).get(symbol, {})
    if not entry.get("date"):
        return False
    days_to = (date.fromisoformat(entry["date"]) - today).days
    return 0 <= days_to <= EARNINGS_BLACKOUT_DAYS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="evaluate but never send")
    args = parser.parse_args()

    now_et = datetime.now(ZoneInfo("America/New_York"))
    today_et = now_et.date()
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    heads_up: list[str] = []

    for symbol in SYMBOLS:
        try:
            df = fetch_daily_with_today(symbol)
        except Exception as exc:
            print(f"WARN {symbol}: fetch failed ({exc})")
            continue
        if df.iloc[-1]["et_date"] != today_et:
            print(f"{symbol}: no live bar for {today_et} — market closed; skipping")
            continue
        fired, detail = provisional_t1(symbol, df)
        print(f"{symbol} provisional T1 {'FIRES' if fired else '-'}  {detail}")
        if not fired:
            continue
        if in_earnings_blackout(symbol, today_et):
            print(f"{symbol}: suppressed (earnings blackout)")
            continue
        key = f"{symbol}:{today_et.isoformat()}"
        if key in state:
            continue
        state[key] = 1
        heads_up.append(
            f"HEADS-UP {symbol}: T1 dip likely fires at today's close ({detail}). "
            "Plan: buy next open IF the nightly alert confirms. A late rally can cancel this."
        )

    if heads_up and not args.dry_run:
        notify(heads_up)
        STATE_PATH.write_text(json.dumps(state, indent=2))
    elif heads_up:
        print("[dry-run] would send:\n" + "\n".join(heads_up))
    else:
        print("no provisional firings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
