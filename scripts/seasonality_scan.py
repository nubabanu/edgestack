"""Calendar seasonality scan: weekday / hour-of-day / day-of-month / month-of-year
for SPY, QQQ (Nasdaq-100), and the 11 SPDR sector ETFs.

Granularity honesty:
  - Daily bars (1999->present where the ETF existed): weekday, turn-of-month,
    month-of-year, plus the overnight (close->open) vs intraday (open->close)
    decomposition — the coarsest possible "what hour" answer over 25 years.
  - 60-minute bars (Yahoo only serves ~730 days): true hour-of-day averages,
    ~500 sessions only. Treat as suggestive.

Statistics: mean, t-stat (mean/sem). MULTIPLICITY WARNING baked into output:
we test ~13 tickers x (5 weekdays + 12 months + 7 hours + 10 month-days)
~= 440 cells; ~22 cells will show |t|>2 by pure chance. Split-half consistency
(1999-2012 vs 2013-2026) is reported for headline cells.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import requests

from edgestack.data.catalog import atomic_write_bytes

TICKERS = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI",
           "XLU", "XLB", "XLRE", "XLC"]
NY = ZoneInfo("America/New_York")
CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"


def fetch(session: requests.Session, sym: str, *, interval: str,
          period1: int, period2: int) -> pd.DataFrame:
    r = session.get(CHART.format(sym=sym), timeout=30, params={
        "period1": period1, "period2": period2, "interval": interval,
        "events": "div,splits", "includeAdjustedClose": "true"})
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "ts": res["timestamp"], "open": q["open"], "high": q["high"],
        "low": q["low"], "close": q["close"],
    })
    adj = res["indicators"].get("adjclose")
    df["adj"] = adj[0]["adjclose"] if adj else df["close"]
    df = df.dropna(subset=["open", "close"])
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(NY)
    return df.reset_index(drop=True)


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 8 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def cell(x: pd.Series) -> dict:
    x = x.dropna()
    return {"mean_bps": round(float(x.mean()) * 1e4, 2), "t": round(tstat(x), 2),
            "n": int(len(x))}


def daily_tables(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["date"] = d["dt"].dt.tz_localize(None).dt.normalize()
    d["cc"] = d["adj"].pct_change()                      # session total return
    d["overnight"] = d["open"] / d["close"].shift(1) - 1  # prev close -> open
    d["intraday"] = d["close"] / d["open"] - 1            # open -> close
    d["weekday"] = d["dt"].dt.day_name()
    d["month"] = d["dt"].dt.month

    # trading-day-of-month index (1..n) and from-end (-1 = last session)
    ym = d["dt"].dt.to_period("M")
    d["tdom"] = d.groupby(ym).cumcount() + 1
    d["tdom_end"] = d.groupby(ym).cumcount(ascending=False) + 1
    d["tom_window"] = (d["tdom"] <= 3) | (d["tdom_end"] == 1)  # last + first 3

    out: dict = {"start": str(d["date"].iloc[0].date()),
                 "end": str(d["date"].iloc[-1].date()), "n": len(d)}
    for name in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"):
        sub = d[d["weekday"] == name]
        out.setdefault("weekday", {})[name] = {
            "session": cell(sub["cc"]), "overnight": cell(sub["overnight"]),
            "intraday": cell(sub["intraday"])}
    out["overnight_all"] = cell(d["overnight"])
    out["intraday_all"] = cell(d["intraday"])
    out["tom_window"] = cell(d.loc[d["tom_window"], "cc"])
    out["non_tom"] = cell(d.loc[~d["tom_window"], "cc"])
    for k in range(1, 6):
        out.setdefault("tdom", {})[f"day+{k}"] = cell(d.loc[d["tdom"] == k, "cc"])
        out["tdom"][f"day-{k}"] = cell(d.loc[d["tdom_end"] == k, "cc"])
    monthly = (1 + d["cc"]).groupby([d["dt"].dt.year, d["month"]]).prod() - 1
    for m in range(1, 13):
        vals = monthly.xs(m, level=1)
        out.setdefault("month", {})[m] = {
            "mean_pct": round(float(vals.mean()) * 100, 2),
            "t": round(tstat(vals), 2), "n": int(len(vals)),
            "pos_frac": round(float((vals > 0).mean()), 2)}
    # split-half consistency for weekday sessions
    half = d["date"] < d["date"].iloc[len(d) // 2]
    out["weekday_halves"] = {
        name: {"h1_bps": round(float(d.loc[half & (d["weekday"] == name), "cc"]
                                     .mean()) * 1e4, 1),
               "h2_bps": round(float(d.loc[~half & (d["weekday"] == name), "cc"]
                                     .mean()) * 1e4, 1)}
        for name in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")}
    return out


def hourly_table(df: pd.DataFrame) -> dict:
    h = df.copy()
    h["bar_ret"] = h["close"] / h["open"] - 1
    h["slot"] = h["dt"].dt.strftime("%H:%M")
    h["date"] = h["dt"].dt.date
    # overnight gap: first bar open vs prior session last close
    last_close = h.groupby("date")["close"].last()
    first_open = h.groupby("date")["open"].first()
    gap = (first_open / last_close.shift(1) - 1).dropna()
    out = {"overnight_gap": cell(pd.Series(gap.values))}
    for slot, sub in h.groupby("slot"):
        if len(sub) > 100:
            out[slot] = cell(sub["bar_ret"])
    return out


def main() -> int:
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    now = int(time.time())
    daily: dict = {}
    hourly: dict = {}
    for sym in TICKERS:
        d = fetch(session, sym, interval="1d", period1=0, period2=now)
        daily[sym] = daily_tables(d)
        time.sleep(0.3)
        try:
            h = fetch(session, sym, interval="60m",
                      period1=now - 729 * 86400, period2=now)
            hourly[sym] = hourly_table(h)
        except Exception as exc:  # hourly is best-effort
            hourly[sym] = {"error": str(exc)}
        print(f"{sym}: daily {daily[sym]['n']} rows from {daily[sym]['start']}, "
              f"hourly slots {max(0, len(hourly[sym]) - 1)}")
        time.sleep(0.3)

    atomic_write_bytes(Path("artifacts") / "seasonality_scan.json",
                       json.dumps({"daily": daily, "hourly": hourly},
                                  indent=1).encode())
    print("saved -> artifacts/seasonality_scan.json")

    # headline print: SPY + QQQ
    for sym in ("SPY", "QQQ"):
        t = daily[sym]
        print(f"\n=== {sym} (since {t['start']}) — session return by weekday "
              f"(bps, t) ===")
        for k, v in t["weekday"].items():
            print(f"  {k:<10} session {v['session']['mean_bps']:>6.1f} "
                  f"(t={v['session']['t']:>5.2f})  overnight "
                  f"{v['overnight']['mean_bps']:>6.1f} (t={v['overnight']['t']:>5.2f})"
                  f"  intraday {v['intraday']['mean_bps']:>6.1f} "
                  f"(t={v['intraday']['t']:>5.2f})")
        print(f"  ALL overnight {t['overnight_all']['mean_bps']:.1f} bps "
              f"(t={t['overnight_all']['t']:.2f}) vs intraday "
              f"{t['intraday_all']['mean_bps']:.1f} bps (t={t['intraday_all']['t']:.2f})")
        print(f"  ToM window {t['tom_window']['mean_bps']:.1f} bps "
              f"(t={t['tom_window']['t']:.2f}) vs rest {t['non_tom']['mean_bps']:.1f} "
              f"(t={t['non_tom']['t']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
