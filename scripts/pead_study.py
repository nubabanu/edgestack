"""PEAD demo study on EDGAR-stamped earnings events (descriptive only).

Construction (Bernard & Thomas seasonal random walk — no analyst data):
  SUE_q = (EPS_q - EPS_{q-4}) / std(trailing 8 seasonal diffs, min 6)
Entry: adjusted OPEN of the first session strictly AFTER the 8-K
acceptance date (US/Eastern) — causal by construction, sacrifices
pre-market releases' same-day open. Forward adjusted open-to-open
returns at 1/5/20/60 sessions, excess vs SPY over the same windows.
Buckets: cross-sectional SUE terciles within each calendar quarter.

Data: data/curated/events/*.parquet (edgar_earnings.py) joined to
data/curated/prices/*.parquet. Results are previously-accessed history
over a hand-picked liquid universe — NOT V2 promotion evidence; the
2024+ split is reported for consistency only.
"""

from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "curated" / "events"
PRICES = ROOT / "data" / "curated" / "prices"
OUT = ROOT / "artifacts" / "pead_study.json"
HORIZONS = (1, 5, 20, 60)
SPLITS = {
    "pre2016": (None, "2015-12-31"),
    "2016_2023": ("2016-01-01", "2023-12-31"),
    "2024plus": ("2024-01-01", None),
}


def adj_open(df: pd.DataFrame) -> pd.Series:
    return df["open"] * df["adj_close"] / df["close"]


def load_prices(sym: str) -> pd.DataFrame | None:
    f = PRICES / f"{sym}.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f).assign(date=lambda d: pd.to_datetime(d["date"]))
    df = df.set_index("date").sort_index()
    df["aopen"] = adj_open(df)
    return df


def sue_series(ev: pd.DataFrame) -> pd.DataFrame:
    q = (
        ev.dropna(subset=["eps", "period_end", "accepted_at"])
        .drop_duplicates(subset=["period_end"])
        .sort_values("period_end")
        .reset_index(drop=True)
    )
    diffs, sues = [], []
    for i in range(len(q)):
        if i < 4:
            sues.append(np.nan)
            continue
        gap = (q.loc[i, "period_end"] - q.loc[i - 4, "period_end"]).days
        if not 330 <= gap <= 400:  # non-contiguous fiscal history
            sues.append(np.nan)
            continue
        diff = q.loc[i, "eps"] - q.loc[i - 4, "eps"]
        diffs.append(diff)
        hist = diffs[-9:-1]  # trailing 8 seasonal diffs, excluding current
        sues.append(diff / np.std(hist, ddof=1) if len(hist) >= 6 and np.std(hist) > 0 else np.nan)
    q["sue"] = sues
    return q.dropna(subset=["sue"])


def main() -> int:
    spy = load_prices("SPY")
    rows = []
    for f in sorted(EVENTS.glob("*.parquet")):
        sym = f.stem
        px = load_prices(sym)
        if px is None:
            continue
        q = sue_series(pd.read_parquet(f))
        for _, ev in q.iterrows():
            et_date = ev["accepted_at"].tz_convert(ZoneInfo("America/New_York")).normalize()
            entries = px.index[px.index > et_date.tz_localize(None)]
            if len(entries) < max(HORIZONS) + 2:
                continue
            e0 = entries[0]
            i0 = px.index.get_loc(e0)
            s0 = spy.index.get_indexer([e0], method="nearest")[0]
            row = {"symbol": sym, "entry": e0, "sue": float(ev["sue"])}
            for h in HORIZONS:
                r = px["aopen"].iloc[i0 + h] / px["aopen"].iloc[i0] - 1
                rb = spy["aopen"].iloc[s0 + h] / spy["aopen"].iloc[s0] - 1
                row[f"exc{h}"] = float(r - rb)
            rows.append(row)
    d = pd.DataFrame(rows)
    d["quarter"] = d["entry"].dt.to_period("Q")
    d["bucket"] = d.groupby("quarter")["sue"].transform(
        lambda s: pd.qcut(s, 3, labels=["low", "mid", "high"], duplicates="drop")
    )
    span = f"{d.entry.min():%Y-%m}..{d.entry.max():%Y-%m}"
    print(f"{len(d)} events, {d.symbol.nunique()} symbols, {span}\n")

    report: dict = {"n_events": len(d)}
    print("Post-announcement EXCESS return vs SPY (adjusted open-to-open):")
    hdr = f"{'horizon':>8} {'low-SUE':>9} {'high-SUE':>9} {'spread':>8} {'t':>6}"
    print(hdr + "  split-consistency")
    for h in HORIZONS:
        lo = d[d.bucket == "low"][f"exc{h}"]
        hi = d[d.bucket == "high"][f"exc{h}"]
        spread = hi.mean() - lo.mean()
        se = np.sqrt(hi.var() / len(hi) + lo.var() / len(lo))
        cons = []
        for lo_d, hi_d in SPLITS.values():
            m = d
            if lo_d:
                m = m[m.entry >= lo_d]
            if hi_d:
                m = m[m.entry <= hi_d]
            s = m[m.bucket == "high"][f"exc{h}"].mean() - m[m.bucket == "low"][f"exc{h}"].mean()
            cons.append(s > 0)
        report[f"h{h}"] = {
            "low": float(lo.mean()),
            "high": float(hi.mean()),
            "spread": float(spread),
            "t": float(spread / se),
            "splits_positive": int(sum(cons)),
        }
        print(
            f"{h:>7}d {lo.mean():>9.2%} {hi.mean():>9.2%} {spread:>8.2%} {spread / se:>6.2f}"
            f"  {sum(cons)}/3 splits positive"
        )
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {OUT}")
    print("Descriptive only: hand-picked surviving mega-caps, previously-accessed history.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
