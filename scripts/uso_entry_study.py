"""USO buy-entry timing study: do the tranche dip/trend triggers transfer to oil?

Question (2026-07-21): should tranche_watch cover USO — i.e., after a
T1/T2/T3-style trigger fires on USO, are forward buy-and-hold returns better
than unconditional entry, the same gate the GO backtest applies to the
equity names?

Honest caveats, stated up front:
- Single previously-accessed series; the 2024+ "holdout" is previously
  accessed history and can never promote anything (repo convention).
- The triggers were developed on ACN/CTSH/EPAM/SPY equities in the 2026-07
  buy-timing research; testing them on oil is a transfer test with zero
  oil-specific tuning (which is the point — tuning here would be mining).
- USO bleeds futures roll costs; unconditional buy-and-hold is a weak
  baseline in contango regimes, which flatters any timing rule. Verdicts
  must beat the baseline in ALL splits, not on the pooled mean alone.
- Entry costs are identical across alternatives (one buy either way), so the
  2 bps convention cancels and is omitted.
- Overlapping forward windows inflate naive significance: triggers are
  deduped to the first firing per horizon-length cluster, and the t-stat is
  computed over those non-overlapping events only.

Method: signal at close t fills at close t+1 (repo convention); forward
return = adj[t+1+h]/adj[t+1] - 1. Splits: the go-backtest single-name splits
(dev 2011-2017, val 2018-2023, holdout 2024+); the curated USO series starts
2015, so dev is effectively 2015-2017. Gate per trigger/horizon: conditional
mean > unconditional mean in all three splits AND pooled non-overlapping
event t-stat >= 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "data" / "curated" / "prices"

# go-backtest splits (single-name convention). The curated USO series starts
# 2015-01-02, so the dev window is effectively 2015-2017 — thin; say so.
SPLITS = {
    "dev_2011_2017": ("2011-01-01", "2017-12-31"),
    "val_2018_2023": ("2018-01-01", "2023-12-31"),
    "holdout_2024_prev_accessed": ("2024-01-01", "2026-12-31"),
}
HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
MIN_EVENTS_PER_SPLIT = 5


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def load_uso() -> pd.DataFrame:
    df = pd.read_parquet(PRICES / "USO.parquet")
    df = df.assign(date=pd.to_datetime(df["date"])).set_index("date").sort_index()
    if df["adj_close"].pct_change().abs().max() > 2.0:
        raise SystemExit("USO series fails the corrupted-series screen; aborting")
    return df


def triggers(df: pd.DataFrame) -> dict[str, pd.Series]:
    c, low, high = df["close"], df["low"], df["high"]
    ret = df["adj_close"].pct_change()
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    r2 = _rsi(c, 2)
    ibs = ((c - low) / (high - low).replace(0, np.nan)).fillna(0.5)
    down3 = (ret < 0) & (ret.shift() < 0) & (ret.shift(2) < 0)
    dip = (r2 < 10) | down3 | (ibs < 0.2)

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_up = macd > macd.ewm(span=9, adjust=False).mean()
    two_of_three = ((c > sma50).astype(int) + macd_up.astype(int) + (vol20 < 0.30).astype(int)) >= 2
    repair_held = two_of_three.rolling(5).min().astype(bool)

    return {
        # T1 dip: raw dip trigger, as watched for ACN/CTSH.
        "t1_dip": dip,
        # SPY-style T1: dip only inside a calm uptrend regime.
        "t1_calm_dip": dip & (c > sma200) & (vol20 < 0.30),
        # T2 repair: first session where 2-of-3 has held 5 straight sessions.
        "t2_repair": repair_held & ~repair_held.shift(fill_value=False),
        # T3 trend: reclaim cross above the 200-DMA.
        "t3_trend_cross": (c > sma200) & (c.shift() <= sma200.shift()),
    }


def dedupe(mask: pd.Series, gap: int) -> pd.Series:
    """First firing per cluster: suppress re-fires within `gap` sessions."""
    out = mask.copy()
    last = -(10**9)
    values = mask.to_numpy()
    keep = np.zeros(len(values), dtype=bool)
    for i, fired in enumerate(values):
        if fired and i - last >= gap:
            keep[i] = True
            last = i
    out[:] = keep
    return out


def main() -> int:
    df = load_uso()
    adj = df["adj_close"]
    trig = triggers(df)
    report: dict = {
        "symbol": "USO",
        "sessions": len(df),
        "span": [str(df.index[0].date()), str(df.index[-1].date())],
        "splits": {k: list(v) for k, v in SPLITS.items()},
        "results": {},
        "survivors": [],
    }
    for hlab, h in HORIZONS.items():
        fwd = adj.shift(-1 - h) / adj.shift(-1) - 1  # fill at close t+1
        for name, mask in trig.items():
            events = dedupe(mask, h)
            row: dict = {}
            beats = 0
            pooled: list[float] = []
            pooled_uncond: list[float] = []
            for slab, (lo, hi) in SPLITS.items():
                in_split = (fwd.index >= lo) & (fwd.index <= hi)
                cond = fwd[in_split & events].dropna()
                uncond = fwd[in_split].dropna()
                row[slab] = {
                    "n": len(cond),
                    "mean_fwd": round(float(cond.mean()), 4) if len(cond) else None,
                    "uncond_mean": round(float(uncond.mean()), 4),
                }
                if len(cond) >= MIN_EVENTS_PER_SPLIT and cond.mean() > uncond.mean():
                    beats += 1
                pooled.extend((cond - uncond.mean()).tolist())
                pooled_uncond.append(float(uncond.mean()))
            excess = np.array(pooled)
            t_stat = (
                float(excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess))))
                if len(excess) > 2 and excess.std(ddof=1) > 0
                else 0.0
            )
            row["beats_unconditional_splits"] = beats
            row["pooled_events"] = len(excess)
            row["pooled_excess_mean"] = round(float(excess.mean()), 4) if len(excess) else None
            row["pooled_t"] = round(t_stat, 2)
            passed = beats == len(SPLITS) and t_stat >= 2
            row["gate"] = "PASS" if passed else "FAIL"
            if passed:
                report["survivors"].append(f"{name}@{hlab}")
            report["results"][f"{name}@{hlab}"] = row
    report["verdict"] = (
        "no trigger beats unconditional entry in all splits with t>=2; "
        "do NOT add USO to tranche_watch"
        if not report["survivors"]
        else f"candidates (needs review before watching): {report['survivors']}"
    )
    report["disclaimer"] = (
        "Historical research on previously-accessed data; holdout previously accessed; "
        "not investment advice."
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
