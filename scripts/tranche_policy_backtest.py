"""Backtest the staged tranche-entry policy vs lump-sum and DCA.

Question: starting the moment a stock first closes 40% below its 52-week
high (the situation ACN/CTSH are in), does the staged T1/T2/T3/CAL plan
from tranche_watch.py beat (a) lump-sum at the episode start and (b) 12
monthly DCA buys?

Episodes: every curated stock, first close with drawdown < -40% (252d
rolling high, >=200 prior sessions); a stock can start a new episode only
after its drawdown recovers above -20%. Fills at the NEXT session's close
(zoo timing convention). Uninvested cash earns 0. Costs ignored (both
policy and baselines trade a handful of times; 2-5 bps does not move the
comparison).

Policy tranches (25% each, each fires once at first occurrence):
  T1 dip     RSI(2)<10 or 3 consecutive down days or IBS<0.2
  T2 repair  2 of 3 (close>SMA50, MACD>signal, vol20<30%) held 5 sessions
  T3 trend   close > SMA200
  CAL        first session of a November after episode start

Honest limitations: episodes cluster in 2020/2022 (not independent
samples); stocks whose data ends before the horizon are valued at their
last available price (delisting proceeds unknown) and counted; results
are previously-accessed history, not a promotable edge.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "data" / "curated" / "prices"
OUT = ROOT / "artifacts" / "tranche_policy_backtest.json"

HORIZONS = {"12m": 252, "24m": 504}
MIN_FWD = 200
TRANCHE_W = 0.25


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def prep(df: pd.DataFrame) -> pd.DataFrame:
    c, h, low = df["close"], df["high"], df["low"]
    adj = df["adj_close"]
    ret = adj.pct_change()
    out = pd.DataFrame(index=df.index)
    out["adj"] = adj
    out["dvol"] = c * df["volume"]
    out["dd"] = c / c.rolling(252, min_periods=200).max() - 1
    r2 = _rsi(c, 2)
    ibs = ((c - low) / (h - low).replace(0, np.nan)).fillna(0.5)
    down3 = (ret < 0) & (ret.shift() < 0) & (ret.shift(2) < 0)
    out["t1"] = (r2 < 10) | down3 | (ibs < 0.2)
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    macd = _ema(c, 12) - _ema(c, 26)
    repair2of3 = (
        (c > s50).astype(int) + (macd > _ema(macd, 9)).astype(int) + (vol20 < 0.30).astype(int)
    ) >= 2
    out["t2"] = repair2of3.rolling(5).min() >= 1
    out["t3"] = c > s200
    out["nov1"] = (out.index.month == 11) & (
        pd.Series(out.index, index=out.index).groupby(out.index.to_period("M")).cumcount() == 0
    )
    return out


def episodes(dd: pd.Series) -> list[int]:
    """Integer positions where a -40% episode starts (rearm above -20%)."""
    starts, armed = [], True
    vals = dd.to_numpy()
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            continue
        if armed and v < -0.40:
            starts.append(i)
            armed = False
        elif not armed and v > -0.20:
            armed = True
    return starts


def simulate(p: pd.DataFrame, start: int, horizon: int) -> dict | None:
    """Terminal values of policy / lump-sum / DCA for one episode."""
    n = len(p)
    end = min(start + horizon, n - 1)
    if end - start < MIN_FWD:
        return None
    adj = p["adj"].to_numpy()
    # Delisted-shell residue (CBE, TIE, ...) shows $0.01-ish quotes and no
    # volume, producing fake 100x fills. An episode is investable only if it
    # stays above $1 adjusted and has real liquidity throughout.
    if np.nanmin(adj[start : end + 1]) < 1.0:
        return None
    if np.nanmedian(p["dvol"].to_numpy()[start : end + 1]) < 1e6:
        return None
    t_end = adj[end]

    def fill(i):  # signal at close i -> filled at close i+1
        return adj[i + 1] if i + 1 <= end else None

    # policy: each tranche once, at first trigger occurrence in (start, end)
    deployed, value_shares, fired_at = 0.0, 0.0, {}
    for name in ("t1", "t2", "t3", "nov1"):
        sig = p[name].to_numpy()
        idx = next((i for i in range(start, end) if sig[i]), None)
        px = fill(idx) if idx is not None else None
        if px:
            value_shares += TRANCHE_W * t_end / px
            deployed += TRANCHE_W
            fired_at[name] = idx - start
    policy = value_shares + (1.0 - deployed)

    lump = t_end / adj[start + 1]

    buys = [start + 1 + 21 * k for k in range(12) if start + 1 + 21 * k <= end]
    dca_invested = len(buys) / 12
    dca = sum((1 / 12) * t_end / adj[b] for b in buys) + (1.0 - dca_invested)

    return {
        "policy": policy,
        "lump": lump,
        "dca": dca,
        "deployed": deployed,
        "fired_at": fired_at,
        "truncated": end < start + horizon,
    }


def main() -> int:
    rows = []
    for f in sorted(glob.glob(str(PRICES / "*.parquet"))):
        sym = Path(f).stem
        df = pd.read_parquet(f)
        if len(df) < 500:
            continue
        df = df.assign(date=pd.to_datetime(df["date"])).set_index("date").sort_index()
        # Corrupted series (TIE alternates $2 / $14,000 bars; CBE quotes $0.01
        # shells) — no real equity moves +/-200% in a day, so treat any such
        # print as a broken symbol and skip it wholesale.
        if df["adj_close"].pct_change().abs().max() > 2.0:
            continue
        p = prep(df)
        for start in episodes(p["dd"]):
            for hlab, h in HORIZONS.items():
                r = simulate(p, start, h)
                if r:
                    rows.append({"symbol": sym, "date": str(p.index[start].date()), "h": hlab, **r})
    res = pd.DataFrame(rows)
    report = {}
    print("Staged tranche policy vs lump-sum vs DCA — all -40% drawdown episodes")
    print("universe: curated catalog; episodes rearm above -20% dd\n")
    for hlab in HORIZONS:
        d = res[res.h == hlab]
        stats = {}
        for col in ("policy", "lump", "dca"):
            x = d[col] - 1
            stats[col] = {
                "median": float(x.median()),
                "mean": float(x.mean()),
                "p10": float(x.quantile(0.10)),
                "p90": float(x.quantile(0.90)),
            }
        stats["n_episodes"] = len(d)
        stats["n_truncated"] = int(d["truncated"].sum())
        stats["policy_beats_lump"] = float((d.policy > d.lump).mean())
        stats["policy_beats_dca"] = float((d.policy > d.dca).mean())
        stats["avg_deployed"] = float(d.deployed.mean())
        report[hlab] = stats
        print(f"--- horizon {hlab}  (n={len(d)} episodes, {stats['n_truncated']} truncated)")
        print(f"{'':<10}{'median':>9}{'mean':>9}{'p10':>9}{'p90':>9}")
        for col in ("policy", "lump", "dca"):
            s = stats[col]
            print(f"  {col:<8}{s['median']:>9.1%}{s['mean']:>9.1%}{s['p10']:>9.1%}{s['p90']:>9.1%}")
        print(
            f"  policy beats lump-sum in {stats['policy_beats_lump']:.0%} of episodes, "
            f"beats DCA in {stats['policy_beats_dca']:.0%}; "
            f"avg deployed {stats['avg_deployed']:.0%}\n"
        )

    # tranche timing profile (24m episodes)
    d24 = res[res.h == "24m"]
    fired = pd.DataFrame(list(d24["fired_at"]))
    print("Tranche firing profile (24m episodes): % fired, median sessions to fire")
    for name, label in [
        ("t1", "T1 dip"),
        ("t2", "T2 repair"),
        ("t3", "T3 trend"),
        ("nov1", "CAL nov"),
    ]:
        col = fired[name] if name in fired else pd.Series(dtype=float)
        print(
            f"  {label:<10} fired {col.notna().mean():.0%} of episodes, "
            f"median {col.median():.0f} sessions after -40% touch"
        )
        report.setdefault("firing", {})[name] = {
            "fired_pct": float(col.notna().mean()),
            "median_sessions": float(col.median()) if col.notna().any() else None,
        }

    # the two names in question, historical episodes
    print("\nACN/CTSH own historical episodes (24m):")
    for _, r in d24[d24.symbol.isin(["ACN", "CTSH"])].iterrows():
        trunc = ", truncated" if r.truncated else ""
        print(
            f"  {r.symbol} {r.date}: policy {r.policy - 1:+.1%}  lump {r.lump - 1:+.1%}  "
            f"dca {r.dca - 1:+.1%}  (deployed {r.deployed:.0%}{trunc})"
        )

    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
