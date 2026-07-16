"""Cross-market replication: does ensemble4 (trained ONLY on US data) survive
on international country ETFs it has never seen?

Also checks the two headline calendar effects per market: turn-of-month and
September weakness. Same protocol as the zoo: signals at close t earn t+1,
2 bps exposure-change costs, splits dev(<=2015) / val(2016-23) / holdout(24+).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import requests
from seasonality_scan import fetch

from edgestack.data.catalog import atomic_write_bytes
from edgestack.strategies import backtest_exposure, ensemble_exposure
from edgestack.validation.advanced_tests import newey_west_alpha
from edgestack.validation.metrics import sharpe_ratio

MARKETS = {
    "EWG": "Germany", "EWJ": "Japan", "EWU": "UK", "EWQ": "France",
    "EWL": "Switzerland", "EWA": "Australia", "EWC": "Canada",
    "EWY": "Korea", "EWT": "Taiwan", "EWH": "Hong Kong",
    "EWS": "Singapore", "EWZ": "Brazil",
}
SPLITS = {"dev_pre2016": ("1996-01-01", "2015-12-31"),
          "val_2016_2023": ("2016-01-01", "2023-12-31"),
          "holdout_2024": ("2024-01-01", "2026-12-31")}
CACHE = Path("data/cache/zoo")


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

    rows = []
    for sym, name in MARKETS.items():
        f = CACHE / f"{sym}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
        else:
            df = fetch(session, sym, interval="1d", period1=0,
                       period2=int(time.time()))
            df = df[["dt", "open", "high", "low", "close", "adj"]]
            df.to_parquet(f, index=False)
            time.sleep(0.3)
        df = (df.assign(date=df["dt"].dt.tz_localize(None).dt.normalize())
                .set_index("date").drop(columns="dt"))
        ret = df["adj"].pct_change()
        pos = ensemble_exposure(df)
        strat = backtest_exposure(pos, ret)

        row = {"symbol": sym, "market": name,
               "since": str(df.index[0].date())}
        passes = 0
        for split, (lo, hi) in SPLITS.items():
            s = strat.loc[lo:hi].dropna()
            if len(s) < 250:
                continue
            b = ret.loc[s.index].fillna(0.0)
            sh_s, sh_b = sharpe_ratio(s.to_numpy()), sharpe_ratio(b.to_numpy())
            row[split] = f"{sh_s:.2f}/{sh_b:.2f}"
            passes += int(sh_s >= sh_b)
        nw = newey_west_alpha(strat.dropna(), ret.loc[strat.dropna().index].fillna(0.0))
        row["alpha_ann"] = round(nw["alpha_ann"], 4)
        row["alpha_t"] = round(nw["alpha_t"], 2)
        row["splits_passed"] = passes
        row["SURVIVOR"] = bool(passes == 3 and nw["alpha_t"] >= 2.0)

        # calendar effects per market
        ym = df.index.to_period("M")
        tdom = pd.Series(df.groupby(ym).cumcount() + 1, index=df.index)
        tde = pd.Series(df.groupby(ym).cumcount(ascending=False) + 1,
                        index=df.index)
        tom = (tdom <= 3) | (tde == 1)
        sep = df.index.month == 9
        row["tom_bps"] = round(float(ret[tom].mean() * 1e4), 1)
        row["rest_bps"] = round(float(ret[~tom].mean() * 1e4), 1)
        row["sep_bps"] = round(float(ret[sep].mean() * 1e4), 1)
        row["nonsep_bps"] = round(float(ret[~sep].mean() * 1e4), 1)
        rows.append(row)

    print(f"{'mkt':<5}{'since':<12}{'dev':>11}{'val':>11}{'hold':>11}"
          f"{'a_t':>6}{'pass':>5}  {'ToM/rest':>12}  {'Sep/other':>12}  SURV")
    n_surv = 0
    for r in rows:
        n_surv += r["SURVIVOR"]
        print(f"{r['symbol']:<5}{r['since']:<12}"
              f"{r.get('dev_pre2016','--'):>11}{r.get('val_2016_2023','--'):>11}"
              f"{r.get('holdout_2024','--'):>11}{r['alpha_t']:>6.2f}"
              f"{r['splits_passed']:>5}  {r['tom_bps']:>5.1f}/{r['rest_bps']:<5.1f}"
              f"  {r['sep_bps']:>5.1f}/{r['nonsep_bps']:<5.1f}"
              f"  {'YES' if r['SURVIVOR'] else ''}")
    tom_pos = sum(1 for r in rows if r["tom_bps"] > r["rest_bps"])
    sep_neg = sum(1 for r in rows if r["sep_bps"] < r["nonsep_bps"])
    print(f"\nensemble4 survivors abroad: {n_surv}/{len(rows)}   "
          f"ToM>rest: {tom_pos}/{len(rows)} markets   "
          f"Sep<other: {sep_neg}/{len(rows)} markets")
    atomic_write_bytes(Path("artifacts") / "xmarket_replication.json",
                       json.dumps(rows, indent=1).encode())
    print("saved -> artifacts/xmarket_replication.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
