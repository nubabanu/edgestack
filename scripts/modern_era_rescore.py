"""Modern-era re-score of the REJECTED edge graveyard (2026-07-21).

Hypothesis under test: "markets changed after 2020, so rules rejected on
full-history splits might be genuinely good in the modern era." This script
re-scores the ENTIRE zoo rule library plus the breadth-timing suite on
post-2020 windows only:

  mod_a 2020-2022  (COVID crash, melt-up, 2022 bear)
  mod_b 2023-2026  (AI bull, previously accessed throughout)

Candidate bar: Sharpe >= buy-and-hold in BOTH modern windows AND pooled
2020+ Newey-West alpha t >= 2.

READ THIS BEFORE BELIEVING ANYTHING BELOW:
- Every row here is previously-accessed data. This is HYPOTHESIS GENERATION,
  never promotion evidence. ~460 trials re-tested on a window chosen after
  looking at it means the expected count of false candidates at t>=2 is
  ~11 BY CHANCE ALONE. A candidate list shorter than that is consistent
  with pure noise.
- The only honest promotion path for a "modern era" candidate is
  pre-registered forward paper tracking (the repo's paper book).
- 6.5 years is ~1.5 market cycles; regime claims on that sample are weak by
  construction.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from breadth_timing import PIT_EARLIEST, breadth_series, load_close_panel, membership_matrix
from breadth_timing import build_rules as breadth_build_rules
from strategy_zoo import TICKERS, build_rules, run_rule

from edgestack.data.catalog import atomic_write_bytes
from edgestack.data.universe_pit import PitSP500Universe
from edgestack.validation.advanced_tests import newey_west_alpha
from edgestack.validation.metrics import sharpe_ratio

ROOT = Path(__file__).resolve().parents[1]
ZOO_CACHE = ROOT / "data" / "cache" / "zoo"
PRICES = ROOT / "data" / "curated" / "prices"
OUT = ROOT / "artifacts" / "modern_era_rescore.json"

WINDOWS = {
    "mod_a_2020_2022": ("2020-01-01", "2022-12-31"),
    "mod_b_2023_2026": ("2023-01-01", "2026-12-31"),
}
POOLED_START = "2020-01-01"


def score(name: str, vehicle: str, pos: pd.Series, ret: pd.Series) -> dict:
    strat = run_rule(pos, ret)
    row: dict = {"rule": name, "vehicle": vehicle}
    ok = True
    for label, (lo, hi) in WINDOWS.items():
        s = strat.loc[lo:hi].dropna()
        bench = ret.loc[s.index].fillna(0.0)
        if len(s) < 200:
            ok = False
            continue
        sh_s, sh_b = sharpe_ratio(s.to_numpy()), sharpe_ratio(bench.to_numpy())
        row[label] = {"sharpe": round(sh_s, 2), "bh_sharpe": round(sh_b, 2)}
        if sh_s < sh_b:
            ok = False
    pooled = strat.loc[POOLED_START:].dropna()
    nw = newey_west_alpha(pooled, ret.loc[pooled.index].fillna(0.0))
    row["alpha_t_2020plus"] = round(nw["alpha_t"], 2)
    row["alpha_ann_2020plus"] = round(nw["alpha_ann"], 4)
    row["MODERN_CANDIDATE"] = bool(ok and nw["alpha_t"] >= 2.0)
    return row


def main() -> int:
    t0 = time.time()
    results, trials = [], 0

    # 1) The full zoo rule library on SPY/QQQ + sector ETFs (zoo cache).
    for sym in TICKERS:
        df = pd.read_parquet(ZOO_CACHE / f"{sym}.parquet")
        df = (
            df.assign(date=df["dt"].dt.tz_localize(None).dt.normalize())
            .set_index("date")
            .drop(columns="dt")
        )
        ret = df["adj"].pct_change()
        for name, pos in build_rules(df).items():
            trials += 1
            results.append(score(name, sym, pos, ret))

    # 2) The breadth suite on SPY/QQQ (curated catalog + PIT membership).
    panel = load_close_panel()
    universe = PitSP500Universe(ROOT / "data" / "cache" / "universe", earliest=PIT_EARLIEST)
    breadth = breadth_series(panel, membership_matrix(universe, panel))
    for sym in ("SPY", "QQQ"):
        df = pd.read_parquet(PRICES / f"{sym}.parquet")
        df = df.assign(date=pd.to_datetime(df["date"])).set_index("date").sort_index()
        close = df["close"].reindex(breadth.index)
        ret = df["adj_close"].reindex(breadth.index).pct_change()
        for name, pos in breadth_build_rules(breadth, close).items():
            trials += 1
            results.append(score(name, sym, pos, ret))

    candidates = [r for r in results if r["MODERN_CANDIDATE"]]
    expected_false = round(trials * 0.025, 1)
    report = {
        "campaign": "modern_era_rescore",
        "windows": {k: list(v) for k, v in WINDOWS.items()},
        "trials": trials,
        "expected_false_candidates_at_t2": expected_false,
        "n_candidates": len(candidates),
        "candidates": sorted(candidates, key=lambda r: -r["alpha_t_2020plus"]),
        "verdict_hint": (
            "candidates <= expected_false_candidates_at_t2 is consistent with noise; "
            "any candidate is a paper-book hypothesis only, never promotion evidence"
        ),
        "disclaimer": (
            "Every window is previously-accessed history chosen after observation; "
            "not investment advice."
        ),
    }
    atomic_write_bytes(OUT, json.dumps(report, indent=2).encode("utf-8"))

    print(f"trials: {trials}; expected false candidates at t>=2 by chance: ~{expected_false}")
    print(f"MODERN-ERA CANDIDATES (both windows >= B&H, pooled 2020+ t>=2): {len(candidates)}")
    for r in sorted(candidates, key=lambda x: -x["alpha_t_2020plus"]):
        a, b = r.get("mod_a_2020_2022", {}), r.get("mod_b_2023_2026", {})
        print(
            f"  {r['rule']:<30}{r['vehicle']:<6}"
            f"a {a.get('sharpe')}/{a.get('bh_sharpe')}  b {b.get('sharpe')}/{b.get('bh_sharpe')}  "
            f"alpha {r['alpha_ann_2020plus']:+.1%}/yr t={r['alpha_t_2020plus']:.2f}"
        )
    print(f"\nfull table -> {OUT} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
