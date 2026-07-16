"""Historical unified-book research; previously accessed and non-promotable in V2.

Permanent-portfolio core + legacy ensemble4-on-QQQ satellite as one
vol-targeted portfolio, constructed jointly.

Execution realism per execution_sensitivity.py: the satellite uses NEXT-OPEN
fills (the honest model — MOC can't fill at the signal close) at 2 bps; the
core rebalances on 5% weight bands at 5 bps.

Selection protocol (no holdout contamination): the (core split, vol target)
combination is chosen on dev(2005-2015)+val(2016-2023) Sharpe only; the
2024+ holdout is reported for the SELECTED combo afterwards. All 9 combos
are printed for transparency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from execution_sensitivity import load, strat_returns

from edgestack.data.catalog import atomic_write_bytes
from edgestack.strategies import ensemble_exposure
from edgestack.validation.advanced_tests import newey_west_alpha
from edgestack.validation.metrics import max_drawdown, sharpe_ratio

CORE_ASSETS = ("SPY", "TLT", "SHY", "GLD")
SPLITS = {
    "dev": ("2005-01-01", "2015-12-31"),
    "val": ("2016-01-01", "2023-12-31"),
    "hold": ("2024-01-01", "2026-12-31"),
}
CORE_COST = 0.0005
BAND = 0.05


def core_returns() -> pd.Series:
    rets = pd.DataFrame({s: load(s)["adj"].pct_change() for s in CORE_ASSETS})
    rets = rets.dropna()
    w = np.full(4, 0.25)
    out = np.zeros(len(rets))
    r = rets.to_numpy()
    for i in range(len(rets)):
        gross = float(w @ r[i])
        w = w * (1 + r[i]) / (1 + gross)
        cost = 0.0
        if np.abs(w - 0.25).max() > BAND:
            cost = CORE_COST * float(np.abs(w - 0.25).sum())
            w = np.full(4, 0.25)
        out[i] = gross - cost
    return pd.Series(out, index=rets.index)


def main() -> int:
    core = core_returns()
    qqq = load("QQQ")
    sat = strat_returns(qqq, ensemble_exposure(qqq), "next-open", 0.0002)
    spy_ret = load("SPY")["adj"].pct_change()

    idx = core.index.intersection(sat.dropna().index)
    core, sat = core.loc[idx], sat.loc[idx]
    spy = spy_ret.loc[idx].fillna(0.0)

    def evaluate(book: pd.Series) -> dict:
        row = {}
        for k, (lo, hi) in SPLITS.items():
            seg = book.loc[lo:hi].dropna()
            row[k] = {
                "sharpe": round(sharpe_ratio(seg.to_numpy()), 2),
                "cagr": round(float(np.prod(1 + seg)) ** (252 / len(seg)) - 1, 4),
                "maxdd": round(max_drawdown(seg.to_numpy()), 3),
            }
        nw = newey_west_alpha(book.dropna(), spy.loc[book.dropna().index])
        row["alpha_ann"] = round(nw["alpha_ann"], 4)
        row["alpha_t"] = round(nw["alpha_t"], 2)
        row["beta"] = round(nw["beta"], 2)
        return row

    results = {}
    print(
        f"{'combo':<26}{'dev':>7}{'val':>7}{'hold':>7}{'holdDD':>8}{'a/yr':>7}{'t':>6}{'beta':>6}"
    )
    for name, series in (("SPY buy&hold", spy), ("core alone", core), ("satellite alone", sat)):
        r = evaluate(series)
        results[name] = r
        print(
            f"{name:<26}{r['dev']['sharpe']:>7.2f}{r['val']['sharpe']:>7.2f}"
            f"{r['hold']['sharpe']:>7.2f}{r['hold']['maxdd']:>8.1%}"
            f"{r['alpha_ann']:>7.1%}{r['alpha_t']:>6.2f}{r['beta']:>6.2f}"
        )

    grid = {}
    for split in (0.8, 0.7, 0.6):
        raw = split * core + (1 - split) * sat
        for tgt in (None, 0.08, 0.10):
            if tgt is None:
                book, label = raw, f"book {int(split * 100)}/{int((1 - split) * 100)}"
            else:
                vol = raw.rolling(60).std() * np.sqrt(252)
                k = (tgt / vol).clip(upper=1.5).shift(1)
                book = (k * raw - 0.0002 * k.diff().abs().fillna(0.0)).dropna()
                label = f"book {int(split * 100)}/{int((1 - split) * 100)} vt{int(tgt * 100)}"
            r = evaluate(book)
            r["devval_sharpe"] = round(
                sharpe_ratio(book.loc["2005-01-01":"2023-12-31"].dropna().to_numpy()), 3
            )
            grid[label] = r
            results[label] = r
            print(
                f"{label:<26}{r['dev']['sharpe']:>7.2f}{r['val']['sharpe']:>7.2f}"
                f"{r['hold']['sharpe']:>7.2f}{r['hold']['maxdd']:>8.1%}"
                f"{r['alpha_ann']:>7.1%}{r['alpha_t']:>6.2f}{r['beta']:>6.2f}"
            )

    chosen = max(grid, key=lambda k: grid[k]["devval_sharpe"])
    print(f"\nSELECTED on dev+val only: {chosen}")
    c = grid[chosen]
    print(
        f"historical 2024+ partition (previously accessed): Sharpe {c['hold']['sharpe']}, "
        f"CAGR {c['hold']['cagr']:+.1%}, maxDD {c['hold']['maxdd']:.1%}  |  "
        f"pooled alpha {c['alpha_ann']:+.1%}/yr t={c['alpha_t']}"
    )
    results["SELECTED"] = chosen
    atomic_write_bytes(
        Path("artifacts") / "unified_book.json", json.dumps(results, indent=1).encode()
    )
    print("saved -> artifacts/unified_book.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
