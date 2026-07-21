"""Evidence-based re-weighting of the ensemble4 families (2026-07-21).

Prompt: the modern-era re-score confirmed the same four validated families
lead post-2020 — so weight them by evidence instead of equally.

Protocol (no holdout contamination, mirroring unified_book.py):
- Weight schemes are PRE-DECLARED below (no grid search): equal (incumbent),
  alpha-t-weighted, Sharpe-weighted, inverse-vol (risk parity), drop-worst.
- Per-family statistics used to derive weights come from dev(1999-2015) +
  val(2016-2023) ONLY.
- Selection rule: a challenger replaces equal-weight only if it beats
  equal's Sharpe in BOTH dev and val (on the primary vehicle, QQQ, at
  realistic NEXT-OPEN fills, 2 bps). The 2024+ holdout is reported for all
  schemes AFTER selection and is previously-accessed history.
- SPY is scored as robustness context, not selection input.

Honest note: weighting schemes derived from in-sample statistics routinely
lose to 1/N out of sample (DeMiguel et al. 2009); equal weight is the
incumbent for a reason. If nothing beats it in both splits, the verdict is
"keep equal weight" and that is a useful result, not a failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from execution_sensitivity import load, strat_returns

from edgestack.data.catalog import atomic_write_bytes
from edgestack.strategies import ENSEMBLE_FAMILIES, family_positions
from edgestack.validation.advanced_tests import newey_west_alpha
from edgestack.validation.metrics import max_drawdown, sharpe_ratio

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "ensemble_weights.json"

COST = 0.0002
FILL = "next-open"  # the honest execution model per execution_sensitivity.py
PRIMARY = "QQQ"
VEHICLES = ("QQQ", "SPY")
DEV = ("1999-01-01", "2015-12-31")
VAL = ("2016-01-01", "2023-12-31")
HOLD = ("2024-01-01", "2026-12-31")


def family_stats(df: pd.DataFrame) -> dict[str, dict]:
    """Per-family dev+val evidence at next-open fills (weight inputs)."""
    fams = family_positions(df)
    ret = df["adj"].pct_change()
    stats = {}
    for name in ENSEMBLE_FAMILIES:
        s = strat_returns(df, fams[name], FILL, COST)
        insample = pd.concat([s.loc[DEV[0] : DEV[1]], s.loc[VAL[0] : VAL[1]]]).dropna()
        nw = newey_west_alpha(insample, ret.loc[insample.index].fillna(0.0))
        stats[name] = {
            "alpha_t_devval": round(nw["alpha_t"], 2),
            "sharpe_devval": round(sharpe_ratio(insample.to_numpy()), 2),
            "daily_vol_devval": float(insample.std(ddof=1)),
        }
    return stats


def weight_schemes(stats: dict[str, dict]) -> dict[str, dict[str, float]]:
    names = list(ENSEMBLE_FAMILIES)
    t = {n: max(stats[n]["alpha_t_devval"], 0.0) for n in names}
    sh = {n: max(stats[n]["sharpe_devval"], 0.0) for n in names}
    iv = {n: 1.0 / stats[n]["daily_vol_devval"] for n in names}
    worst = min(names, key=lambda n: stats[n]["alpha_t_devval"])

    def norm(w: dict[str, float]) -> dict[str, float]:
        total = sum(w.values())
        return {k: round(v / total, 3) for k, v in w.items()}

    return {
        "equal": dict.fromkeys(names, 0.25),
        "alpha_t_weighted": norm(t),
        "sharpe_weighted": norm(sh),
        "inverse_vol": norm(iv),
        "drop_worst": norm({n: (0.0 if n == worst else 1.0) for n in names}),
    }


def score_scheme(df: pd.DataFrame, weights: dict[str, float]) -> dict:
    fams = family_positions(df)
    w = pd.Series(weights).reindex(fams.columns).fillna(0.0)
    pos = (fams * w).sum(axis=1) / w.sum()
    s = strat_returns(df, pos, FILL, COST)
    ret = df["adj"].pct_change()
    out = {}
    for label, (lo, hi) in {"dev": DEV, "val": VAL, "holdout_prev_accessed": HOLD}.items():
        seg = s.loc[lo:hi].dropna()
        bench = ret.loc[seg.index].fillna(0.0)
        out[label] = {
            "sharpe": round(sharpe_ratio(seg.to_numpy()), 2),
            "bh_sharpe": round(sharpe_ratio(bench.to_numpy()), 2),
            "maxdd": round(max_drawdown(seg.to_numpy()), 3),
        }
    pooled = s.dropna()
    nw = newey_west_alpha(pooled, ret.loc[pooled.index].fillna(0.0))
    out["alpha_t_pooled"] = round(nw["alpha_t"], 2)
    out["alpha_ann_pooled"] = round(nw["alpha_ann"], 4)
    return out


def main() -> int:
    frames = {sym: load(sym) for sym in VEHICLES}
    stats = family_stats(frames[PRIMARY])
    schemes = weight_schemes(stats)

    results: dict[str, dict] = {sym: {} for sym in VEHICLES}
    for sym in VEHICLES:
        for scheme, weights in schemes.items():
            results[sym][scheme] = {"weights": weights, **score_scheme(frames[sym], weights)}

    # Selection on the primary vehicle, dev+val only: beat equal in BOTH.
    equal = results[PRIMARY]["equal"]
    selected = "equal"
    for scheme in ("alpha_t_weighted", "sharpe_weighted", "inverse_vol", "drop_worst"):
        r = results[PRIMARY][scheme]
        beats_equal_both = (
            r["dev"]["sharpe"] > equal["dev"]["sharpe"]
            and r["val"]["sharpe"] > equal["val"]["sharpe"]
        )
        if beats_equal_both and (
            selected == "equal"
            or r["dev"]["sharpe"] + r["val"]["sharpe"]
            > results[PRIMARY][selected]["dev"]["sharpe"]
            + results[PRIMARY][selected]["val"]["sharpe"]
        ):
            selected = scheme

    report = {
        "campaign": "ensemble_weights",
        "fill": FILL,
        "primary_vehicle": PRIMARY,
        "family_stats_devval": stats,
        "schemes": results,
        "selected_on_devval_only": selected,
        "selection_rule": "challenger must beat equal-weight Sharpe in BOTH dev and val on QQQ",
        "disclaimer": (
            "Legacy-status research; holdout previously accessed; selection used dev+val only; "
            "not investment advice."
        ),
    }
    atomic_write_bytes(OUT, json.dumps(report, indent=2).encode("utf-8"))

    print(f"family dev+val evidence on {PRIMARY} ({FILL}, 2 bps):")
    for name, st in stats.items():
        print(f"  {name:<14} alpha_t={st['alpha_t_devval']:+.2f} sharpe={st['sharpe_devval']:+.2f}")
    print("\nscheme results on QQQ (dev / val / holdout*, *previously accessed):")
    for scheme, r in results[PRIMARY].items():
        tag = " <== SELECTED" if scheme == selected else ""
        print(
            f"  {scheme:<18} {r['dev']['sharpe']:+.2f} / {r['val']['sharpe']:+.2f} / "
            f"{r['holdout_prev_accessed']['sharpe']:+.2f}  "
            f"(B&H {r['dev']['bh_sharpe']:+.2f}/{r['val']['bh_sharpe']:+.2f}/"
            f"{r['holdout_prev_accessed']['bh_sharpe']:+.2f})  "
            f"pooled t={r['alpha_t_pooled']:.2f}{tag}"
        )
    print(f"\nweights: {json.dumps(schemes[selected])}")
    print(f"full table -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
