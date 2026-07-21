"""Breadth-based market timing suite: the first genuinely new signal family
testable with data already in the catalog (2026-07-21 campaign).

Question: do S&P breadth internals (participation, washouts, thrusts,
McClellan, net new highs) time SPY/QQQ exposure better than buy-and-hold,
under the standard zoo bar?

Breadth inputs are computed over POINT-IN-TIME S&P membership
(edgestack.data.universe_pit, change log reliable from ~mid-2000s), NOT the
survivorship-biased full catalog. Members whose price history is missing on
free feeds simply drop out of the denominator; per-split coverage is
reported so the reader can judge the early years.

Rules are PRE-DECLARED (this file is the registration): 11 rules x 2
vehicles = 22 trials, every one counted for the multiplicity note. No
parameter search was run; thresholds are the textbook defaults (Zweig
0.40/0.615, participation 50%, washout 15%/30%).

Conventions: signal at close t earns session t+1; 2 bps per unit exposure
change; long/flat only. Splits: dev 2011-2015 (the curated
panel's design horizon starts 2011), val 2016-2023, holdout 2024+
(previously accessed; never promotion evidence). Survivor bar identical to
docs/strategy-zoo.md: Sharpe >= buy-and-hold in ALL splits AND pooled
Newey-West alpha t >= 2.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from edgestack.data.catalog import atomic_write_bytes
from edgestack.data.universe_pit import PitSP500Universe
from edgestack.validation.advanced_tests import newey_west_alpha
from edgestack.validation.metrics import max_drawdown, sharpe_ratio

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "data" / "curated" / "prices"
OUT = ROOT / "artifacts" / "breadth_timing.json"

COST = 0.0002
VEHICLES = ("SPY", "QQQ")
SPLITS = {
    "dev_2011_2015": ("2011-01-01", "2015-12-31"),
    "val_2016_2023": ("2016-01-01", "2023-12-31"),
    "holdout_2024_prev_accessed": ("2024-01-01", "2026-12-31"),
}
PIT_EARLIEST = date(2000, 1, 3)
MIN_MEMBERS_WITH_PRICE = 100  # below this the breadth reading is untrustworthy


def load_close_panel() -> pd.DataFrame:
    """Wide close panel over every catalog symbol passing the corrupted screen."""
    frames = {}
    for f in sorted(PRICES.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["date", "close", "adj_close"])
        if df["adj_close"].pct_change().abs().max() > 2.0:
            continue  # corrupted series screen (CBE/TIE-style shells)
        s = df.assign(date=pd.to_datetime(df["date"])).set_index("date")["close"].sort_index()
        frames[f.stem] = s[~s.index.duplicated()]
    return pd.DataFrame(frames)


def membership_matrix(universe: PitSP500Universe, panel: pd.DataFrame) -> pd.DataFrame:
    """Wide bool frame: was symbol a PIT S&P member on each panel date."""
    mask = pd.DataFrame(False, index=panel.index, columns=panel.columns)
    dates = panel.index.date
    spans: dict[str, list[tuple[date, date]]] = {}
    for interval in universe.intervals:
        spans.setdefault(interval.symbol, []).append((interval.start, interval.end))
    for symbol, symbol_spans in spans.items():
        if symbol not in mask.columns:
            continue
        col = np.zeros(len(mask), dtype=bool)
        for start, end in symbol_spans:
            col |= (dates >= start) & (dates < end)
        mask[symbol] = col
    return mask


def breadth_series(panel: pd.DataFrame, member: pd.DataFrame) -> pd.DataFrame:
    """Daily breadth internals over PIT members with available prices."""
    have = panel.notna() & member
    n = have.sum(axis=1)

    out = pd.DataFrame(index=panel.index)
    out["n_members_with_price"] = n

    for window in (20, 50, 200):
        above = (panel > panel.rolling(window).mean()) & have
        out[f"pct_above_{window}"] = above.sum(axis=1) / n.replace(0, np.nan)

    ret = panel.pct_change()
    adv = ((ret > 0) & have).sum(axis=1) / n.replace(0, np.nan)
    dec = ((ret < 0) & have).sum(axis=1) / n.replace(0, np.nan)
    out["adv_frac"] = adv
    out["mcclellan"] = (adv - dec).ewm(span=19, adjust=False).mean() - (adv - dec).ewm(
        span=39, adjust=False
    ).mean()

    hi252 = (panel >= panel.rolling(252).max()) & have
    lo252 = (panel <= panel.rolling(252).min()) & have
    out["nhnl_frac"] = (hi252.sum(axis=1) - lo252.sum(axis=1)) / n.replace(0, np.nan)

    # Untrustworthy readings (too few priced members) become NaN -> flat.
    thin = n < MIN_MEMBERS_WITH_PRICE
    out.loc[thin, out.columns.difference(["n_members_with_price"])] = np.nan
    return out


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def hold_after(trigger: pd.Series, sessions: int) -> pd.Series:
    """Position = 1 for `sessions` sessions after each trigger firing."""
    return trigger.astype(float).rolling(sessions, min_periods=1).max().fillna(0.0)


def build_rules(b: pd.DataFrame, vehicle_close: pd.Series) -> dict[str, pd.Series]:
    p20, p50, p200 = b["pct_above_20"], b["pct_above_50"], b["pct_above_200"]
    osc, nhnl = b["mcclellan"], b["nhnl_frac"]
    ema10 = b["adv_frac"].ewm(span=10, adjust=False).mean()

    zweig = (ema10 > 0.615) & (
        (ema10 < 0.40).shift(1).rolling(10, min_periods=1).max().astype(bool)
    )
    washout = p20 < 0.15
    washout_recross = (p20 > 0.30) & (
        washout.shift(1).rolling(10, min_periods=1).max().astype(bool)
    )
    r2 = _rsi(vehicle_close, 2)

    rules = {
        "bt_washout20_hold21": hold_after(washout, 21),
        "bt_washout20_recross_hold63": hold_after(washout_recross, 63),
        "bt_participation_200": (p200 > 0.50).astype(float),
        "bt_participation_hysteresis": None,  # built below (stateful)
        "bt_zweig_thrust_hold126": hold_after(zweig, 126),
        "bt_mcclellan_positive": (osc > 0).astype(float),
        "bt_mcclellan_washout_hold21": hold_after(osc < -0.05, 21),
        "bt_nhnl_positive": (nhnl > 0).astype(float),
        "bt_price_and_breadth": (
            (vehicle_close > vehicle_close.rolling(200).mean()) & (p200 > 0.50)
        ).astype(float),
        "bt_dip_in_breadth_uptrend": hold_after((r2 < 10) & (p200 > 0.50), 5),
        "bt_pct50_recross_up": hold_after((p50 > 0.50) & (p50.shift() <= 0.50), 63),
    }

    # Hysteresis gate: long when participation crosses above 55%, flat below 45%.
    state, states = 0.0, np.zeros(len(p200))
    values = p200.to_numpy()
    for i, v in enumerate(values):
        if np.isnan(v):
            state = 0.0
        elif v > 0.55:
            state = 1.0
        elif v < 0.45:
            state = 0.0
        states[i] = state
    rules["bt_participation_hysteresis"] = pd.Series(states, index=p200.index)

    # NaN breadth (thin coverage / warmup) means flat, never a phantom signal.
    return {k: v.fillna(0.0) for k, v in rules.items()}


def run_rule(pos: pd.Series, ret: pd.Series) -> pd.Series:
    p = pos.shift(1).fillna(0.0)
    return p * ret - COST * p.diff().abs().fillna(0.0)


def main() -> int:
    t0 = time.time()
    panel = load_close_panel()
    universe = PitSP500Universe(ROOT / "data" / "cache" / "universe", earliest=PIT_EARLIEST)
    member = membership_matrix(universe, panel)
    breadth = breadth_series(panel, member)
    print(
        f"panel {panel.shape[1]} symbols x {panel.shape[0]} sessions; "
        f"breadth built in {time.time() - t0:.0f}s"
    )

    coverage = {
        split: {
            "mean_members_with_price": round(
                float(breadth.loc[lo:hi, "n_members_with_price"].mean()), 1
            )
        }
        for split, (lo, hi) in SPLITS.items()
    }

    results, trials = [], 0
    for sym in VEHICLES:
        df = pd.read_parquet(PRICES / f"{sym}.parquet")
        df = df.assign(date=pd.to_datetime(df["date"])).set_index("date").sort_index()
        close = df["close"].reindex(breadth.index)
        ret = df["adj_close"].reindex(breadth.index).pct_change()
        for name, pos in build_rules(breadth, close).items():
            trials += 1
            strat = run_rule(pos, ret)
            row: dict = {"rule": name, "vehicle": sym, "avg_exposure": round(float(pos.mean()), 2)}
            ok_all = True
            for split, (lo, hi) in SPLITS.items():
                s = strat.loc[lo:hi].dropna()
                bench = ret.loc[s.index].fillna(0.0)
                if len(s) < 200:
                    ok_all = False
                    continue
                sh_s, sh_b = sharpe_ratio(s.to_numpy()), sharpe_ratio(bench.to_numpy())
                row[split] = {
                    "sharpe": round(sh_s, 2),
                    "bh_sharpe": round(sh_b, 2),
                    "maxdd": round(max_drawdown(s.to_numpy()), 3),
                }
                if sh_s < sh_b:
                    ok_all = False
            pooled = strat.dropna()
            nw = newey_west_alpha(pooled, ret.loc[pooled.index].fillna(0.0))
            row["alpha_t"] = round(nw["alpha_t"], 2)
            row["alpha_ann"] = round(nw["alpha_ann"], 4)
            row["SURVIVOR"] = bool(ok_all and nw["alpha_t"] >= 2.0)
            results.append(row)

    survivors = [r for r in results if r["SURVIVOR"]]
    near = [
        r
        for r in results
        if not r["SURVIVOR"]
        and r["alpha_t"] >= 1.5
        and sum(
            1
            for split in SPLITS
            if isinstance(r.get(split), dict) and r[split]["sharpe"] >= r[split]["bh_sharpe"]
        )
        >= 2
    ]
    report = {
        "campaign": "breadth_timing",
        "as_of": str(breadth.index[-1].date()),
        "trials": trials,
        "expected_false_positives_at_t2": round(trials * 0.025, 1),
        "pit_coverage": coverage,
        "survivors": survivors,
        "near_survivors": near,
        "results": results,
        "caveats": [
            "PIT membership change log reliable from ~mid-2000s; earlier breadth "
            "leans on fewer, survivor-tilted priced members (see pit_coverage)",
            "holdout 2024+ is previously accessed and can never promote a sleeve",
            "thresholds are textbook defaults, pre-declared; no search was run",
        ],
        "disclaimer": "Historical research on previously-accessed data; not investment advice.",
    }
    atomic_write_bytes(OUT, json.dumps(report, indent=2).encode("utf-8"))

    print(f"\ntrials: {trials} (expect ~{trials * 0.025:.1f} false |t|>2 by chance)")
    print(f"coverage: {json.dumps(coverage)}")
    print(f"SURVIVORS: {len(survivors)}")
    for r in survivors + near:
        tag = "SURVIVOR" if r["SURVIVOR"] else "near"
        print(
            f"  [{tag}] {r['rule']:<30}{r['vehicle']:<5} alpha {r['alpha_ann']:+.1%}/yr "
            f"t={r['alpha_t']:.2f} exp={r['avg_exposure']}"
        )
    print(f"\nfull table -> {OUT} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
