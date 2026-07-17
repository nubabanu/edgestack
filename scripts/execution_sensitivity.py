"""Historical execution sensitivity for legacy-status strategies; non-promotable in V2.

Our zoo convention ("decide at close t-1, earn session t") implicitly fills
AT the very close that produced the signal — impossible live, since MOC
orders must be submitted ~10 min before the close. This script quantifies
what realistic execution does to ensemble4:

    same-close   fill at the signal close        (optimistic bound; zoo default)
    next-open    fill at the next session's open (fully realistic, MOO)
    next-close   MOC submitted next day          (one-session delay; worst case)

plus cost levels 0/2/5/10 bps per unit exposure change. If the edge dies
under next-open fills, it was never real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from edgestack.data.catalog import atomic_write_bytes
from edgestack.strategies import ensemble_exposure
from edgestack.validation.advanced_tests import newey_west_alpha
from edgestack.validation.metrics import sharpe_ratio

CACHE = Path("data/cache/zoo")
SPLITS = {
    "dev": ("1999-01-01", "2015-12-31"),
    "val": ("2016-01-01", "2023-12-31"),
    "hold": ("2024-01-01", "2026-12-31"),
}


def load(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(CACHE / f"{sym}.parquet")
    return (
        df.assign(date=df["dt"].dt.tz_localize(None).dt.normalize())
        .set_index("date")
        .drop(columns="dt")
    )


def strat_returns(df: pd.DataFrame, pos: pd.Series, fill: str, cost: float) -> pd.Series:
    """Daily strategy returns under an execution model.

    same-close: held_t = pos_{t-1}, earns close_{t-1}->close_t.
    next-close: held_t = pos_{t-2} (MOC filled one session later).
    next-open : exposure changes execute at open_t; day t return =
                held_old * (open_t/close_{t-1}-1) + held_new * (close_t/open_t-1).
    """
    ret = df["adj"].pct_change()
    if fill == "same-close":
        held = pos.shift(1).fillna(0.0)
        return held * ret - cost * held.diff().abs().fillna(0.0)
    if fill == "next-close":
        held = pos.shift(2).fillna(0.0)
        return held * ret - cost * held.diff().abs().fillna(0.0)
    if fill == "next-open":
        # split the day at the open (raw prices approximate the adj ratio)
        overnight = (df["open"] / df["close"].shift() - 1).fillna(0.0)
        intraday = (df["close"] / df["open"] - 1).fillna(0.0)
        held_old = pos.shift(2).fillna(0.0)  # position from two closes ago
        held_new = pos.shift(1).fillna(0.0)  # updated at today's open
        gross = held_old * overnight + held_new * intraday
        return gross - cost * (held_new - held_old).abs().fillna(0.0)
    raise ValueError(fill)


def main() -> int:
    out = {}
    print(
        f"{'sym':<5}{'fill model':<12}{'cost':>6}{'dev Sh':>8}{'val Sh':>8}"
        f"{'hold Sh':>8}{'a/yr':>7}{'t':>6}"
    )
    for sym in ("SPY", "QQQ"):
        df = load(sym)
        ret = df["adj"].pct_change()
        pos = ensemble_exposure(df)
        bh = {k: sharpe_ratio(ret.loc[lo:hi].dropna().to_numpy()) for k, (lo, hi) in SPLITS.items()}
        print(
            f"{sym:<5}{'buy&hold':<12}{'':>6}{bh['dev']:>8.2f}{bh['val']:>8.2f}{bh['hold']:>8.2f}"
        )
        for fill in ("same-close", "next-open", "next-close"):
            for cost in (0.0, 0.0002, 0.0005, 0.001):
                s = strat_returns(df, pos, fill, cost)
                row = {}
                for k, (lo, hi) in SPLITS.items():
                    seg = s.loc[lo:hi].dropna()
                    row[k] = round(sharpe_ratio(seg.to_numpy()), 2)
                nw = newey_west_alpha(s.dropna(), ret.loc[s.dropna().index].fillna(0.0))
                row["alpha_ann"] = round(nw["alpha_ann"], 4)
                row["alpha_t"] = round(nw["alpha_t"], 2)
                out[f"{sym}/{fill}/{int(cost * 1e4)}bps"] = row
                if cost in (0.0002, 0.001):
                    print(
                        f"{sym:<5}{fill:<12}{int(cost * 1e4):>4}bp"
                        f"{row['dev']:>8.2f}{row['val']:>8.2f}{row['hold']:>8.2f}"
                        f"{row['alpha_ann']:>7.1%}{row['alpha_t']:>6.2f}"
                    )
    atomic_write_bytes(
        Path("artifacts") / "execution_sensitivity.json", json.dumps(out, indent=1).encode()
    )
    print("\nsaved -> artifacts/execution_sensitivity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
