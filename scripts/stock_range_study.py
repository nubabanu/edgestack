"""Psychology rangers: stocks in multi-year NOMINAL price ranges - and do ranges hold?

Question (2026-07-23, user request): human psychology (round-number anchors,
remembered prices) should make SOME stocks oscillate between support and
resistance. Find the stocks currently doing that, and answer the question
that decides tradability: once a stock has ranged for three years, what
happens NEXT?

Honest caveats, stated up front:
- Anchoring psychology operates on NOMINAL, visible chart prices, so bands
  here use the split-adjusted close (not CPI-deflated, not dividend-
  adjusted); forward RETURNS use adj_close (total return).
- "Currently ranging" is trivially findable - at any moment some stocks are
  mid-range by chance. The tradable claim is range PERSISTENCE, measured
  here out-of-sample: for every 3-year ranged setup in 25 years of history,
  the next-12-month outcome (range holds / breaks up / breaks down) and the
  forward total return of buying near the floor vs buying any day.
- Setups are sampled 252 sessions apart per symbol to avoid overlap
  double-counting; results pool across symbols (cross-sectional dependence
  remains - same-date setups share market shocks; treat t-stats as upper
  bounds).
- The academic prior cuts AGAINST the bounce story: George & Hwang (2004)
  found 52-week-high proximity predicts CONTINUATION (momentum), not
  reversal. Previously-accessed data; not investment advice.

Definitions: window = 756 sessions (~3y); band = [p10, p90] of window closes;
ranged = band ratio >= 1.25 AND >= 3 hysteresis traversals AND close inside
band. Outcome window = next 252 sessions; hold = never beyond +/-15% of the
band; first breach beyond sets break direction.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from level_range_scan import count_traversals  # noqa: E402  (same hysteresis logic)

PRICES = ROOT / "data" / "curated" / "prices"
REPORT_PATH = ROOT / "artifacts" / "stock_range_study.json"

WINDOW = 756
FWD = 252
MIN_BAND_RATIO = 1.25
MIN_TRIPS = 3
BREACH_TOL = 0.15
NEAR_FLOOR = 0.25
NEAR_CEIL = 0.75
MIN_DVOL_M = 5.0
MAX_ABS_RETURN = 2.0
ETFS = frozenset(
    [
        "SPY",
        "TLT",
        "SHY",
        "GLD",
        "SLV",
        "USO",
        "BNO",
        "UNG",
        "UGA",
        "QQQ",
        "IWM",
        "DIA",
        "XLK",
        "SMH",
        "XLF",
        "XLE",
        "XLY",
        "XLP",
        "XLV",
        "EWJ",
        "EWC",
        "UUP",
        "DBC",
        "DBA",
    ]
)


def stock_symbols() -> list[str]:
    return sorted(
        p.stem
        for p in PRICES.glob("*.parquet")
        if "=" not in p.stem and not p.stem.startswith("^") and p.stem not in ETFS
    )


def load(symbol: str) -> pd.DataFrame | None:
    df = pd.read_parquet(PRICES / f"{symbol}.parquet")
    df = df.assign(date=pd.to_datetime(df["date"])).set_index("date").sort_index()
    if len(df) < WINDOW + 10:
        return None
    if np.log(df["adj_close"].clip(lower=1e-9)).diff().abs().max() > MAX_ABS_RETURN:
        return None
    tail = df.tail(504)
    volume = tail["volume"] if "volume" in tail.columns else pd.Series(0.0, index=tail.index)
    if float((tail["close"] * volume.fillna(0.0)).median()) < MIN_DVOL_M * 1e6:
        return None
    return df


def band_state(closes: np.ndarray) -> dict | None:
    """Ranged-or-not for one window of closes; band + position if ranged."""
    floor = float(np.quantile(closes, 0.10))
    ceiling = float(np.quantile(closes, 0.90))
    if floor <= 0 or ceiling / floor < MIN_BAND_RATIO:
        return None
    trips = count_traversals(closes, floor, ceiling)
    last = float(closes[-1])
    if trips < MIN_TRIPS or not floor <= last <= ceiling:
        return None
    return {
        "floor": floor,
        "ceiling": ceiling,
        "trips": trips,
        "position": (last - floor) / (ceiling - floor),
    }


def outcome(closes_fwd: np.ndarray, floor: float, ceiling: float) -> str:
    lo, hi = floor * (1 - BREACH_TOL), ceiling * (1 + BREACH_TOL)
    below = closes_fwd < lo
    above = closes_fwd > hi
    if not below.any() and not above.any():
        return "hold"
    first_below = int(np.argmax(below)) if below.any() else 10**9
    first_above = int(np.argmax(above)) if above.any() else 10**9
    return "break_down" if first_below < first_above else "break_up"


def main() -> int:
    current: dict[str, dict] = {}
    outcomes = {"hold": 0, "break_up": 0, "break_down": 0}
    floor_fwd: list[float] = []
    ceil_fwd: list[float] = []
    uncond_fwd: list[float] = []
    setups = 0

    for symbol in stock_symbols():
        try:
            df = load(symbol)
        except Exception:
            continue
        if df is None:
            continue
        closes = df["close"].to_numpy(dtype=float)
        adj = df["adj_close"].to_numpy(dtype=float)
        n = len(closes)

        state = band_state(closes[-WINDOW:])
        if state:
            current[symbol] = {
                "floor": round(state["floor"], 2),
                "ceiling": round(state["ceiling"], 2),
                "trips_3y": state["trips"],
                "position_now": round(state["position"], 2),
                "last": round(float(closes[-1]), 2),
            }

        for i in range(WINDOW, n - FWD, FWD):  # non-overlapping annual sampling
            fwd_ret = adj[i + FWD] / adj[i] - 1
            uncond_fwd.append(fwd_ret)
            st = band_state(closes[i - WINDOW : i])
            if st is None:
                continue
            setups += 1
            outcomes[outcome(closes[i : i + FWD], st["floor"], st["ceiling"])] += 1
            if st["position"] <= NEAR_FLOOR:
                floor_fwd.append(fwd_ret)
            elif st["position"] >= NEAR_CEIL:
                ceil_fwd.append(fwd_ret)

    def stats(values: list[float]) -> dict:
        arr = np.array(values)
        if len(arr) < 3:
            return {"n": len(arr)}
        t = float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr))))
        return {"n": len(arr), "mean": round(float(arr.mean()), 4), "t": round(t, 2)}

    total = max(sum(outcomes.values()), 1)
    report = {
        "batch_id": uuid.uuid4().hex[:12],
        "definitions": {
            "window": WINDOW,
            "fwd": FWD,
            "min_trips": MIN_TRIPS,
            "band": "nominal close p10..p90",
        },
        "historical_setups": setups,
        "outcome_rates": {k: round(v / total, 3) for k, v in outcomes.items()},
        "outcome_counts": outcomes,
        "fwd_12m_near_floor": stats(floor_fwd),
        "fwd_12m_near_ceiling": stats(ceil_fwd),
        "fwd_12m_unconditional": stats(uncond_fwd),
        "current_rangers": dict(sorted(current.items(), key=lambda kv: -kv[1]["trips_3y"])),
        "disclaimer": (
            "Nominal-band psychology screen; cross-sectional dependence inflates "
            "t-stats; previously-accessed data; not investment advice."
        ),
    }
    from edgestack.data.catalog import atomic_write_bytes

    atomic_write_bytes(REPORT_PATH, json.dumps(report, indent=2).encode("utf-8"))

    print(f"historical 3y-ranged setups (non-overlapping): {setups}")
    print(f"next-12m outcomes: {report['outcome_rates']}")
    print(f"buy NEAR FLOOR of a ranged stock, fwd 12m: {report['fwd_12m_near_floor']}")
    print(f"buy NEAR CEILING of a ranged stock, fwd 12m: {report['fwd_12m_near_ceiling']}")
    print(f"buy ANY sampled day, fwd 12m:               {report['fwd_12m_unconditional']}")
    print(f"\ncurrently ranging stocks ({len(current)}), by 3y traversal count:")
    print(f"{'symbol':10s} {'floor':>9s} {'ceiling':>9s} {'trips3y':>8s} {'now@':>5s} {'last':>9s}")
    for sym, row in list(report["current_rangers"].items())[:25]:
        print(
            f"{sym:10s} {row['floor']:>9} {row['ceiling']:>9} "
            f"{row['trips_3y']:>8} {row['position_now']:>5} {row['last']:>9}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
