"""Level-range scan: which assets keep bouncing between the same REAL price levels?

Question (2026-07-23, user request): find assets that are periodic by PRICE
LEVEL, not by time - e.g. "falls to ~$50, rallies to ~$100, repeatedly" -
with inflation included, so the floor/ceiling are in today's dollars.

Why this is better-posed than the sine-wave question: commodities have an
economic anchor for level cycles - a production-cost FLOOR (below it, supply
shuts) and a demand-destruction CEILING (above it, consumption and
substitution bite) - and both anchors live in REAL dollars. Stocks have no
such anchors; they drift with retained earnings and never owe a return to
any level.

Honest caveats, stated up front:
- Range-boundness also happens by luck. Every metric is compared to 300
  zero-drift random walks with matched volatility run through the identical
  procedure; a candidate matters only above the null's 95th percentile.
- The band is fitted on the FIRST HALF of history only (10th/90th real-price
  percentiles) and all headline numbers are OUT-OF-SAMPLE: traversals and
  containment measured on the second half against the frozen band.
- Deflation uses US CPI (CPIAUCSL) for everything, including non-USD
  listings - a simplification; local CPIs differ. ^-prefixed indices (VIX,
  OVX) are unit-free and are NOT deflated.
- Futures are the continuous front-month series: roll yield means the
  TRADEABLE return of holding through a range traversal is worse than the
  spot path suggests (storage/carry eats part of every round trip).
- A stable historical floor is not a law: cost curves move (shale broke the
  $80 oil floor in 2014), and "always" in 25 years is ~a dozen observations.
- Previously-accessed data; a strategy needs the survivor-bar study before
  any alert or ticket. Not investment advice.

Method: real price = nominal * CPI(latest)/CPI(t). Band: [p10, p90] of the
first half, kept only if p90/p10 >= MIN_BAND_RATIO. Traversals counted with
hysteresis (touch below floor -> touch above ceiling = one traversal, and
back). OOS score = traversals per year in half 2; containment = share of
half-2 days inside [0.85*floor, 1.15*ceiling].
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

PRICES = ROOT / "data" / "curated" / "prices"
CPI_CACHE = ROOT / "data" / "cache" / "cpi_cpiaucsl.csv"
REPORT_PATH = ROOT / "artifacts" / "level_range_scan.json"

MIN_SESSIONS = 2520
MIN_PRICE = 1.0
MAX_ABS_RETURN = 2.0
MIN_BAND_RATIO = 1.3  # ceiling must be >= 30% above floor to be worth trading
BAND_LO_Q = 0.10
BAND_HI_Q = 0.90
CONTAINMENT_TOL = 0.15  # band breach tolerance for the containment share
MIN_CONTAINMENT = 0.80
NULL_DRAWS = 300
TOP_N = 20


def load_cpi() -> pd.Series:
    df = pd.read_csv(CPI_CACHE)
    s = pd.Series(
        df["CPIAUCSL"].to_numpy(), index=pd.to_datetime(df["observation_date"])
    ).sort_index()
    return s


def real_prices(symbol: str, cpi: pd.Series) -> tuple[pd.Series, str | None]:
    df = pd.read_parquet(PRICES / f"{symbol}.parquet")
    df = df.assign(date=pd.to_datetime(df["date"])).set_index("date").sort_index()
    px = df["adj_close"].astype(float)
    px = px[px > 0]
    if len(px) < MIN_SESSIONS:
        return pd.Series(dtype=float), "too short"
    if np.log(px).diff().abs().max() > MAX_ABS_RETURN:
        return pd.Series(dtype=float), "corrupted (|ret|>200%)"
    if float(df["close"].iloc[-1]) < MIN_PRICE:
        return pd.Series(dtype=float), "price < $1"
    if symbol.startswith("^"):
        return px, None  # unit-free index: no deflation
    factor = float(cpi.iloc[-1]) / cpi.reindex(px.index, method="ffill")
    factor = factor.ffill().bfill()
    return px * factor, None


def count_traversals(prices: np.ndarray, floor: float, ceiling: float) -> int:
    """Hysteresis traversal count: floor-touch -> ceiling-touch (or reverse) = 1."""
    state = 0  # 0 unknown, -1 last extreme was floor, +1 last extreme was ceiling
    traversals = 0
    for p in prices:
        if p <= floor:
            if state == 1:
                traversals += 1
            state = -1
        elif p >= ceiling:
            if state == -1:
                traversals += 1
            state = 1
    return traversals


def analyze(px: pd.Series) -> dict | None:
    n = len(px)
    half = n // 2
    first, second = px.iloc[:half], px.iloc[half:]
    floor = float(first.quantile(BAND_LO_Q))
    ceiling = float(first.quantile(BAND_HI_Q))
    if floor <= 0 or ceiling / floor < MIN_BAND_RATIO:
        return None
    years1 = max((first.index[-1] - first.index[0]).days / 365.25, 1e-9)
    years2 = max((second.index[-1] - second.index[0]).days / 365.25, 1e-9)
    trips1 = count_traversals(first.to_numpy(), floor, ceiling)
    trips2 = count_traversals(second.to_numpy(), floor, ceiling)
    inside = (
        (second >= floor * (1 - CONTAINMENT_TOL)) & (second <= ceiling * (1 + CONTAINMENT_TOL))
    ).mean()
    last = float(px.iloc[-1])
    position = (last - floor) / (ceiling - floor)
    return {
        "sessions": n,
        "floor_real": round(floor, 2),
        "ceiling_real": round(ceiling, 2),
        "band_ratio": round(ceiling / floor, 2),
        "trips_insample": trips1,
        "trips_oos": trips2,
        "trips_oos_per_year": round(trips2 / years2, 2),
        "trips_is_per_year": round(trips1 / years1, 2),
        "oos_containment": round(float(inside), 3),
        "last_real_price": round(last, 2),
        "band_position_now": round(position, 2),
    }


def null_distribution(vols: list[float], lengths: list[int], rng: np.random.Generator) -> dict:
    """Zero-drift random walks, matched vol/length, identical banding procedure."""
    scores = []
    for _ in range(NULL_DRAWS):
        i = int(rng.integers(len(vols)))
        n, vol = lengths[i], vols[i]
        px = pd.Series(
            100.0 * np.exp(np.cumsum(rng.standard_normal(n) * vol)),
            index=pd.bdate_range("2000-01-03", periods=n),
        )
        row = analyze(px)
        if row is None or row["oos_containment"] < MIN_CONTAINMENT:
            scores.append(0.0)
        else:
            scores.append(row["trips_oos_per_year"])
    return {
        "draws": NULL_DRAWS,
        "oos_trips_rate_p50": round(float(np.percentile(scores, 50)), 3),
        "oos_trips_rate_p95": round(float(np.percentile(scores, 95)), 3),
        "contained_share": round(float(np.mean([s > 0 for s in scores])), 3),
    }


def main() -> int:
    cpi = load_cpi()
    rng = np.random.default_rng(20260723)
    results: dict[str, dict] = {}
    dropped = 0
    vols: list[float] = []
    lengths: list[int] = []
    for path in sorted(PRICES.glob("*.parquet")):
        symbol = path.stem
        try:
            px, reason = real_prices(symbol, cpi)
        except Exception:
            dropped += 1
            continue
        if reason:
            dropped += 1
            continue
        row = analyze(px)
        vols.append(float(np.log(px).diff().std()))
        lengths.append(len(px))
        if row is None:
            continue
        results[symbol] = row

    null = null_distribution(vols, lengths, rng)
    survivors = {
        sym: row
        for sym, row in results.items()
        if row["oos_containment"] >= MIN_CONTAINMENT
        and row["trips_oos_per_year"] > null["oos_trips_rate_p95"]
        and row["trips_oos"] >= 2
    }
    ranked = sorted(
        (kv for kv in results.items() if kv[1]["oos_containment"] >= MIN_CONTAINMENT),
        key=lambda kv: -kv[1]["trips_oos_per_year"],
    )

    report = {
        "batch_id": uuid.uuid4().hex[:12],
        "universe_banded": len(results),
        "dropped_or_unbanded": dropped,
        "null": null,
        "survivors": {sym: results[sym] for sym in sorted(survivors)},
        "top_contained_by_oos_trips": dict(ranked[:TOP_N]),
        "disclaimer": (
            "Real (CPI-deflated) level ranges; band fitted on half 1, judged on "
            "half 2; roll yield not modeled for futures; cost floors move; "
            "previously-accessed data; not investment advice."
        ),
    }
    from edgestack.data.catalog import atomic_write_bytes

    atomic_write_bytes(REPORT_PATH, json.dumps(report, indent=2).encode("utf-8"))

    print(f"banded {len(results)} symbols ({dropped} dropped/unbandable)")
    print(
        f"NULL: median OOS trips/yr {null['oos_trips_rate_p50']}, "
        f"95th pct {null['oos_trips_rate_p95']}, "
        f"contained-by-chance share {null['contained_share']}"
    )
    header = (
        f"{'symbol':12s} {'floor$':>8s} {'ceil$':>8s} {'ratio':>6s} "
        f"{'tripsIS':>8s} {'tripsOOS':>9s} {'oos/yr':>7s} {'contain':>8s} {'now@':>5s}"
    )
    print("\nsurvivors (contained OOS, trips/yr above null 95th, >=2 OOS trips):")
    print(header)
    for sym in sorted(survivors, key=lambda s: -results[s]["trips_oos_per_year"]):
        row = results[sym]
        print(
            f"{sym:12s} {row['floor_real']:>8} {row['ceiling_real']:>8} "
            f"{row['band_ratio']:>6} {row['trips_insample']:>8} {row['trips_oos']:>9} "
            f"{row['trips_oos_per_year']:>7} {row['oos_containment']:>8} "
            f"{row['band_position_now']:>5}"
        )
    print(f"\ntop {TOP_N} contained ranges by OOS traversal rate:")
    print(header)
    for sym, row in ranked[:TOP_N]:
        print(
            f"{sym:12s} {row['floor_real']:>8} {row['ceiling_real']:>8} "
            f"{row['band_ratio']:>6} {row['trips_insample']:>8} {row['trips_oos']:>9} "
            f"{row['trips_oos_per_year']:>7} {row['oos_containment']:>8} "
            f"{row['band_position_now']:>5}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
