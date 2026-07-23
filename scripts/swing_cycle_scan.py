"""Swing-cycle scan: assets that swing up-down-up-down between FUZZY price zones.

Question (2026-07-23, user request): find assets that repeatedly dip to
"around" one level and rally to "around" another (e.g. ~50 -> ~120 -> ~50),
at weekly, monthly, or multi-month scale, without long-term decay. Unlike
level_range_scan's fixed quantile bands, zones here are fuzzy and may drift
with the trend.

Honest caveats, stated up front:
- Volatile random walks also swing constantly; what they do NOT do is keep
  bottoming in the same zone. Every candidate is scored against 300
  zero-drift matched-volatility random walks run through the identical
  zigzag pipeline; the discriminator is trough-zone TIGHTNESS, and only
  scores above the null's 95th percentile count.
- The stock_range_study verdict stands: buying the "usual dip zone" of a
  ranging stock historically earned EXACTLY the unconditional baseline
  (+14.8% vs +14.8% fwd 12m). Survivors here are watch-context, not signals;
  no promotion without a survivor-bar study.
- Zones are NOMINAL (psychology anchors on visible prices); the drift filter
  (>= -2%/yr log) excludes decayers per the user's "eventually up or flat".
- Futures are continuous front-month series (roll effects inside swings).
- Previously-accessed data; not investment advice.

Method: log-scale zigzag pivots at reversal thresholds 15% / 30% / 50%;
metrics = swings/year, detrended log-pivot zone std (troughs and peaks),
swing-magnitude consistency, drift, split-half swing counts. Score =
swings_per_year * exp(-2 * trough_zone_std). Survivors feed
artifacts/swing_zones.json for the nightly display-only zone watcher.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PRICES = ROOT / "data" / "curated" / "prices"
REPORT_PATH = ROOT / "artifacts" / "swing_cycle_scan.json"
ZONES_PATH = ROOT / "artifacts" / "swing_zones.json"

THETAS = (0.15, 0.30, 0.50)
MIN_SESSIONS = 2016  # ~8 years
MIN_PRICE = 1.0
MAX_ABS_RETURN = 2.0
MIN_DVOL_M = 5.0  # stocks only; futures/indices exempt
MIN_DRIFT_PER_YEAR = -0.02  # log; "eventually up or stays the same"
MIN_SWINGS_PER_HALF = 2
NULL_DRAWS = 300
TOP_PER_THETA = 15
WATCHLIST_MAX = 10


def zigzag(log_px: np.ndarray, theta: float) -> tuple[list[int], list[int]]:
    """(peak_indices, trough_indices) of confirmed zigzag pivots.

    A pivot is confirmed when price retraces `theta` (in log space) from the
    running extreme of the current leg. Pivots alternate by construction."""
    peaks: list[int] = []
    troughs: list[int] = []
    hi = lo = 0
    direction = 0  # +1 rising leg (next pivot is a peak), -1 falling, 0 undecided
    for i in range(1, len(log_px)):
        if log_px[i] > log_px[hi]:
            hi = i
        if log_px[i] < log_px[lo]:
            lo = i
        if direction >= 0 and log_px[hi] - log_px[i] >= theta:
            peaks.append(hi)
            direction = -1
            hi = lo = i
        elif direction <= 0 and log_px[i] - log_px[lo] >= theta:
            troughs.append(lo)
            direction = 1
            hi = lo = i
    return peaks, troughs


def analyze(px: pd.Series, theta: float) -> dict | None:
    log_px = np.log(px.to_numpy(dtype=float))
    n = len(log_px)
    peaks, troughs = zigzag(log_px, theta)
    swings = len(peaks) + len(troughs)
    if len(peaks) < 2 or len(troughs) < 2:
        return None
    years = max((px.index[-1] - px.index[0]).days / 365.25, 1e-9)
    drift = (log_px[-1] - log_px[0]) / years
    if drift < MIN_DRIFT_PER_YEAR:
        return None
    half = n // 2
    swings_h1 = sum(1 for i in peaks + troughs if i < half)
    swings_h2 = swings - swings_h1
    if min(swings_h1, swings_h2) < MIN_SWINGS_PER_HALF:
        return None

    t = np.array(troughs, dtype=float)
    trough_lp = log_px[troughs]
    peak_lp = log_px[peaks]
    # remove linear drift from the pivot levels so rising zones still count
    trough_resid = trough_lp - np.polynomial.polynomial.polyval(
        t, np.polynomial.polynomial.polyfit(t, trough_lp, 1)
    )
    p = np.array(peaks, dtype=float)
    peak_resid = peak_lp - np.polynomial.polynomial.polyval(
        p, np.polynomial.polynomial.polyfit(p, peak_lp, 1)
    )
    trough_std = float(trough_resid.std(ddof=1))
    peak_std = float(peak_resid.std(ddof=1))

    last_peak_px = float(np.exp(peak_lp[-1]))
    last = float(px.iloc[-1])
    # size of the typical down-swing (log) from adjacent peak->trough pairs
    down_sizes = []
    for pk in peaks:
        nxt = [tr for tr in troughs if tr > pk]
        if nxt:
            down_sizes.append(log_px[pk] - log_px[nxt[0]])
    med_down = float(np.median(down_sizes)) if down_sizes else theta
    current_dd = np.log(last_peak_px / last) if last_peak_px > last else 0.0
    position = float(current_dd / med_down) if med_down > 0 else 0.0

    score = (swings / years) * float(np.exp(-2.0 * trough_std))
    return {
        "swings": swings,
        "swings_per_year": round(swings / years, 2),
        "trough_zone_std": round(trough_std, 3),
        "peak_zone_std": round(peak_std, 3),
        "drift_per_year": round(float(drift), 3),
        "median_trough": round(float(np.exp(np.median(trough_lp))), 2),
        "median_peak": round(float(np.exp(np.median(peak_lp))), 2),
        # zones from the LAST 3 pivots: the current fuzzy zone for trending names
        "trough_zone_recent": round(float(np.exp(np.median(trough_lp[-3:]))), 2),
        "peak_zone_recent": round(float(np.exp(np.median(peak_lp[-3:]))), 2),
        "median_down_swing_pct": round(100 * (1 - np.exp(-med_down)), 1),
        "last": round(last, 2),
        "dip_position_now": round(position, 2),
        "score": round(score, 3),
    }


def load_prices(symbol: str) -> pd.Series | None:
    df = pd.read_parquet(PRICES / f"{symbol}.parquet")
    df = df.assign(date=pd.to_datetime(df["date"])).set_index("date").sort_index()
    px = df["adj_close"].astype(float)
    px = px[px > 0]
    if len(px) < MIN_SESSIONS:
        return None
    if np.log(px).diff().abs().max() > MAX_ABS_RETURN:
        return None
    if float(df["close"].iloc[-1]) < MIN_PRICE:
        return None
    if "=" not in symbol and not symbol.startswith("^"):
        tail = df.tail(504)
        volume = tail["volume"] if "volume" in tail.columns else pd.Series(0.0, index=tail.index)
        if float((tail["close"] * volume.fillna(0.0)).median()) < MIN_DVOL_M * 1e6:
            return None
    return px


def null_scores(
    theta: float, vols: list[float], lengths: list[int], rng: np.random.Generator
) -> float:
    """95th percentile score of zero-drift matched-vol random walks."""
    scores = []
    for _ in range(NULL_DRAWS):
        i = int(rng.integers(len(vols)))
        px = pd.Series(
            100.0 * np.exp(np.cumsum(rng.standard_normal(lengths[i]) * vols[i])),
            index=pd.bdate_range("2000-01-03", periods=lengths[i]),
        )
        row = analyze(px, theta)
        scores.append(row["score"] if row else 0.0)
    return float(np.percentile(scores, 95))


def main() -> int:
    rng = np.random.default_rng(20260723)
    prices: dict[str, pd.Series] = {}
    vols: list[float] = []
    lengths: list[int] = []
    for path in sorted(PRICES.glob("*.parquet")):
        try:
            px = load_prices(path.stem)
        except Exception:
            continue
        if px is None:
            continue
        prices[path.stem] = px
        vols.append(float(np.log(px).diff().std()))
        lengths.append(len(px))

    report: dict = {
        "batch_id": uuid.uuid4().hex[:12],
        "universe": len(prices),
        "thetas": {},
        "disclaimer": (
            "Fuzzy-zone zigzag scan; zones are nominal and drift-adjusted; "
            "stock_range_study found zero floor-bounce edge - survivors are "
            "watch context, not signals; previously-accessed data; not advice."
        ),
    }
    watchlist: list[dict] = []
    for theta in THETAS:
        rows = {}
        for sym, px in prices.items():
            row = analyze(px, theta)
            if row:
                rows[sym] = row
        bar = null_scores(theta, vols, lengths, rng)
        survivors = {s: r for s, r in rows.items() if r["score"] > bar}
        top = sorted(rows.items(), key=lambda kv: -kv[1]["score"])[:TOP_PER_THETA]
        report["thetas"][f"{int(theta * 100)}pct"] = {
            "null_score_p95": round(bar, 3),
            "candidates": len(rows),
            "survivors": len(survivors),
            "survivor_names": sorted(survivors),
            "top": dict(top),
        }
        for sym, row in sorted(survivors.items(), key=lambda kv: -kv[1]["score"]):
            watchlist.append(
                {
                    "symbol": sym,
                    "theta_pct": int(theta * 100),
                    "trough_zone": row["trough_zone_recent"],
                    "peak_zone": row["peak_zone_recent"],
                    "swings_per_year": row["swings_per_year"],
                    "score": row["score"],
                }
            )
        print(
            f"\n=== theta {theta:.0%}: {len(rows)} candidates, null bar {bar:.3f}, "
            f"{len(survivors)} survivors ==="
        )
        header = (
            f"{'symbol':11s} {'sw/yr':>6s} {'zoneStd':>8s} {'trough$':>9s} "
            f"{'peak$':>9s} {'downsw%':>8s} {'drift':>7s} {'dip@':>5s} {'score':>7s} surv"
        )
        print(header)
        for sym, row in top:
            mark = "YES" if sym in survivors else "-"
            print(
                f"{sym:11s} {row['swings_per_year']:>6} {row['trough_zone_std']:>8} "
                f"{row['median_trough']:>9} {row['median_peak']:>9} "
                f"{row['median_down_swing_pct']:>8} {row['drift_per_year']:>7} "
                f"{row['dip_position_now']:>5} {row['score']:>7}  {mark}"
            )

    seen: set[str] = set()
    unique: list[dict] = []
    for entry in sorted(watchlist, key=lambda e: -e["score"]):
        if entry["symbol"] not in seen:
            seen.add(entry["symbol"])
            unique.append(entry)
    zones = {
        "_howto": (
            "Watchlist for scripts/swing_zone_watch.py. Edit freely; the "
            "watcher alerts (display-only) when close <= trough_zone * 1.10."
        ),
        "updated": date.today().isoformat(),
        "entries": unique[:WATCHLIST_MAX],
    }

    from edgestack.data.catalog import atomic_write_bytes

    atomic_write_bytes(REPORT_PATH, json.dumps(report, indent=2).encode("utf-8"))
    atomic_write_bytes(ZONES_PATH, json.dumps(zones, indent=2).encode("utf-8"))
    print(f"\nwatchlist -> {ZONES_PATH.name}: {[e['symbol'] for e in zones['entries']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
