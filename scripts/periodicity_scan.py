"""Periodicity scan: which assets move most like a sine/cosine wave?

Question (2026-07-23, user request): across stocks, ETFs, and commodity
futures in the catalog, which series have the most sinusoidal price cycles,
and how strong is their correlation with a fitted sine wave?

Honest caveats, stated up front:
- Scanning ~600 series for spectral peaks WILL surface sine-lookalikes by
  pure chance. Every candidate metric is therefore compared against a NULL
  distribution: the identical procedure run on white-noise random walks of
  matched length. A candidate matters only if it beats the null's 95th
  percentile AND keeps the same dominant period in both data halves.
- The test operates on log RETURNS, not prices. A random-walk PRICE series
  shows huge spurious low-frequency power (1/f^2 spectrum) and would make
  everything look periodic; return spectra are flat under the null.
- Sine-fit R^2 on detrended log PRICE is reported because that is the
  intuitive "correlates with a sine wave" number - but it is biased upward
  by construction (the period is chosen from the same data), which is
  exactly why the null quantiles are printed next to it.
- Detected seasonality is not a tradable edge by itself: futures carry the
  cycle in the CURVE (contango/backwardation prices the winter premium in
  advance), so buying an ETF ahead of the seasonal peak mostly pays the roll
  cost, not the cycle. Any strategy built on a finding here still owes the
  standard survivor-bar study before alerts or tickets.
- All data previously accessed; not investment advice.

Method: daily log returns of adj_close; FFT periodogram over periods
MIN_PERIOD..MAX_PERIOD trading days; Fisher g-test at the dominant ordinate;
split-half stability of the dominant period; sine+cosine fit at the dominant
period on linearly detrended log price -> correlation R, R^2, amplitude.
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
REPORT_PATH = ROOT / "artifacts" / "periodicity_scan.json"

MIN_SESSIONS = 2520  # ~10 years: an annual cycle must repeat ~10 times
MIN_PERIOD = 40  # trading days (~2 months); shorter is noise/microstructure
MAX_PERIOD = 756  # ~3 years; longer cannot repeat often enough to test
MIN_PRICE = 1.0
MAX_ABS_RETURN = 2.0  # corrupted-series screen (repo convention)
STABILITY_TOL = 0.20  # halves must agree on the period within 20%
NULL_DRAWS = 300
TOP_N = 20
ANNUAL_PERIOD = 252


def list_symbols() -> list[str]:
    return sorted(p.stem for p in PRICES.glob("*.parquet"))


def load_returns(symbol: str) -> tuple[pd.Series, float, str | None]:
    """(log returns, median daily dollar volume in millions LOCAL ccy, drop reason)."""
    df = pd.read_parquet(PRICES / f"{symbol}.parquet")
    df = df.assign(date=pd.to_datetime(df["date"])).set_index("date").sort_index()
    adj = df["adj_close"].astype(float)
    adj = adj[adj > 0]
    if len(adj) < MIN_SESSIONS:
        return pd.Series(dtype=float), 0.0, "too short"
    ret = np.log(adj).diff().dropna()
    if ret.abs().max() > MAX_ABS_RETURN:
        return pd.Series(dtype=float), 0.0, "corrupted (|ret|>200%)"
    if float(df["close"].iloc[-1]) < MIN_PRICE:
        return pd.Series(dtype=float), 0.0, "price < $1"
    tail = df.tail(504)
    volume = tail["volume"] if "volume" in tail.columns else pd.Series(0.0, index=tail.index)
    dvol = float((tail["close"] * volume.fillna(0.0)).median()) / 1e6
    return ret, dvol, None


def periodogram(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(periods, power) for the in-band FFT periodogram of demeaned returns."""
    x = returns - returns.mean()
    n = len(x)
    power = np.abs(np.fft.rfft(x)) ** 2 / n
    freqs = np.fft.rfftfreq(n)
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    band = (periods >= MIN_PERIOD) & (periods <= MAX_PERIOD)
    return periods[band], power[band]


def dominant_period(returns: np.ndarray) -> tuple[float, float, float]:
    """(period, fisher_g, fisher_p) of the strongest in-band cycle."""
    periods, power = periodogram(returns)
    if len(power) < 8 or power.sum() <= 0:
        return np.nan, 0.0, 1.0
    idx = int(np.argmax(power))
    m = len(power)
    g = float(power[idx] / power.sum())
    p_value = float(min(1.0, m * (1.0 - g) ** (m - 1)))
    return float(periods[idx]), g, p_value


def sine_fit(log_price: np.ndarray, period: float) -> tuple[float, float]:
    """(corr_r, amplitude_pct) of a sine+cosine at `period` vs detrended log price."""
    n = len(log_price)
    t = np.arange(n)
    trend = np.polynomial.polynomial.polyfit(t, log_price, 1)
    resid = log_price - np.polynomial.polynomial.polyval(t, trend)
    design = np.column_stack(
        [np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period), np.ones(n)]
    )
    coef, *_ = np.linalg.lstsq(design, resid, rcond=None)
    fitted = design @ coef
    denom = resid.std()
    corr = float(np.corrcoef(fitted, resid)[0, 1]) if denom > 0 and fitted.std() > 0 else 0.0
    amplitude = float(np.hypot(coef[0], coef[1]))
    return corr, amplitude


def analyze(returns: pd.Series) -> dict:
    r = returns.to_numpy()
    period, g, p_value = dominant_period(r)
    half = len(r) // 2
    p1, _, _ = dominant_period(r[:half])
    p2, _, _ = dominant_period(r[half:])
    stable = (
        np.isfinite(p1)
        and np.isfinite(p2)
        and abs(p1 - p2) / max(p1, p2) <= STABILITY_TOL
        and np.isfinite(period)
    )
    log_price = np.concatenate([[0.0], np.cumsum(r)])
    corr, amplitude = sine_fit(log_price, period) if np.isfinite(period) else (0.0, 0.0)
    periods, power = periodogram(r)
    annual_idx = int(np.argmin(np.abs(periods - ANNUAL_PERIOD))) if len(periods) else 0
    annual_share = float(power[annual_idx] / power.sum()) if len(power) and power.sum() else 0.0
    return {
        "sessions": len(r),
        "dominant_period_days": round(period, 1) if np.isfinite(period) else None,
        "half1_period": round(p1, 1) if np.isfinite(p1) else None,
        "half2_period": round(p2, 1) if np.isfinite(p2) else None,
        "period_stable": bool(stable),
        "fisher_g": round(g, 5),
        "fisher_p": round(p_value, 5),
        "sine_corr_r": round(corr, 4),
        "sine_r2": round(corr**2, 4),
        "amplitude_pct": round(100 * amplitude, 2),
        "annual_power_share": round(annual_share, 5),
    }


def null_distribution(lengths: list[int], rng: np.random.Generator) -> dict:
    """Identical procedure on white-noise returns: what chance alone produces."""
    corrs, gs, stables = [], [], 0
    for _ in range(NULL_DRAWS):
        n = int(rng.choice(lengths))
        r = rng.standard_normal(n) * 0.015
        period, g, _ = dominant_period(r)
        half = n // 2
        p1, _, _ = dominant_period(r[:half])
        p2, _, _ = dominant_period(r[half:])
        if np.isfinite(p1) and np.isfinite(p2) and abs(p1 - p2) / max(p1, p2) <= STABILITY_TOL:
            stables += 1
        log_price = np.concatenate([[0.0], np.cumsum(r)])
        corr, _ = sine_fit(log_price, period)
        corrs.append(abs(corr))
        gs.append(g)
    return {
        "draws": NULL_DRAWS,
        "sine_corr_p50": round(float(np.percentile(corrs, 50)), 4),
        "sine_corr_p95": round(float(np.percentile(corrs, 95)), 4),
        "fisher_g_p95": round(float(np.percentile(gs, 95)), 5),
        "stable_rate": round(stables / NULL_DRAWS, 4),
    }


def main() -> int:
    import argparse

    global MIN_PERIOD, MAX_PERIOD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-period", type=int, default=MIN_PERIOD, help="band floor, days")
    parser.add_argument("--max-period", type=int, default=MAX_PERIOD, help="band cap, days")
    parser.add_argument(
        "--min-dvol",
        type=float,
        default=0.0,
        help="liquidity screen: min median daily volume in millions (LOCAL currency)",
    )
    args = parser.parse_args()
    default_band = (args.min_period, args.max_period) == (MIN_PERIOD, MAX_PERIOD)
    MIN_PERIOD, MAX_PERIOD = args.min_period, args.max_period
    report_path = (
        REPORT_PATH
        if default_band
        else REPORT_PATH.with_name(f"periodicity_scan_{MIN_PERIOD}_{MAX_PERIOD}.json")
    )

    rng = np.random.default_rng(20260723)
    results: dict[str, dict] = {}
    dropped: dict[str, str] = {}
    lengths: list[int] = []
    for symbol in list_symbols():
        try:
            returns, dvol, reason = load_returns(symbol)
        except Exception as exc:  # unreadable parquet etc.
            dropped[symbol] = f"load failed: {exc}"
            continue
        if reason:
            dropped[symbol] = reason
            continue
        if args.min_dvol and dvol < args.min_dvol:
            dropped[symbol] = f"illiquid (median dvol {dvol:.1f}M < {args.min_dvol}M)"
            continue
        row = analyze(returns)
        row["median_dvol_m"] = round(dvol, 1)
        results[symbol] = row
        lengths.append(len(returns))

    null = null_distribution(lengths or [2520], rng)

    survivors = {
        sym: row
        for sym, row in results.items()
        if row["period_stable"]
        and abs(row["sine_corr_r"]) > null["sine_corr_p95"]
        and row["fisher_g"] > null["fisher_g_p95"]
    }
    ranked = sorted(results.items(), key=lambda kv: -abs(kv[1]["sine_corr_r"]))

    report = {
        "batch_id": uuid.uuid4().hex[:12],
        "universe": len(results),
        "dropped": len(dropped),
        "drop_reasons": dropped,
        "band_days": [MIN_PERIOD, MAX_PERIOD],
        "null": null,
        "survivors_beating_null_and_stable": {sym: results[sym] for sym in sorted(survivors)},
        "top_by_sine_corr": dict(ranked[:TOP_N]),
        "disclaimer": (
            "Historical scan on previously-accessed data; sine-fit R is biased "
            "upward because the period is chosen from the same data (compare "
            "against the null quantiles); seasonality in futures is largely "
            "priced into the curve; not investment advice."
        ),
    }
    from edgestack.data.catalog import atomic_write_bytes

    atomic_write_bytes(report_path, json.dumps(report, indent=2).encode("utf-8"))

    print(f"scanned {len(results)} symbols ({len(dropped)} dropped)")
    print(
        f"NULL (white noise, {NULL_DRAWS} draws): median |sine r| "
        f"{null['sine_corr_p50']}, 95th pct {null['sine_corr_p95']}, "
        f"g 95th pct {null['fisher_g_p95']}, stable-by-chance rate {null['stable_rate']}"
    )
    print("\nsurvivors (stable period AND beats null 95th pct on BOTH metrics):")
    header = (
        f"{'symbol':12s} {'period_d':>8s} {'h1/h2':>14s} "
        f"{'sine_r':>7s} {'g':>7s} {'amp%':>6s} {'dvolM':>8s}"
    )
    print(header)
    for sym in sorted(survivors, key=lambda s: -abs(results[s]["sine_corr_r"])):
        row = results[sym]
        print(
            f"{sym:12s} {row['dominant_period_days']:>8} "
            f"{row['half1_period']}/{row['half2_period']:>6} "
            f"{row['sine_corr_r']:>7} {row['fisher_g']:>7} "
            f"{row['amplitude_pct']:>6} {row['median_dvol_m']:>8}"
        )
    print(f"\ntop {TOP_N} by raw |sine correlation| (most are NOT significant - check flags):")
    print(header + "  stable")
    for sym, row in ranked[:TOP_N]:
        print(
            f"{sym:12s} {row['dominant_period_days']:>8} "
            f"{row['half1_period']}/{row['half2_period']:>6} "
            f"{row['sine_corr_r']:>7} {row['fisher_g']:>7} "
            f"{row['amplitude_pct']:>6} {row['median_dvol_m']:>8}  "
            f"{'YES' if row['period_stable'] else '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
