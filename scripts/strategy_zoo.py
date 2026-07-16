"""Historical, non-actionable strategy zoo; no result is V2 promotion evidence.

Test every daily-bar-testable strategy family from the
canonical retail catalogue, on SPY/QQQ + 11 sector ETFs, 1999->present.

Families covered here (time-series, long/flat, signal at close t -> earns
session t+1, 2 bps per unit exposure change):
  trend (MA crosses, Donchian, Supertrend, PSAR, ADX, TS-momentum, 52wh)
  mean reversion (RSI2/RSI14, Bollinger fade, IBS, N-down-days, pullback)
  breakout / volatility compression (NR7, inside day, BB squeeze, 20d/52w)
  candlestick reversals (event studies: next-day excess return)
  volatility management (vol targeting, vol regime)
  position management (trailing/chandelier stop, fixed stop + reentry)
  composites (trend+dip, trend+ToM, dual momentum)

Splits: DEV 1999-2015, VAL 2016-2023, HISTORICAL 2024+ (previously accessed;
not used for V2 promotion decisions). Historical survivor bar: Sharpe >=
buy-and-hold in all three splits AND
pooled Newey-West alpha t >= 2 vs the same instrument's buy-and-hold.
Every (rule, instrument) run is counted as a trial for the multiplicity note.
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
from edgestack.validation.advanced_tests import newey_west_alpha
from edgestack.validation.metrics import max_drawdown, sharpe_ratio

COST = 0.0002
TICKERS = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]
SPLITS = {
    "dev_1999_2015": ("1999-01-01", "2015-12-31"),
    "val_2016_2023": ("2016-01-01", "2023-12-31"),
    "holdout_2024": ("2024-01-01", "2026-12-31"),
}
CACHE = Path("data/cache/zoo")


# ---------------------------------------------------------------- indicators
def sma(s, n):
    return s.rolling(n).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    trn = atr(df, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / trn
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / trn
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.Series:
    a = atr(df, n)
    mid = (df["high"] + df["low"]) / 2
    ub, lb = (mid + mult * a).to_numpy(), (mid - mult * a).to_numpy()
    close = df["close"].to_numpy()
    pos = np.zeros(len(df))
    trend_up, level = True, lb[0] if np.isfinite(lb[0]) else 0.0
    for i in range(1, len(df)):
        if trend_up:
            level = max(level, lb[i]) if np.isfinite(lb[i]) else level
            if close[i] < level:
                trend_up, level = False, ub[i]
        else:
            level = min(level, ub[i]) if np.isfinite(ub[i]) else level
            if close[i] > level:
                trend_up, level = True, lb[i]
        pos[i] = 1.0 if trend_up else 0.0
    return pd.Series(pos, index=df.index)


def psar(df: pd.DataFrame, af0=0.02, afmax=0.2) -> pd.Series:
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    pos = np.zeros(n)
    up, sar, ep, af = True, low[0], high[0], af0
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            if low[i] < sar:
                up, sar, ep, af = False, ep, low[i], af0
            elif high[i] > ep:
                ep, af = high[i], min(af + af0, afmax)
        else:
            if high[i] > sar:
                up, sar, ep, af = True, ep, high[i], af0
            elif low[i] < ep:
                ep, af = low[i], min(af + af0, afmax)
        pos[i] = 1.0 if up else 0.0
    return pd.Series(pos, index=df.index)


def hold_n(trigger: pd.Series, n: int) -> pd.Series:
    """Position = 1 for n sessions after each trigger."""
    return (trigger.astype(float).rolling(n, min_periods=1).max()).fillna(0.0)


# ---------------------------------------------------------------- rule library
def build_rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    c, h, low = df["close"], df["high"], df["low"]
    ret = df["adj"].pct_change()
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    s200, s50, s20, s10 = sma(c, 200), sma(c, 50), sma(c, 20), sma(c, 10)
    r2, r14 = rsi(c, 2), rsi(c, 14)
    bb_mid, bb_sd = s20, c.rolling(20).std()
    ibs = ((c - low) / (h - low).replace(0, np.nan)).fillna(0.5)
    dc_hi20, dc_lo10 = h.rolling(20).max(), low.rolling(10).min()
    dc_hi55, dc_lo20 = h.rolling(55).max(), low.rolling(20).min()
    rng = h - low
    nr7 = rng == rng.rolling(7).min()
    inside = (h < h.shift()) & (low > low.shift())
    hi52, hi20 = c.rolling(252).max(), c.rolling(20).max()
    adx14, pdi, mdi = adx(df)
    tdom = df.groupby(df.index.to_period("M")).cumcount() + 1
    tdom_end = df.groupby(df.index.to_period("M")).cumcount(ascending=False) + 1
    tom = (tdom <= 3) | (tdom_end == 1)

    rules: dict[str, pd.Series] = {
        # --- trend family ---
        "trend_sma10x20": (s10 > s20).astype(float),
        "trend_sma20x50": (s20 > s50).astype(float),
        "trend_sma50x200": (s50 > s200).astype(float),
        "trend_px_gt_sma200": (c > s200).astype(float),
        "trend_px_gt_sma50": (c > s50).astype(float),
        "trend_ema9x20": (ema(c, 9) > ema(c, 20)).astype(float),
        "trend_macd": (ema(c, 12) - ema(c, 26) > ema(ema(c, 12) - ema(c, 26), 9)).astype(float),
        "trend_supertrend": supertrend(df),
        "trend_psar": psar(df),
        "trend_adx25_di": ((adx14 > 25) & (pdi > mdi)).astype(float),
        "mom_ts_12m": (c / c.shift(252) > 1).astype(float),
        "mom_ts_6m": (c / c.shift(126) > 1).astype(float),
        "mom_ts_3m": (c / c.shift(63) > 1).astype(float),
        "mom_near_52wh": (c >= 0.98 * hi52).astype(float),
        # --- Donchian / turtle ---
        "brk_donchian_20_10": None,  # filled below (stateful)
        "brk_donchian_55_20": None,
        # --- breakout / compression ---
        "brk_20d_high_hold10": hold_n((c > hi20.shift()).astype(float), 10),
        "brk_52w_high_hold21": hold_n((c > hi52.shift()).astype(float), 21),
        "brk_nr7_up_hold5": hold_n(((c > h.shift()) & nr7.shift()).astype(float), 5),
        "brk_inside_up_hold5": hold_n(((c > h.shift()) & inside.shift()).astype(float), 5),
        "brk_bb_squeeze": hold_n(
            (
                (c > bb_mid + 2 * bb_sd)
                & ((4 * bb_sd / bb_mid).rolling(120).rank(pct=True).shift() < 0.2)
            ).astype(float),
            10,
        ),
        # --- mean reversion (long-only, trend-gated where classic) ---
        "mr_rsi2_dip_uptrend": ((r2 < 10) & (c > s200)).astype(float),
        "mr_rsi14_30": (r14 < 30).astype(float),
        "mr_bb_fade_uptrend": ((c < bb_mid - 2 * bb_sd) & (c > s200)).astype(float),
        "mr_ibs_low": (ibs < 0.2).astype(float),
        "mr_3down_days": ((ret < 0) & (ret.shift() < 0) & (ret.shift(2) < 0)).astype(float),
        "mr_5d_low_uptrend": ((c <= low.rolling(5).min().shift() * 1.001) & (c > s200)).astype(
            float
        ),
        # --- volatility management ---
        "vol_target_10pct": (0.10 / vol20).clip(upper=1.5).fillna(0.0),
        "vol_regime_lt20": (vol20 < 0.20).astype(float),
        # --- composites ---
        "cmb_trend_or_rsi2dip": np.maximum(
            (c > s200).astype(float) * 1.0, (r2 < 10).astype(float) * 1.0
        ),
        "cmb_trend_and_lowvol": ((c > s200) & (vol20 < 0.20)).astype(float),
        "cmb_trend_plus_tom": ((c > s200) | tom).astype(float),
        "cmb_trend_dip_buy": ((c > s200) & ((r2 < 25) | (ibs < 0.3))).astype(float),
    }

    # Donchian stateful long/flat: enter on N-high, exit on M-low
    for name, ehi, xlo in (
        ("brk_donchian_20_10", dc_hi20, dc_lo10),
        ("brk_donchian_55_20", dc_hi55, dc_lo20),
    ):
        e = (c > ehi.shift()).to_numpy()
        x = (c < xlo.shift()).to_numpy()
        pos = np.zeros(len(c))
        on = False
        for i in range(len(c)):
            if on and x[i]:
                on = False
            elif not on and e[i]:
                on = True
            pos[i] = 1.0 if on else 0.0
        rules[name] = pd.Series(pos, index=c.index)

    # Chandelier trailing stop on buy-and-hold (position mgmt family)
    a22 = atr(df, 22)
    chand = c.rolling(22).max() - 3 * a22
    e = (c > hi20.shift()).to_numpy()
    x = (c < chand.shift()).to_numpy()
    pos = np.zeros(len(c))
    on = True
    for i in range(len(c)):
        if on and x[i]:
            on = False
        elif not on and e[i]:
            on = True
        pos[i] = 1.0 if on else 0.0
    rules["pm_chandelier_stop"] = pd.Series(pos, index=c.index)

    # Fixed -5% stop from rolling entry, re-enter after new 20d high
    dd_from_20h = c / hi20 - 1
    rules["pm_stop5_reenter"] = ((dd_from_20h > -0.05) | (c > hi20.shift())).astype(float)
    return rules


CANDLES = {
    "cs_bull_engulf": lambda o, h, low, c: (
        (c > o) & (c.shift() < o.shift()) & (c > o.shift()) & (o < c.shift())
    ),
    "cs_bear_engulf": lambda o, h, low, c: (
        (c < o) & (c.shift() > o.shift()) & (c < o.shift()) & (o > c.shift())
    ),
    "cs_hammer": lambda o, h, low, c: (
        ((np.minimum(o, c) - low) > 2 * (c - o).abs()) & ((h - np.maximum(o, c)) < (c - o).abs())
    ),
    "cs_shooting_star": lambda o, h, low, c: (
        ((h - np.maximum(o, c)) > 2 * (c - o).abs()) & ((np.minimum(o, c) - low) < (c - o).abs())
    ),
    "cs_doji": lambda o, h, low, c: (c - o).abs() < 0.1 * (h - low),
    "cs_3_white_soldiers": lambda o, h, low, c: (
        (c > o)
        & (c.shift() > o.shift())
        & (c.shift(2) > o.shift(2))
        & (c > c.shift())
        & (c.shift() > c.shift(2))
    ),
    "cs_outside_up": lambda o, h, low, c: (h > h.shift()) & (low < low.shift()) & (c > c.shift()),
}


def run_rule(pos: pd.Series, ret: pd.Series) -> pd.Series:
    p = pos.shift(1).fillna(0.0)
    return p * ret - COST * p.diff().abs().fillna(0.0)


def main() -> int:
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

    results, candle_rows, trials = [], [], 0
    for sym in TICKERS:
        cache_f = CACHE / f"{sym}.parquet"
        if cache_f.exists():
            df = pd.read_parquet(cache_f)
        else:
            df = fetch(session, sym, interval="1d", period1=0, period2=int(time.time()))
            df = df[["dt", "open", "high", "low", "close", "adj"]]
            df.to_parquet(cache_f, index=False)
            time.sleep(0.3)
        df = (
            df.assign(date=df["dt"].dt.tz_localize(None).dt.normalize())
            .set_index("date")
            .drop(columns="dt")
        )
        ret = df["adj"].pct_change()
        rules = build_rules(df)
        for name, pos in rules.items():
            trials += 1
            strat = run_rule(pos, ret)
            row = {"rule": name, "symbol": sym, "avg_exposure": round(float(pos.mean()), 2)}
            ok_all = True
            for split, (lo, hi) in SPLITS.items():
                s = strat.loc[lo:hi].dropna()
                b = ret.loc[s.index].fillna(0.0)
                if len(s) < 200:
                    ok_all = False
                    continue
                sh_s, sh_b = sharpe_ratio(s.to_numpy()), sharpe_ratio(b.to_numpy())
                row[split] = {
                    "sharpe": round(sh_s, 2),
                    "bh_sharpe": round(sh_b, 2),
                    "cagr": round(float(np.prod(1 + s)) ** (252 / len(s)) - 1, 4),
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

        # candlestick event studies: next-session excess return after pattern
        o, h, low, c = df["open"], df["high"], df["low"], df["close"]
        nxt = ret.shift(-1)
        for name, fn in CANDLES.items():
            trials += 1
            sig = fn(o, h, low, c).fillna(False)
            hit = nxt[sig].dropna()
            base = nxt.dropna()
            if len(hit) < 50:
                continue
            edge = hit.mean() - base.mean()
            t = edge / (hit.std(ddof=1) / np.sqrt(len(hit)))
            candle_rows.append(
                {
                    "pattern": name,
                    "symbol": sym,
                    "n": len(hit),
                    "next_day_excess_bps": round(edge * 1e4, 1),
                    "t": round(float(t), 2),
                }
            )
        print(f"{sym}: {len(rules)} rules + {len(CANDLES)} patterns ({time.time() - t0:.0f}s)")

    survivors = [r for r in results if r["SURVIVOR"]]
    print(f"\ntrials: {trials} (expect ~{trials * 0.025:.0f} false |t|>2 by chance)")
    print(f"survivors (Sharpe>=B&H in all 3 splits AND alpha t>=2): {len(survivors)}")
    for r in survivors:
        print(
            f"  {r['rule']:<26}{r['symbol']:<6}alpha {r['alpha_ann']:+.1%}/yr "
            f"t={r['alpha_t']:.2f} exp={r['avg_exposure']}"
        )

    # near-survivors: pass 2 of 3 splits with pooled alpha_t >= 1.5
    def n_splits_ok(r):
        return sum(1 for s in SPLITS if s in r and r[s]["sharpe"] >= r[s]["bh_sharpe"])

    near = [r for r in results if not r["SURVIVOR"] and n_splits_ok(r) >= 2 and r["alpha_t"] >= 1.5]
    near.sort(key=lambda r: -r["alpha_t"])
    print(f"\nnear-survivors (2/3 splits, t>=1.5): {len(near)}; top 12:")
    for r in near[:12]:
        print(
            f"  {r['rule']:<26}{r['symbol']:<6}alpha {r['alpha_ann']:+.1%}/yr t={r['alpha_t']:.2f}"
        )

    cdf = pd.DataFrame(candle_rows)
    if len(cdf):
        pooled = cdf.groupby("pattern").apply(
            lambda g: pd.Series(
                {
                    "n": g["n"].sum(),
                    "mean_bps": round(
                        float(np.average(g["next_day_excess_bps"], weights=g["n"])), 1
                    ),
                    "median_t": round(float(g["t"].median()), 2),
                }
            ),
            include_groups=False,
        )
        print("\n=== candlestick patterns: next-day EXCESS return (pooled across 11 ETFs) ===")
        print(pooled.sort_values("mean_bps", ascending=False).to_string())

    atomic_write_bytes(
        Path("artifacts") / "strategy_zoo.json",
        json.dumps(
            {
                "trials": trials,
                "results": results,
                "candles": candle_rows,
                "survivors": [f"{r['rule']}/{r['symbol']}" for r in survivors],
            },
            indent=1,
        ).encode(),
    )
    print(f"\nsaved -> artifacts/strategy_zoo.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
