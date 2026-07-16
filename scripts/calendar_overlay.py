"""Calendar-conditioned leverage overlay on SPY/QQQ — the combination test.

Pre-registered rules (all documented in decades-old literature — Halloween
effect, September weakness, turn-of-month, trend filter — NOT tuned here):

    exposure L(t), decided at close t-1, applied to session t:
      base                      1.0
      September                 0.5
      October + November        1.5
      turn-of-month (last 1 + first 3 sessions)   +0.5  (cap 2.0)
      7th trading day of November                 0.0   (measured worst day)
      RISK GATES (override, applied last):
        close < 200dma          -> min(L, 0.5)
        20d realized vol > 20%  -> min(L, 1.0)

Costs: 2 bps per unit of exposure traded; borrowing on (L-1)+ at 13-week
T-bill + 150 bps (IBKR-like); idle cash (1-L)+ earns the T-bill rate.

Splits: full history, pre-2016 "development era", 2016-2023 OOS,
2024-2026 untouched holdout. Benchmark: unlevered buy-and-hold.
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

TRADE_COST = 0.0002  # 2 bps per unit exposure change (SPY/QQQ spreads)
BORROW_SPREAD = 0.015  # broker margin spread over T-bills


def tbill_series(session: requests.Session, index: pd.DatetimeIndex) -> pd.Series:
    try:
        irx = fetch(session, "^IRX", interval="1d", period1=0, period2=int(time.time()))
        s = pd.Series(
            irx["close"].to_numpy() / 100.0, index=irx["dt"].dt.tz_localize(None).dt.normalize()
        )
        return s.reindex(index).ffill().fillna(0.02)
    except Exception:
        return pd.Series(0.02, index=index)


def build_exposure(df: pd.DataFrame, base: float = 1.0) -> pd.Series:
    d = df.copy()
    d["date"] = d["dt"].dt.tz_localize(None).dt.normalize()
    d = d.set_index("date")
    ym = d.index.to_period("M")
    tdom = pd.Series(d.groupby(ym).cumcount() + 1, index=d.index)
    tdom_end = pd.Series(d.groupby(ym).cumcount(ascending=False) + 1, index=d.index)
    month = pd.Series(d.index.month, index=d.index)

    L = pd.Series(1.0, index=d.index)
    L[month == 9] = 0.5
    L[(month == 10) | (month == 11)] = 1.5
    tom = (tdom <= 3) | (tdom_end == 1)
    L = (L + tom * 0.5).clip(upper=2.0)
    L[(month == 11) & (tdom == 7)] = 0.0

    sma200 = d["close"].rolling(200).mean()
    ret = d["adj"].pct_change()
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    L = L.mask(d["close"] < sma200, L.clip(upper=0.5))
    L = L.mask(vol20 > 0.20, L.clip(upper=1.0))
    L.iloc[:200] = 1.0  # warmup: plain buy-and-hold
    # decided at close t-1, applied to session t
    applied = L.shift(1).fillna(1.0)
    if base != 1.0:  # levered variant: scale whole series
        applied = (applied * base).clip(upper=2.5)
    return applied, ret


def run(sym: str, session: requests.Session) -> dict:
    df = fetch(session, sym, interval="1d", period1=0, period2=int(time.time()))
    L, ret = build_exposure(df)
    rf = tbill_series(session, L.index) / 252.0
    borrow = (L - 1).clip(lower=0) * (rf + BORROW_SPREAD / 252.0)
    cash = (1 - L).clip(lower=0) * rf
    costs = L.diff().abs().fillna(0.0) * TRADE_COST
    strat = L * ret - borrow + cash - costs

    out = {"symbol": sym, "avg_exposure": round(float(L.mean()), 2)}
    for label, lo, hi in (
        ("full", str(L.index[0].date()), "2026-12-31"),
        ("development_pre2016", "1994-01-01", "2015-12-31"),
        ("OOS_2016_2023", "2016-01-01", "2023-12-31"),
        ("HOLDOUT_2024_2026", "2024-01-01", "2026-12-31"),
    ):
        s = strat.loc[lo:hi].dropna()
        b = ret.loc[s.index].fillna(0.0)
        if len(s) < 200:
            continue
        yrs = len(s) / 252
        nw = newey_west_alpha(s, b)
        out[label] = {
            "years": round(yrs, 1),
            "strat_cagr": round((float(np.prod(1 + s)) ** (1 / yrs)) - 1, 4),
            "bh_cagr": round((float(np.prod(1 + b)) ** (1 / yrs)) - 1, 4),
            "strat_sharpe": round(sharpe_ratio(s.to_numpy()), 2),
            "bh_sharpe": round(sharpe_ratio(b.to_numpy()), 2),
            "strat_maxdd": round(max_drawdown(s.to_numpy()), 3),
            "bh_maxdd": round(max_drawdown(b.to_numpy()), 3),
            "nw_alpha_ann": round(nw["alpha_ann"], 4),
            "alpha_t": round(nw["alpha_t"], 2),
            "beta": round(nw["beta"], 2),
        }
    return out


def main() -> int:
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    results = [run(sym, session) for sym in ("SPY", "QQQ")]
    for r in results:
        print(
            f"\n=== {r['symbol']} calendar-leverage overlay (avg exposure {r['avg_exposure']}x) ==="
        )
        print(
            f"{'period':<22}{'strat':>8}{'B&H':>8}{'Sh(s)':>7}{'Sh(b)':>7}"
            f"{'DD(s)':>8}{'DD(b)':>8}{'alpha':>8}{'t':>6}"
        )
        for k, v in r.items():
            if not isinstance(v, dict):
                continue
            print(
                f"{k:<22}{v['strat_cagr']:>8.1%}{v['bh_cagr']:>8.1%}"
                f"{v['strat_sharpe']:>7.2f}{v['bh_sharpe']:>7.2f}"
                f"{v['strat_maxdd']:>8.1%}{v['bh_maxdd']:>8.1%}"
                f"{v['nw_alpha_ann']:>8.1%}{v['alpha_t']:>6.2f}"
            )
    atomic_write_bytes(
        Path("artifacts") / "calendar_overlay.json", json.dumps(results, indent=2).encode()
    )
    print("\nsaved -> artifacts/calendar_overlay.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
