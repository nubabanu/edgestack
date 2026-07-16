"""Allocation zoo: monthly/yearly strategy families on a multi-asset ETF
universe, 2003->present (limited by ETF inception dates).

Families: passive allocations (60/40, permanent, all-weather, equal-weight),
rebalancing variants, trend/TAA (Faber 10-month SMA, GTAA, TSMOM), dual
momentum (GEM), asset-class & sector rotation, risk parity, vol targeting,
seasonality (Halloween, January barometer), and combinations.

Monthly signals at month-end close -> hold next month. Costs: 5 bps one-way
x turnover. Splits: dev 2008-2015, val 2016-2023, holdout 2024+.
Benchmarks: SPY buy-and-hold AND 60/40 monthly. Survivor bar: Sharpe >=
BOTH benchmarks in all three splits AND pooled NW alpha (vs SPY) t >= 2.
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

COST = 0.0005
ASSETS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "SHY", "LQD", "HYG", "GLD", "DBC", "VNQ"]
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]
SPLITS = {
    "dev_2008_2015": ("2008-01-31", "2015-12-31"),
    "val_2016_2023": ("2016-01-01", "2023-12-31"),
    "holdout_2024": ("2024-01-01", "2026-12-31"),
}
CACHE = Path("data/cache/zoo")


def monthly_prices(tickers: list[str]) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    cols = {}
    for sym in tickers:
        f = CACHE / f"{sym}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
        else:
            df = fetch(session, sym, interval="1d", period1=0, period2=int(time.time()))
            df = df[["dt", "open", "high", "low", "close", "adj"]]
            df.to_parquet(f, index=False)
            time.sleep(0.3)
        s = pd.Series(df["adj"].to_numpy(), index=df["dt"].dt.tz_localize(None).dt.normalize())
        cols[sym] = s.resample("ME").last()
    return pd.DataFrame(cols)


def run_weights(w: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    w = w.reindex(columns=rets.columns).fillna(0.0)
    held = w.shift(1)
    gross = (held * rets).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    return gross - COST * turnover


def const_weights(idx, alloc: dict[str, float], every: int = 1) -> pd.DataFrame:
    """Rebalance to fixed weights every `every` months (0 = never: drift)."""
    w = pd.DataFrame(0.0, index=idx, columns=list(alloc))
    for i, _t in enumerate(idx):
        if every and i % every == 0:
            for k, v in alloc.items():
                w.iloc[i, w.columns.get_loc(k)] = v
        else:
            w.iloc[i] = np.nan
    return w.ffill().fillna(0.0)


def drift_6040(rets: pd.DataFrame, bands: float | None = None) -> pd.DataFrame:
    """60/40 with drift; rebalance only when |w_spy - .6| > bands (None=never)."""
    idx = rets.index
    w = pd.DataFrame(0.0, index=idx, columns=["SPY", "IEF"])
    ws, wb = 0.6, 0.4
    for i, t in enumerate(idx):
        if i:
            gs = ws * (1 + np.nan_to_num(rets.loc[t, "SPY"]))
            gb = wb * (1 + np.nan_to_num(rets.loc[t, "IEF"]))
            ws, wb = gs / (gs + gb), gb / (gs + gb)
        if bands is not None and abs(ws - 0.6) > bands:
            ws, wb = 0.6, 0.4
        w.loc[t] = [ws, wb]
    return w


def main() -> int:
    t0 = time.time()
    px = monthly_prices(ASSETS + SECTORS)
    rets = px.pct_change()
    idx = rets.index
    spy = rets["SPY"].fillna(0.0)
    mom12_1 = px.shift(1) / px.shift(12) - 1.0  # 12-1 monthly momentum
    mom6 = px / px.shift(6) - 1.0
    sma10 = px.rolling(10).mean()
    above10 = px > sma10
    vol36 = rets.rolling(36).std()

    strategies: dict[str, pd.DataFrame] = {}

    # --- passive & rebalancing family ---
    strategies["S_6040_monthly"] = const_weights(idx, {"SPY": 0.6, "IEF": 0.4}, 1)
    strategies["S_6040_annual"] = const_weights(idx, {"SPY": 0.6, "IEF": 0.4}, 12)
    strategies["S_6040_band5pct"] = drift_6040(rets, bands=0.05)
    strategies["S_6040_never_rebal"] = drift_6040(rets, bands=None)
    strategies["S_equal_weight_13"] = const_weights(idx, {a: 1 / len(ASSETS) for a in ASSETS}, 1)
    strategies["S_permanent_portfolio"] = const_weights(
        idx, {"SPY": 0.25, "TLT": 0.25, "SHY": 0.25, "GLD": 0.25}, 3
    )
    strategies["S_all_weather"] = const_weights(
        idx, {"SPY": 0.30, "TLT": 0.40, "IEF": 0.15, "GLD": 0.075, "DBC": 0.075}, 3
    )

    # risk parity (inverse 36m vol, 4 assets)
    rp_assets = ["SPY", "TLT", "GLD", "DBC"]
    iv = (1 / vol36[rp_assets]).replace([np.inf, -np.inf], np.nan)
    strategies["S_risk_parity_4"] = iv.div(iv.sum(axis=1), axis=0).fillna(0.0)

    # --- trend / TAA family ---
    w = pd.DataFrame(0.0, index=idx, columns=["SPY"])
    w["SPY"] = above10["SPY"].astype(float)  # Faber timing on SPY
    strategies["T_spy_10m_sma"] = w
    gtaa = ["SPY", "EFA", "IEF", "GLD", "VNQ"]
    strategies["T_faber_gtaa5"] = above10[gtaa].astype(float) * 0.2
    pos12 = (mom12_1[ASSETS] > 0) & above10[ASSETS]
    strategies["T_tsmom_12m_multi"] = pos12.astype(float).div(
        pos12.sum(axis=1).clip(lower=1), axis=0
    )

    # dual momentum (GEM): SPY vs EFA by 12-1; absolute filter vs SHY; else IEF
    gem = pd.DataFrame(0.0, index=idx, columns=["SPY", "EFA", "IEF"])
    rel = mom12_1["SPY"] >= mom12_1["EFA"]
    abs_ok = mom12_1[["SPY", "EFA"]].max(axis=1) > mom12_1["SHY"]
    gem.loc[rel & abs_ok, "SPY"] = 1.0
    gem.loc[~rel & abs_ok, "EFA"] = 1.0
    gem.loc[~abs_ok, "IEF"] = 1.0
    strategies["T_dual_momentum_gem"] = gem

    # asset-class momentum top-3 of 13 (12-1)
    def top_n_weights(scores: pd.DataFrame, n: int) -> pd.DataFrame:
        r = scores.rank(axis=1, ascending=False)
        w = (r <= n).astype(float)
        return w.div(w.sum(axis=1).clip(lower=1), axis=0)

    strategies["T_asset_mom_top3"] = top_n_weights(mom12_1[ASSETS], 3)
    strategies["T_asset_mom_top3_trendgate"] = top_n_weights(mom12_1[ASSETS], 3) * above10[
        ASSETS
    ].astype(float)
    strategies["R_sector_mom_top3_12_1"] = top_n_weights(mom12_1[SECTORS], 3)
    strategies["R_sector_mom_top3_6m"] = top_n_weights(mom6[SECTORS], 3)
    strategies["R_bond_duration_rot"] = top_n_weights(mom6[["TLT", "IEF", "SHY"]], 1)

    # vol targeting
    spy_vol = rets["SPY"].rolling(12).std() * np.sqrt(12)
    wv = pd.DataFrame({"SPY": (0.10 / spy_vol).clip(upper=1.0)}, index=idx)
    strategies["V_spy_voltarget_10"] = wv.fillna(0.0)
    w6040v = (
        strategies["S_6040_monthly"]
        .mul(
            (
                0.08
                / (
                    (strategies["S_6040_monthly"].shift(1) * rets).sum(axis=1).rolling(12).std()
                    * np.sqrt(12)
                )
            ).clip(upper=1.5),
            axis=0,
        )
        .fillna(0.0)
    )
    strategies["V_6040_voltarget_8"] = w6040v

    # --- seasonality family ---
    hall = pd.DataFrame(0.0, index=idx, columns=["SPY", "SHY"])
    winter = idx.month.isin([11, 12, 1, 2, 3, 4])
    hall.loc[winter, "SPY"] = 1.0
    hall.loc[~winter, "SHY"] = 1.0
    strategies["C_halloween_nov_apr"] = hall

    jan = pd.DataFrame(0.0, index=idx, columns=["SPY"])
    jan_ret = rets["SPY"].groupby(idx.year).transform(lambda g: g.iloc[0] if len(g) else np.nan)
    jan["SPY"] = np.where((jan_ret > 0) | (idx.month == 1), 1.0, 0.0)
    strategies["C_january_barometer"] = jan

    # --- combinations ---
    strategies["X_gem_plus_gtaa"] = 0.5 * gem.reindex(columns=["SPY", "EFA", "IEF"]).fillna(
        0
    ).reindex(columns=px.columns, fill_value=0.0) + 0.5 * strategies["T_faber_gtaa5"].reindex(
        columns=px.columns, fill_value=0.0
    )
    strategies["X_sector_mom_trendgate"] = top_n_weights(mom12_1[SECTORS], 3) * above10[
        SECTORS
    ].astype(float)

    bench6040 = run_weights(strategies["S_6040_monthly"], rets)
    results, trials = [], 0
    for name, w in strategies.items():
        trials += 1
        strat = run_weights(w, rets)
        row = {"strategy": name}
        ok = True
        for split, (lo, hi) in SPLITS.items():
            s = strat.loc[lo:hi].dropna()
            if len(s) < 24:
                ok = False
                continue
            b_spy = spy.loc[s.index]
            b_64 = bench6040.loc[s.index]
            sh = sharpe_ratio(s.to_numpy(), periods_per_year=12)
            sh_spy = sharpe_ratio(b_spy.to_numpy(), periods_per_year=12)
            sh_64 = sharpe_ratio(b_64.to_numpy(), periods_per_year=12)
            row[split] = {
                "sharpe": round(sh, 2),
                "spy": round(sh_spy, 2),
                "s6040": round(sh_64, 2),
                "cagr": round(float(np.prod(1 + s)) ** (12 / len(s)) - 1, 4),
                "maxdd": round(max_drawdown(s.to_numpy()), 3),
            }
            if sh < max(sh_spy, sh_64):
                ok = False
        pooled = strat.loc["2008-01-31":].dropna()
        nw = newey_west_alpha(pooled, spy.loc[pooled.index], lags=3)
        row["alpha_t"] = round(nw["alpha_t"], 2)
        # newey_west_alpha annualizes x252 (daily convention); rescale to monthly
        row["alpha_ann"] = round(nw["alpha_ann"] * 12 / 252, 4)
        row["beta"] = round(nw["beta"], 2)
        row["SURVIVOR"] = bool(ok and nw["alpha_t"] >= 2.0)
        results.append(row)

    results.sort(key=lambda r: -r["alpha_t"])
    print(f"\ntrials: {trials}")
    print(f"{'strategy':<30}{'dev':>12}{'val':>12}{'hold':>12}{'a/yr':>7}{'t':>6}{'beta':>6}  SURV")
    for r in results:

        def cell(s, row=r):
            return (
                f"{row[s]['sharpe']:.2f}/{max(row[s]['spy'], row[s]['s6040']):.2f}"
                if s in row
                else "--"
            )

        print(
            f"{r['strategy']:<30}{cell('dev_2008_2015'):>12}"
            f"{cell('val_2016_2023'):>12}{cell('holdout_2024'):>12}"
            f"{r['alpha_ann']:>7.1%}{r['alpha_t']:>6.2f}{r['beta']:>6.2f}"
            f"  {'<== SURVIVOR' if r['SURVIVOR'] else ''}"
        )

    atomic_write_bytes(
        Path("artifacts") / "allocation_zoo.json",
        json.dumps({"trials": trials, "results": results}, indent=1).encode(),
    )
    print(f"\nsaved -> artifacts/allocation_zoo.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
