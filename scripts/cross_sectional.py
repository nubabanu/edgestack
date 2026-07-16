"""Cross-sectional ranked portfolios — the architecture the timing campaigns
never tested. Literature-standard, parameter-free specifications (no tuning
on our data):

    MOM     12-1 momentum, top decile, monthly rebalance      (Jegadeesh-Titman)
    LOWVOL  252d volatility, bottom decile, monthly            (Ang et al.)
    REV5    5-day losers, bottom decile, weekly                (Lehmann)
    MOMLV   mean of momentum + low-vol ranks, top decile, monthly
    SPY     buy and hold
    SPY200  SPY when above its 200dma at signal close, else cash (signal lag 1d)

Universe: point-in-time S&P members, price >= 5, 252 sessions of history.
Signals at close t -> positions earn returns from t+1 (no same-close fills).
Costs: 15 bps one-way x two-sided turnover per rebalance (conservative).
Long-short top-minus-bottom is reported HYPOTHETICAL (no borrow data).

Periods: 2012-2023 (research era) and 2024->present (UNTOUCHED holdout —
never consumed by any prior experiment in this project).
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from edgestack.config import load_config
from edgestack.data.catalog import DataCatalog, atomic_write_bytes
from edgestack.data.universe_pit import PitSP500Universe
from edgestack.logging import configure
from edgestack.validation.advanced_tests import newey_west_alpha
from edgestack.validation.metrics import max_drawdown, sharpe_ratio

COST_ONEWAY = 0.0015  # 15 bps per side, conservative


def build_matrices():
    cfg = load_config("configs/live.yaml")  # guard at 2026-12-31: full history
    catalog = DataCatalog(cfg)
    panel = catalog.load_panel()
    px = panel.assign(p=panel["adj_close"].where(panel["adj_close"].notna(), panel["close"]))
    prices = px.pivot(index="date", columns="symbol", values="p").sort_index()
    closes = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    rets = prices.pct_change()

    universe = PitSP500Universe(Path("data/cache/universe"), earliest=date(2011, 1, 1))
    member = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    spans: dict[str, list] = {}
    for iv in universe.intervals:
        spans.setdefault(iv.symbol, []).append(iv)
    for symbol in prices.columns:
        for iv in spans.get(symbol, []):
            member.loc[
                (member.index >= pd.Timestamp(iv.start)) & (member.index < pd.Timestamp(iv.end)),
                symbol,
            ] = True
    eligible = member & (closes >= 5.0) & prices.notna()
    return prices, rets, eligible


def decile_weights(
    score: pd.DataFrame, eligible: pd.DataFrame, rebalance_every: int, top: bool, warmup: int
) -> pd.DataFrame:
    """Equal-weight decile portfolio, signals at close t -> held from t+1."""
    dates = score.index
    weights = pd.DataFrame(0.0, index=dates, columns=score.columns)
    current = pd.Series(0.0, index=score.columns)
    for i, t in enumerate(dates):
        if i >= warmup and (i - warmup) % rebalance_every == 0:
            row = score.loc[t].where(eligible.loc[t])
            valid = row.dropna()
            if len(valid) >= 50:
                k = max(10, len(valid) // 10)
                chosen = valid.nlargest(k) if top else valid.nsmallest(k)
                current = pd.Series(0.0, index=score.columns)
                current[chosen.index] = 1.0 / k
        weights.iloc[i] = current
    return weights.shift(1).fillna(0.0)  # positions earn NEXT session's return


def portfolio_returns(weights: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    gross = (weights * rets).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    return gross - COST_ONEWAY * turnover


def summarize(r: pd.Series, spy: pd.Series, label: str, lo: str, hi: str) -> dict:
    seg = r.loc[lo:hi].dropna()
    spy_seg = spy.loc[seg.index].fillna(0.0)
    if len(seg) < 120:
        return {"label": label, "n": len(seg)}
    years = len(seg) / 252
    cum = float(np.prod(1 + seg) - 1)
    nw = newey_west_alpha(seg, spy_seg)
    return {
        "label": label,
        "cagr": round((1 + cum) ** (1 / years) - 1, 4),
        "sharpe": round(sharpe_ratio(seg.to_numpy()), 2),
        "maxdd": round(max_drawdown(seg.to_numpy()), 3),
        "nw_alpha_ann": round(nw["alpha_ann"], 4),
        "alpha_t": round(nw["alpha_t"], 2),
        "beta": round(nw["beta"], 2),
        "avg_turnover_ann": None,
    }


def main() -> int:
    configure()
    prices, rets, eligible = build_matrices()
    print(f"matrix: {rets.shape[0]} sessions x {rets.shape[1]} symbols")

    mom = prices.shift(21) / prices.shift(252) - 1.0
    vol = rets.rolling(252).std()
    rev5 = prices.pct_change(5)
    mom_rank = mom.rank(axis=1, pct=True)
    lv_rank = (-vol).rank(axis=1, pct=True)
    momlv = (mom_rank + lv_rank) / 2

    spy_ret = rets["SPY"].fillna(0.0)
    spy_px = prices["SPY"]
    sma200 = spy_px.rolling(200).mean()
    gate = (spy_px > sma200).shift(1).fillna(False)
    spy200 = spy_ret.where(gate, 0.0)

    strategies = {
        "MOM_top_decile": portfolio_returns(decile_weights(mom, eligible, 21, True, 260), rets),
        "LOWVOL_bottom_decile": portfolio_returns(
            decile_weights(-vol, eligible, 21, True, 260), rets
        ),
        "REV5_losers_weekly": portfolio_returns(
            decile_weights(-rev5, eligible, 5, True, 260), rets
        ),
        "MOMLV_combined": portfolio_returns(decile_weights(momlv, eligible, 21, True, 260), rets),
        "SPY_buy_hold": spy_ret,
        "SPY_200dma_gate": spy200,
    }
    mom_short = portfolio_returns(decile_weights(-mom, eligible, 21, True, 260), rets)
    ls = strategies["MOM_top_decile"] - mom_short  # HYPOTHETICAL: no borrow data
    strategies["MOM_LS_hypothetical"] = ls

    out: dict = {}
    for period, (lo, hi) in {
        "research_2012_2023": ("2012-06-01", "2023-12-31"),
        "HOLDOUT_2024_2026": ("2024-01-01", "2026-12-31"),
    }.items():
        rows = []
        for name, series in strategies.items():
            rows.append(summarize(series, spy_ret, name, lo, hi))
        out[period] = rows
        print(f"\n=== {period} ===")
        print(
            f"{'strategy':<24}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'NWalpha':>9}{'t':>6}{'beta':>6}"
        )
        for row in rows:
            if "cagr" not in row:
                continue
            print(
                f"{row['label']:<24}{row['cagr']:>8.1%}{row['sharpe']:>8.2f}"
                f"{row['maxdd']:>8.1%}{row['nw_alpha_ann']:>9.1%}"
                f"{row['alpha_t']:>6.2f}{row['beta']:>6.2f}"
            )

    # per-year table for the two leaders vs SPY
    yearly = {}
    for name in (
        "MOM_top_decile",
        "MOMLV_combined",
        "LOWVOL_bottom_decile",
        "SPY_buy_hold",
        "SPY_200dma_gate",
    ):
        s = strategies[name]
        yearly[name] = {
            str(y): round(float(np.prod(1 + s.loc[str(y)]) - 1), 4)
            for y in range(2013, 2027)
            if len(s.loc[str(y)]) > 30
        }
    out["yearly"] = yearly
    atomic_write_bytes(
        Path("artifacts") / "cross_sectional.json", json.dumps(out, indent=2).encode()
    )
    print("\nsaved -> artifacts/cross_sectional.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
