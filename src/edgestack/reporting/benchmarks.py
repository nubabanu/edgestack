"""Benchmark and alpha evaluation framework (roadmap phase 2).

A strategy must never be judged on raw cumulative return. This module builds
the comparison set — market total return, exposure-matched market, equal-
weight point-in-time universe, random-entry matched benchmarks — and the
alpha regression with block-bootstrap confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from edgestack.validation.bootstrap import block_bootstrap_ci
from edgestack.validation.metrics import max_drawdown, sharpe_ratio

SESSIONS = 252


def daily_total_returns(panel: pd.DataFrame, symbol: str) -> pd.Series:
    """Total-return daily series for one symbol (adj_close, close fallback)."""
    bars = panel.loc[panel["symbol"] == symbol].sort_values("date")
    px = bars["adj_close"].where(bars["adj_close"].notna(), bars["close"])
    out = px.pct_change()
    out.index = pd.to_datetime(bars["date"])
    return out.dropna()


def equal_weight_pit_returns(panel: pd.DataFrame,
                             member_mask: pd.Series) -> pd.Series:
    """Equal-weighted daily total return of point-in-time-eligible rows."""
    df = panel.loc[member_mask.to_numpy(), ["symbol", "date", "adj_close", "close"]].copy()
    df["px"] = df["adj_close"].where(df["adj_close"].notna(), df["close"])
    df = df.sort_values(["symbol", "date"])
    df["ret"] = df.groupby("symbol")["px"].pct_change()
    out = df.groupby("date")["ret"].mean().dropna()
    out.index = pd.to_datetime(out.index)
    return out


@dataclass(frozen=True)
class AlphaReport:
    beta: float
    annualized_alpha: float
    alpha_ci: tuple[float, float]
    correlation: float
    n_days: int


def alpha_regression(strategy: pd.Series, market: pd.Series,
                     *, rng: np.random.Generator, n_boot: int = 500) -> AlphaReport:
    """OLS of strategy on market daily returns; alpha CI via paired block bootstrap."""
    joined = pd.concat([strategy.rename("s"), market.rename("m")], axis=1).dropna()
    s, m = joined["s"].to_numpy(), joined["m"].to_numpy()
    if len(s) < 60:
        return AlphaReport(float("nan"), float("nan"),
                           (float("nan"), float("nan")), float("nan"), len(s))

    def fit_alpha(idx: np.ndarray) -> tuple[float, float]:
        mm, ss = m[idx], s[idx]
        var = mm.var()
        beta = float(np.cov(ss, mm)[0, 1] / var) if var > 0 else 0.0
        alpha = float(ss.mean() - beta * mm.mean())
        return alpha, beta

    n = len(s)
    alpha, beta = fit_alpha(np.arange(n))
    block, n_blocks = 20, int(np.ceil(n / 20))
    alphas = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = ((starts[:, None] + np.arange(block)) % n).reshape(-1)[:n]
        alphas[b] = fit_alpha(idx)[0]
    lo, hi = np.quantile(alphas, [0.025, 0.975])
    return AlphaReport(
        beta=beta,
        annualized_alpha=alpha * SESSIONS,
        alpha_ci=(float(lo) * SESSIONS, float(hi) * SESSIONS),
        correlation=float(np.corrcoef(s, m)[0, 1]),
        n_days=n,
    )


def summarize_returns(rets: pd.Series | np.ndarray, *,
                      rng: np.random.Generator, label: str) -> dict:
    arr = np.asarray(rets, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 30:
        return {"label": label, "n_days": len(arr)}
    sr_ci = block_bootstrap_ci(
        arr, rng=rng, block_length=20, n_boot=500,
        stat=lambda x: float(x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else 0.0,
    )
    ann = np.sqrt(SESSIONS)
    years = len(arr) / SESSIONS
    cum = float(np.prod(1 + arr) - 1)
    return {
        "label": label,
        "n_days": len(arr),
        "cum_return": cum,
        "cagr": float((1 + cum) ** (1 / max(years, 1e-9)) - 1),
        "ann_vol": float(arr.std(ddof=1) * ann),
        "sharpe": sharpe_ratio(arr),
        "sharpe_ci_ann": (float(sr_ci[0] * ann), float(sr_ci[1] * ann)),
        "max_drawdown": max_drawdown(arr),
    }


def concentration_diagnostics(trades: pd.DataFrame) -> dict:
    """Profit concentration: by symbol, by month, and excluding the best of each."""
    if trades.empty:
        return {}
    total = float(trades["net_pnl"].sum())
    by_symbol = trades.groupby("symbol")["net_pnl"].sum().sort_values(ascending=False)
    months = pd.to_datetime(trades["exit_session"]).dt.to_period("M")
    by_month = trades.groupby(months)["net_pnl"].sum().sort_values(ascending=False)
    best10 = trades["net_pnl"].nlargest(10).sum()
    return {
        "total_net_pnl": total,
        "n_unique_symbols": int(trades["symbol"].nunique()),
        "best_symbol": str(by_symbol.index[0]),
        "best_symbol_share": float(by_symbol.iloc[0] / total) if total else None,
        "pnl_excl_best_symbol": float(total - by_symbol.iloc[0]),
        "best_month": str(by_month.index[0]),
        "pnl_excl_best_month": float(total - by_month.iloc[0]),
        "pnl_excl_best_10_trades": float(total - best10),
    }
