"""Advanced selection-aware inference (references: docs/references.md).

- Hansen (2005) SPA / White (2000) Reality Check via `arch` — does the best of
  a *searched set* of strategies genuinely beat a benchmark once snooping over
  the whole set is priced in?
- Romano-Wolf (2005) StepM — which individual rules are superior, with
  familywise control under dependence.
- Ledoit-Wolf (2008)-style robust Sharpe-difference inference via paired
  circular block bootstrap (no i.i.d.-normal assumption).
- Newey-West (1987) HAC standard errors for the alpha regression.
- Chordia-Goyal-Saretto (2020) empirical t-hurdles as acceptance constants.

All functions take explicit seeds. `arch` works on losses, so returns are
negated internally — callers always pass RETURNS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgestack.exceptions import ValidationError

#: Chordia, Goyal & Saretto (2020) suggested hurdles.
CGS_HURDLE_TIME_SERIES = 3.8
CGS_HURDLE_CROSS_SECTION = 3.4


def _align(*series: pd.Series) -> pd.DataFrame:
    df = pd.concat(list(series), axis=1).dropna()
    if len(df) < 60:
        raise ValidationError(f"need >=60 aligned observations, got {len(df)}")
    return df


def spa_test(
    benchmark_returns: pd.Series,
    model_returns: pd.DataFrame,
    *,
    reps: int = 1000,
    block_size: int = 20,
    seed: int = 42,
) -> dict:
    """Hansen SPA: H0 = no model in the set beats the benchmark.

    A small consistent p-value means the best searched strategy outperforms
    the benchmark even after paying for the whole search.
    """
    from arch.bootstrap import SPA

    joined = pd.concat([benchmark_returns.rename("__bench__"), model_returns], axis=1).dropna()
    if len(joined) < 60:
        raise ValidationError("need >=60 aligned observations for SPA")
    bench_losses = -joined["__bench__"].to_numpy()
    model_losses = -joined.drop(columns="__bench__").to_numpy()
    spa = SPA(
        bench_losses,
        model_losses,
        reps=reps,
        block_size=block_size,
        bootstrap="stationary",
        seed=seed,
    )
    spa.compute()
    mean_excess = joined.drop(columns="__bench__").mean() - joined["__bench__"].mean()
    return {
        "p_lower": float(spa.pvalues["lower"]),
        "p_consistent": float(spa.pvalues["consistent"]),
        "p_upper": float(spa.pvalues["upper"]),
        "best_model": str(mean_excess.idxmax()),
        "best_mean_daily_excess": float(mean_excess.max()),
        "n_models": int(model_losses.shape[1]),
        "n_days": len(joined),
    }


def stepm_superior(
    benchmark_returns: pd.Series,
    model_returns: pd.DataFrame,
    *,
    reps: int = 1000,
    block_size: int = 20,
    size: float = 0.05,
    seed: int = 42,
) -> list[str]:
    """Romano-Wolf StepM: names of models genuinely superior to the benchmark."""
    from arch.bootstrap import StepM

    joined = pd.concat([benchmark_returns.rename("__bench__"), model_returns], axis=1).dropna()
    if len(joined) < 60:
        raise ValidationError("need >=60 aligned observations for StepM")
    stepm = StepM(
        -joined["__bench__"].to_numpy(),
        -joined.drop(columns="__bench__"),
        size=size,
        reps=reps,
        block_size=block_size,
        bootstrap="stationary",
        seed=seed,
    )
    stepm.compute()
    return [str(name) for name in stepm.superior_models]


def sharpe_difference_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
    *,
    n_boot: int = 2000,
    block_size: int = 20,
    seed: int = 42,
) -> dict:
    """Ledoit-Wolf-style robust test of H0: Sharpe(a) == Sharpe(b).

    Paired circular block bootstrap of the Sharpe difference — respects serial
    dependence and fat tails; no normality assumed.
    """
    df = _align(returns_a.rename("a"), returns_b.rename("b"))
    a, b = df["a"].to_numpy(), df["b"].to_numpy()

    def sharpe(x: np.ndarray) -> float:
        sd = x.std(ddof=1)
        return float(x.mean() / sd) if sd > 0 else 0.0

    observed = sharpe(a) - sharpe(b)
    n = len(a)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = ((starts[:, None] + np.arange(block_size)) % n).reshape(-1)[:n]
        diffs[i] = sharpe(a[idx]) - sharpe(b[idx])
    centered = diffs - diffs.mean()
    p = float((np.abs(centered) >= abs(observed)).mean())
    return {
        "sharpe_a_ann": sharpe(a) * float(np.sqrt(252)),
        "sharpe_b_ann": sharpe(b) * float(np.sqrt(252)),
        "delta_sharpe_ann": observed * float(np.sqrt(252)),
        "p_two_sided": max(p, 1.0 / n_boot),
        "n_days": n,
    }


def newey_west_alpha(
    strategy: pd.Series,
    market: pd.Series,
    *,
    lags: int = 10,
) -> dict:
    """CAPM alpha with Newey-West HAC standard errors (cross-checks the
    block-bootstrap alpha CI in reporting/benchmarks)."""
    df = _align(strategy.rename("s"), market.rename("m"))
    y = df["s"].to_numpy()
    x = np.column_stack([np.ones(len(df)), df["m"].to_numpy()])
    xtx_inv = np.linalg.inv(x.T @ x)
    beta_hat = xtx_inv @ x.T @ y
    resid = y - x @ beta_hat

    # Long-run covariance of the score x_t * u_t with Bartlett weights.
    g = x * resid[:, None]
    s = g.T @ g
    n = len(y)
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = g[lag:].T @ g[:-lag]
        s += w * (gamma + gamma.T)
    cov = xtx_inv @ s @ xtx_inv
    alpha_se = float(np.sqrt(cov[0, 0]))
    alpha = float(beta_hat[0])
    return {
        "alpha_ann": alpha * 252,
        "alpha_se_ann": alpha_se * 252,
        "alpha_t": alpha / alpha_se if alpha_se > 0 else 0.0,
        "beta": float(beta_hat[1]),
        "lags": lags,
        "n_days": n,
    }


def passes_cgs_hurdle(t_stat: float, *, cross_sectional: bool = True) -> bool:
    """Chordia-Goyal-Saretto multiplicity-calibrated significance hurdle."""
    hurdle = CGS_HURDLE_CROSS_SECTION if cross_sectional else CGS_HURDLE_TIME_SERIES
    return abs(t_stat) >= hurdle
