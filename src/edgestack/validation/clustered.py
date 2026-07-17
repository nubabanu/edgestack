"""Session-clustered portfolio inference and deterministic stationary bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgestack.exceptions import ValidationError
from edgestack.validation.metrics import effective_sample_size


def aggregate_session_contributions(
    contributions: pd.DataFrame,
    *,
    date_col: str = "date",
    return_col: str = "net_return_contribution",
) -> pd.Series:
    """Collapse concurrent trades to one portfolio contribution per session."""
    if date_col not in contributions or return_col not in contributions:
        raise ValidationError(f"contributions require {date_col!r} and {return_col!r}")
    frame = contributions[[date_col, return_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame[return_col] = pd.to_numeric(frame[return_col], errors="coerce")
    frame = frame.dropna()
    return frame.groupby(date_col, sort=True)[return_col].sum().astype(float)


def session_effective_sample_size(returns: pd.Series) -> float:
    return effective_sample_size(returns.dropna().to_numpy(dtype=float))


def stationary_bootstrap_indices(
    n: int,
    *,
    n_boot: int = 2_000,
    mean_block_length: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Politis-Romano stationary-bootstrap index matrix."""
    if n < 3:
        raise ValidationError("stationary bootstrap needs at least 3 sessions")
    if mean_block_length < 1 or n_boot < 1:
        raise ValidationError("bootstrap parameters must be positive")
    rng = np.random.default_rng(seed)
    restart_probability = 1.0 / min(mean_block_length, n)
    out = np.empty((n_boot, n), dtype=int)
    out[:, 0] = rng.integers(0, n, size=n_boot)
    restart = rng.random((n_boot, n - 1)) < restart_probability
    fresh = rng.integers(0, n, size=(n_boot, n - 1))
    for t in range(1, n):
        continued = (out[:, t - 1] + 1) % n
        out[:, t] = np.where(restart[:, t - 1], fresh[:, t - 1], continued)
    return out


def stationary_mean_test(
    returns: pd.Series,
    *,
    n_boot: int = 2_000,
    mean_block_length: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    values = returns.dropna().to_numpy(dtype=float)
    indices = stationary_bootstrap_indices(
        len(values), n_boot=n_boot, mean_block_length=mean_block_length, seed=seed
    )
    observed = float(values.mean())
    centered = values - observed
    null_means = centered[indices].mean(axis=1)
    pvalue = (float((null_means >= observed).sum()) + 1.0) / (n_boot + 1.0)
    sampled = values[indices].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "mean": observed,
        "pvalue_greater": pvalue,
        "ci_low": float(low),
        "ci_high": float(high),
        "session_ess": session_effective_sample_size(returns),
    }


def paired_sharpe_improvement(
    strategy: pd.Series,
    benchmark: pd.Series,
    *,
    n_boot: int = 2_000,
    mean_block_length: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    frame = pd.concat([strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
    if len(frame) < 60:
        raise ValidationError("paired Sharpe inference needs at least 60 sessions")
    values = frame.to_numpy(dtype=float)

    def sharpe(x: np.ndarray) -> np.ndarray:
        means = x.mean(axis=1)
        std = x.std(axis=1, ddof=1)
        return np.divide(means, std, out=np.zeros_like(means), where=std > 0) * np.sqrt(252)

    idx = stationary_bootstrap_indices(
        len(values), n_boot=n_boot, mean_block_length=mean_block_length, seed=seed
    )
    sampled_strategy = values[idx, 0]
    sampled_benchmark = values[idx, 1]
    differences = sharpe(sampled_strategy) - sharpe(sampled_benchmark)
    observed = float(
        frame["strategy"].mean() / frame["strategy"].std(ddof=1) * np.sqrt(252)
        - frame["benchmark"].mean() / frame["benchmark"].std(ddof=1) * np.sqrt(252)
    )
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "delta_sharpe": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "pvalue_one_sided": (float((differences <= 0).sum()) + 1.0) / (n_boot + 1.0),
        "n_sessions": float(len(frame)),
    }
