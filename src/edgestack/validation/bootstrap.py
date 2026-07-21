"""Bootstrap machinery: IID and circular-block resampling, CIs and p-values.

Pure numpy, explicit ``rng`` everywhere — results are reproducible per seed.
Block bootstrap preserves short-range serial dependence that IID resampling
would destroy, which matters for overlapping-return statistics.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from edgestack.exceptions import ValidationError

Stat = Callable[[np.ndarray], float]


def _check(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        raise ValidationError(f"need at least 3 observations, got {len(arr)}")
    return arr


def bootstrap_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    stat: Stat = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ``stat`` (IID resampling)."""
    arr = _check(values)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    stats = np.apply_along_axis(stat, 1, arr[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def block_bootstrap_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    block_length: int = 20,
    stat: Stat = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Circular block bootstrap CI, preserving serial dependence within blocks.

    Fully vectorized: an index matrix of shape (n_boot, n) gathers all
    resamples at once; blocks wrap circularly via modulo indexing.
    """
    arr = _check(values)
    n = len(arr)
    block_length = max(1, min(block_length, n))
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block_length)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    samples = arr[idx.reshape(n_boot, -1)[:, :n]]
    if stat is np.mean:
        stats = samples.mean(axis=1)
    else:
        stats = np.apply_along_axis(stat, 1, samples)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def suggested_block_length(values: np.ndarray, *, fallback: int = 20) -> dict:
    """Politis-White optimal block length — a REPORT-ONLY diagnostic.

    Wraps ``arch.bootstrap.optimal_block_length`` (lazy import). Never feeds
    the block length actually used by validation — the configured
    ``block_length_sessions`` stays canonical; this exists so logs can show
    how far the fixed choice sits from the data-driven estimate. Falls back
    (``source="fallback"``) when arch is unavailable, the cleaned series is
    shorter than 100 observations (the estimator is unstable there), or the
    estimate comes back non-finite or below 1.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    result = {
        "stationary": float(fallback),
        "circular": float(fallback),
        "configured": float(fallback),
        "source": "fallback",
        "n": len(arr),
    }
    if len(arr) < 100:
        return result
    try:
        from arch.bootstrap import optimal_block_length
    except ImportError:
        return result
    est = optimal_block_length(arr)
    stationary = float(est["stationary"].iloc[0])
    circular = float(est["circular"].iloc[0])
    if not (np.isfinite(stationary) and np.isfinite(circular)) or stationary < 1 or circular < 1:
        return result
    result.update(
        {
            "stationary": max(1.0, stationary),
            "circular": max(1.0, circular),
            "source": "politis_white",
        }
    )
    return result


def mean_pvalue_bootstrap(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int = 2000,
    alternative: str = "greater",
) -> float:
    """Bootstrap p-value for H0: mean == 0 via null-centered resampling.

    The sample is shifted to have mean zero (the null), resampled, and the
    p-value is the fraction of resampled means at least as extreme as the
    observed one. A +1 correction keeps p away from an impossible zero.
    """
    arr = _check(values)
    observed = float(arr.mean())
    centered = arr - observed
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    null_means = centered[idx].mean(axis=1)
    if alternative == "greater":
        extreme = int((null_means >= observed).sum())
    elif alternative == "less":
        extreme = int((null_means <= observed).sum())
    elif alternative == "two-sided":
        extreme = int((np.abs(null_means) >= abs(observed)).sum())
    else:
        raise ValidationError(f"unknown alternative: {alternative}")
    return (extreme + 1) / (n_boot + 1)
