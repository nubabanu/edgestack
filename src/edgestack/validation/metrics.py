"""Performance and reliability metrics.

Includes the probabilistic and deflated Sharpe ratio (Bailey & López de
Prado): the DSR discounts an observed Sharpe for non-normality AND for the
number of trials searched, using the expected maximum Sharpe among N
independent noise trials as the null benchmark.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from edgestack.exceptions import ValidationError

EULER_GAMMA = 0.5772156649015329
SESSIONS_PER_YEAR = 252


def _clean(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        raise ValidationError("need at least 2 observations")
    return arr


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = SESSIONS_PER_YEAR) -> float:
    arr = _clean(returns)
    sd = arr.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(arr.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, periods_per_year: int = SESSIONS_PER_YEAR) -> float:
    arr = _clean(returns)
    downside = arr[arr < 0]
    if len(downside) == 0:
        return float("inf") if arr.mean() > 0 else 0.0
    dd = np.sqrt(np.mean(downside**2))
    if dd == 0:
        return 0.0
    return float(arr.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of the compounded return path (<= 0)."""
    arr = _clean(returns)
    equity = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def var_es(returns: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Historical value-at-risk and expected shortfall at ``level`` (as losses <= 0)."""
    arr = _clean(returns)
    var = float(np.quantile(arr, 1.0 - level))
    tail = arr[arr <= var]
    es = float(tail.mean()) if len(tail) else var
    return var, es


def hit_rate(returns: np.ndarray) -> float:
    arr = _clean(returns)
    return float((arr > 0).mean())


def profit_factor(returns: np.ndarray) -> float:
    arr = _clean(returns)
    gains = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def effective_sample_size(values: np.ndarray, max_lag: int | None = None) -> float:
    """n / (1 + 2 * sum of positive autocorrelations), truncated at first negative.

    Overlapping or serially correlated observations carry less information
    than their raw count; shrinkage and gates use this instead of n.
    """
    arr = _clean(values)
    n = len(arr)
    centered = arr - arr.mean()
    denom = float(centered @ centered)
    if denom == 0:
        return float(n)
    max_lag = max_lag or min(n // 4, 100)
    rho_sum = 0.0
    for lag in range(1, max_lag + 1):
        rho = float(centered[lag:] @ centered[:-lag]) / denom
        if rho <= 0:
            break
        rho_sum += rho
    return float(n / (1.0 + 2.0 * rho_sum))


def subperiod_sign_consistency(values: np.ndarray, n_periods: int = 4) -> float:
    """Fraction of contiguous subperiods whose mean has the full-sample sign."""
    arr = _clean(values)
    overall = np.sign(arr.mean())
    if overall == 0:
        return 0.0
    chunks = np.array_split(arr, n_periods)
    same = sum(1 for c in chunks if len(c) and np.sign(c.mean()) == overall)
    return same / n_periods


# ---------------------------------------------------------------------------
# Probabilistic / deflated Sharpe
# ---------------------------------------------------------------------------


def probabilistic_sharpe_ratio(
    observed_sr: float, benchmark_sr: float, n: int, skew: float, kurt: float
) -> float:
    """P(true SR > benchmark_sr) accounting for non-normal returns.

    ``observed_sr`` and ``benchmark_sr`` are per-period (NOT annualized);
    ``kurt`` is ordinary kurtosis (normal = 3).
    """
    if n < 2:
        raise ValidationError("need n >= 2 for PSR")
    denom = np.sqrt(1.0 - skew * observed_sr + (kurt - 1.0) / 4.0 * observed_sr**2)
    if not np.isfinite(denom) or denom <= 0:
        return 0.0
    z = (observed_sr - benchmark_sr) * np.sqrt(n - 1.0) / denom
    return float(sps.norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """E[max SR] among ``n_trials`` zero-skill trials with SR variance ``var_sharpe``."""
    if n_trials < 1:
        raise ValidationError("n_trials must be >= 1")
    if n_trials == 1 or var_sharpe <= 0:
        return 0.0
    sd = np.sqrt(var_sharpe)
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sd * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    var_sharpe_across_trials: float | None = None,
) -> float:
    """DSR: P(true SR > 0) after deflating for the search over ``n_trials``.

    When the cross-trial SR variance is unknown we approximate it with the
    estimator variance of a zero-mean SR over this sample length, ``1/n`` —
    a documented approximation, conservative for long samples.
    """
    arr = _clean(returns)
    n = len(arr)
    sd = arr.std(ddof=1)
    if sd == 0:
        return 0.0
    sr = float(arr.mean() / sd)  # per-period SR
    skew = float(sps.skew(arr))
    kurt = float(sps.kurtosis(arr, fisher=False))
    var_sr = var_sharpe_across_trials if var_sharpe_across_trials is not None else 1.0 / n
    sr0 = expected_max_sharpe(max(1, n_trials), var_sr)
    return probabilistic_sharpe_ratio(sr, sr0, n, skew, kurt)
