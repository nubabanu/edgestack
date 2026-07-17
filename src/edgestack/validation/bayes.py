"""Bayesian shrinkage for win rates and mean returns.

Small samples produce exaggerated statistics; conservative priors pull them
toward "no edge" until the evidence earns its way out.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from edgestack.exceptions import ValidationError


def beta_binomial_posterior(
    wins: int, losses: int, prior_a: float = 1.0, prior_b: float = 1.0
) -> tuple[float, float]:
    """Posterior Beta(a, b) parameters for a win rate."""
    if wins < 0 or losses < 0:
        raise ValidationError("wins/losses must be non-negative")
    return prior_a + wins, prior_b + losses


def prob_win_rate_above(
    wins: int,
    losses: int,
    threshold: float = 0.5,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
) -> float:
    """P(true win rate > threshold | data) under a Beta prior."""
    a, b = beta_binomial_posterior(wins, losses, prior_a, prior_b)
    return float(1.0 - sps.beta.cdf(threshold, a, b))


def beta_credible_interval(
    wins: int,
    losses: int,
    alpha: float = 0.05,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
) -> tuple[float, float]:
    a, b = beta_binomial_posterior(wins, losses, prior_a, prior_b)
    lo, hi = sps.beta.ppf([alpha / 2, 1 - alpha / 2], a, b)
    return float(lo), float(hi)


def normal_posterior_mean(
    values: np.ndarray,
    prior_mean: float = 0.0,
    prior_pseudo_n: float = 30.0,
) -> tuple[float, float]:
    """Posterior (mean, sd-of-mean) for the average of ``values``.

    Conjugate normal update with a "no edge" prior worth ``prior_pseudo_n``
    observations: small samples are pulled hard toward ``prior_mean``.
    Observation variance is taken from the sample (plug-in).
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        raise ValidationError("need at least 2 observations")
    n = len(arr)
    obs_var = float(arr.var(ddof=1))
    if obs_var == 0:
        obs_var = 1e-12
    post_mean = (prior_pseudo_n * prior_mean + n * float(arr.mean())) / (prior_pseudo_n + n)
    post_var_of_mean = obs_var / (prior_pseudo_n + n)
    return post_mean, float(np.sqrt(post_var_of_mean))


def prob_mean_positive(
    values: np.ndarray, prior_mean: float = 0.0, prior_pseudo_n: float = 30.0
) -> float:
    """P(true mean > 0 | data) under the shrinkage posterior."""
    mean, sd = normal_posterior_mean(values, prior_mean, prior_pseudo_n)
    if sd == 0:
        return 1.0 if mean > 0 else 0.0
    return float(1.0 - sps.norm.cdf(0.0, loc=mean, scale=sd))


def normal_credible_interval(
    values: np.ndarray,
    alpha: float = 0.05,
    prior_mean: float = 0.0,
    prior_pseudo_n: float = 30.0,
) -> tuple[float, float]:
    mean, sd = normal_posterior_mean(values, prior_mean, prior_pseudo_n)
    lo, hi = sps.norm.ppf([alpha / 2, 1 - alpha / 2], loc=mean, scale=sd)
    return float(lo), float(hi)
