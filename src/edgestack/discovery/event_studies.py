"""Event-study statistics for a set of conditional trade outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edgestack.validation.bayes import normal_credible_interval, prob_mean_positive
from edgestack.validation.bootstrap import block_bootstrap_ci, mean_pvalue_bootstrap
from edgestack.validation.metrics import (
    effective_sample_size,
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    subperiod_sign_consistency,
    var_es,
)

MIN_OBS = 12


@dataclass(frozen=True)
class ReturnStudy:
    n: int
    effective_n: float
    mean: float
    median: float
    std: float
    downside_deviation: float
    hit_rate: float
    p_value: float
    ci_low: float
    ci_high: float
    posterior_prob_positive: float
    credible_low: float
    credible_high: float
    value_at_risk: float
    expected_shortfall: float
    max_drawdown: float
    sharpe: float
    sortino: float
    profit_factor: float
    win_loss_ratio: float
    subperiod_consistency: float


def study_returns(
    returns: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int = 1000,
    block_length: int = 10,
    sessions_per_trade: int = 1,
    prior_pseudo_n: float = 30.0,
) -> ReturnStudy | None:
    """Full distribution statistics for a sequence of (net) trade returns.

    Returns None when there are too few observations to say anything —
    an explicit "not enough evidence", never a fabricated statistic.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < MIN_OBS:
        return None

    wins = arr[arr > 0]
    losses = arr[arr < 0]
    win_loss = float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else 0.0
    var, es = var_es(arr)
    ci_low, ci_high = block_bootstrap_ci(
        arr, rng=rng, block_length=block_length, n_boot=n_boot
    )
    cred_low, cred_high = normal_credible_interval(arr, prior_pseudo_n=prior_pseudo_n)
    downside = arr[arr < 0]
    dd_dev = float(np.sqrt(np.mean(downside**2))) if len(downside) else 0.0
    periods_per_year = max(1, 252 // max(1, sessions_per_trade))

    return ReturnStudy(
        n=len(arr),
        effective_n=effective_sample_size(arr),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        std=float(arr.std(ddof=1)),
        downside_deviation=dd_dev,
        hit_rate=hit_rate(arr),
        p_value=mean_pvalue_bootstrap(arr, rng=rng, n_boot=n_boot),
        ci_low=ci_low,
        ci_high=ci_high,
        posterior_prob_positive=prob_mean_positive(arr, prior_pseudo_n=prior_pseudo_n),
        credible_low=cred_low,
        credible_high=cred_high,
        value_at_risk=var,
        expected_shortfall=es,
        max_drawdown=max_drawdown(arr),
        sharpe=sharpe_ratio(arr, periods_per_year),
        sortino=sortino_ratio(arr, periods_per_year),
        profit_factor=profit_factor(arr),
        win_loss_ratio=win_loss,
        subperiod_consistency=subperiod_sign_consistency(arr),
    )
