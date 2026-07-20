"""Calendar-time log-growth comparisons and conservative Kelly sizing."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from edgestack.exceptions import ValidationError
from edgestack.validation.clustered import stationary_bootstrap_indices

REQUIRED_GROWTH_COMPARATORS = (
    "buy_now_spy",
    "risk_matched_spy",
    "diversified_baseline",
    "financed_spy_same_risk",
)


def calendar_log_returns(returns: pd.Series) -> pd.Series:
    """Convert a complete calendar return stream, including waiting days, to log returns."""
    values = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise ValidationError("log-growth estimation requires returns")
    if bool((values <= -1.0).any()):
        raise ValidationError("log growth is undefined after a total-loss return")
    return pd.Series(np.log1p(values.to_numpy()), index=values.index, name=values.name)


def annualized_log_growth(returns: pd.Series, *, periods_per_year: int = 252) -> float:
    values = calendar_log_returns(returns)
    return float(values.mean() * periods_per_year)


def log_growth_superiority(
    strategy: pd.Series,
    comparators: Mapping[str, pd.Series],
    *,
    n_boot: int = 1_000,
    seed: int = 42,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Return 95% lower bounds for paired annualized log-growth improvement."""
    result: dict[str, float] = {}
    for offset, name in enumerate(REQUIRED_GROWTH_COMPARATORS):
        if name not in comparators:
            raise ValidationError(f"missing growth comparator {name}")
        joined = pd.concat(
            [strategy.rename("strategy"), comparators[name].rename("benchmark")], axis=1
        ).dropna()
        if len(joined) < 60:
            raise ValidationError(f"growth comparator {name} has fewer than 60 paired sessions")
        difference = (
            calendar_log_returns(joined["strategy"]).to_numpy()
            - calendar_log_returns(joined["benchmark"]).to_numpy()
        )
        indices = stationary_bootstrap_indices(
            len(difference),
            n_boot=n_boot,
            mean_block_length=min(20, max(2, len(difference) // 10)),
            seed=seed + offset,
        )
        boot = difference[indices].mean(axis=1) * periods_per_year
        result[name] = float(np.quantile(boot, 0.05))
    return result


def financed_risk_matched_spy(
    spy_returns: pd.Series,
    *,
    target_volatility: float,
    financing_rate: float,
    cash_yield: float,
    maximum_leverage: float = 5.0,
) -> pd.Series:
    """Construct SPY at the same realized risk with financing and cash carry."""
    realized = float(spy_returns.dropna().std(ddof=1) * math.sqrt(252))
    leverage = min(maximum_leverage, target_volatility / realized) if realized > 0 else 0.0
    borrowed = max(0.0, leverage - 1.0)
    idle_cash = max(0.0, 1.0 - leverage)
    daily_financing = financing_rate / 252.0
    daily_cash = cash_yield / 252.0
    return leverage * spy_returns - borrowed * daily_financing + idle_cash * daily_cash


def shrinkage_quarter_kelly_limit(
    returns: pd.Series,
    *,
    stressed_annual_variance: float,
    maximum_leverage: float = 5.0,
    prior_observations: float = 252.0,
) -> float:
    """Use a zero-mean shrinkage prior and a lower-confidence excess return."""
    values = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(values) < 60 or stressed_annual_variance <= 0:
        return 0.0
    sample_mean = float(values.mean())
    shrinkage = len(values) / (len(values) + prior_observations)
    conservative_annual_return = max(0.0, shrinkage * sample_mean * 252.0)
    raw = 0.25 * conservative_annual_return / stressed_annual_variance
    return float(min(maximum_leverage, max(0.0, raw)))
