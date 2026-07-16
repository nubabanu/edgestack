"""Validated composite strategies distilled from the research campaigns."""

from edgestack.strategies.composite import (
    ENSEMBLE_FAMILIES,
    backtest_exposure,
    ensemble_exposure,
    family_positions,
    seasonal_multiplier,
)

__all__ = [
    "ENSEMBLE_FAMILIES",
    "backtest_exposure",
    "ensemble_exposure",
    "family_positions",
    "seasonal_multiplier",
]
