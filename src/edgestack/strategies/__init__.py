"""Archived legacy composites; excluded from V2 scoring, promotion, and execution."""

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
