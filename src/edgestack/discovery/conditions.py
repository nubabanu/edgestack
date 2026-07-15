"""Evaluate rule-AST conditions against feature frames.

Quantile predicates resolve their thresholds through a train-fitted
:class:`QuantileBinner` — never against the frame being evaluated — so the
same rule definition is re-evaluable in any period without leakage.
NaN feature values never satisfy a predicate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgestack.exceptions import LeakageError, ValidationError
from edgestack.features.binning import QuantileBinner
from edgestack.types import AllOf, AnyOf, Condition, Predicate

_OPS = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
    "==": np.equal,
    "!=": np.not_equal,
}


def evaluate_condition(
    cond: Condition, df: pd.DataFrame, binner: QuantileBinner | None = None
) -> pd.Series:
    """Boolean mask of rows satisfying ``cond``."""
    if isinstance(cond, Predicate):
        if cond.feature not in df.columns:
            raise ValidationError(f"condition references unknown feature {cond.feature!r}")
        values = df[cond.feature]
        if cond.quantile is not None:
            if binner is None:
                raise LeakageError(
                    f"quantile predicate on {cond.feature} requires a train-fitted binner"
                )
            threshold: float | int | bool | str = binner.threshold(cond.feature, cond.quantile)
        else:
            threshold = cond.value  # type: ignore[assignment]
        mask = _OPS[cond.op](values, threshold)
        return pd.Series(mask, index=df.index).fillna(False).astype(bool)
    if isinstance(cond, AllOf):
        out = pd.Series(True, index=df.index)
        for sub in cond.conditions:
            out &= evaluate_condition(sub, df, binner)
        return out
    if isinstance(cond, AnyOf):
        out = pd.Series(False, index=df.index)
        for sub in cond.conditions:
            out |= evaluate_condition(sub, df, binner)
        return out
    raise ValidationError(f"unknown condition node: {type(cond)!r}")
