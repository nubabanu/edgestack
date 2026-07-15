"""False-discovery-rate control for batched hypothesis searches.

Implements Benjamini-Hochberg (independent / positively dependent tests) and
Benjamini-Yekutieli (arbitrary dependence, more conservative). Adjusted
p-values ("q-values") are the smallest FDR level at which each test would be
rejected — enforced monotone as in the step-up procedure.
"""

from __future__ import annotations

import numpy as np

from edgestack.exceptions import ValidationError


def _validate(pvalues: np.ndarray) -> np.ndarray:
    arr = np.asarray(pvalues, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValidationError("pvalues must be a non-empty 1-d array")
    if np.isnan(arr).any() or (arr < 0).any() or (arr > 1).any():
        raise ValidationError("pvalues must lie in [0, 1] with no NaN")
    return arr


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """BH adjusted p-values (q-values)."""
    arr = _validate(pvalues)
    m = len(arr)
    order = np.argsort(arr)
    ranked = arr[order] * m / (np.arange(m) + 1)
    # Step-up: enforce monotonicity from the largest rank downward.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


def benjamini_yekutieli(pvalues: np.ndarray) -> np.ndarray:
    """BY adjusted p-values: BH scaled by the harmonic factor c(m)."""
    arr = _validate(pvalues)
    m = len(arr)
    c_m = float(np.sum(1.0 / (np.arange(m) + 1)))
    return np.clip(benjamini_hochberg(arr) * c_m, 0.0, 1.0)


def fdr_adjust(pvalues: np.ndarray, method: str = "benjamini_hochberg") -> np.ndarray:
    if method == "benjamini_hochberg":
        return benjamini_hochberg(pvalues)
    if method == "benjamini_yekutieli":
        return benjamini_yekutieli(pvalues)
    raise ValidationError(f"unknown FDR method: {method}")


def fdr_reject(
    pvalues: np.ndarray, alpha: float = 0.05, method: str = "benjamini_hochberg"
) -> tuple[np.ndarray, np.ndarray]:
    """Return (reject_mask, q_values) at FDR level ``alpha``."""
    q = fdr_adjust(pvalues, method)
    return q <= alpha, q
