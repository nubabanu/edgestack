"""Explainable Boosting Machine stage: effect shapes and feature shortlist.

The EBM is DIAGNOSTIC ONLY (per the brief): it identifies which features carry
stable signal for the excess-net target, where thresholds live, and which
pairwise interactions matter — the shortlist then constrains RuleFit's search.
It never emits a trading rule itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgestack.exceptions import ValidationError
from edgestack.logging import get_logger, log_event

log = get_logger("ebm")

MAX_FIT_ROWS = 150_000


def ebm_feature_report(
    features: pd.DataFrame,
    target: pd.Series,
    feature_cols: tuple[str, ...],
    *,
    seed: int = 42,
    max_interactions: int = 10,
    shortlist_fraction: float = 0.05,
) -> dict:
    """Fit an EBM regressor and report importances, shapes and a shortlist.

    ``shortlist_fraction``: keep features whose importance exceeds this
    fraction of the maximum term importance.
    """
    try:
        from interpret.glassbox import ExplainableBoostingRegressor
    except ImportError as exc:  # pragma: no cover - extra not installed
        raise ValidationError(
            "interpret is not installed; pip install edgestack[rules]"
        ) from exc

    frame = features[list(feature_cols)].copy()
    y = target.to_numpy(dtype=float)
    mask = np.isfinite(y) & frame.notna().any(axis=1).to_numpy()
    frame, y = frame.loc[mask], y[mask]
    if len(frame) < 1000:
        raise ValidationError("need >=1000 rows for the EBM stage")
    if len(frame) > MAX_FIT_ROWS:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(frame), size=MAX_FIT_ROWS, replace=False)
        frame, y = frame.iloc[idx], y[idx]

    ebm = ExplainableBoostingRegressor(
        interactions=max_interactions, random_state=seed, n_jobs=-1,
    )
    ebm.fit(frame, y)

    names = list(ebm.term_names_)
    importances = [float(v) for v in ebm.term_importances()]
    singles = {n: imp for n, imp in zip(names, importances) if " & " not in n}
    pairs = {n: imp for n, imp in zip(names, importances) if " & " in n}
    max_imp = max(importances) if importances else 0.0
    shortlist = sorted(
        (n for n, imp in singles.items() if imp >= shortlist_fraction * max_imp),
        key=lambda n: -singles[n],
    )
    log_event(log, 20, "ebm fitted", rows=len(frame),
              shortlist=len(shortlist), interactions=len(pairs))
    return {
        "n_rows": len(frame),
        "feature_importance": dict(sorted(singles.items(), key=lambda kv: -kv[1])),
        "top_interactions": dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:max_interactions]),
        "shortlist": shortlist,
    }
