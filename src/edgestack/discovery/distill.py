"""Distill stable rules into one small, auditable position policy.

Inputs are BINARY indicators (one column per stable structural rule — each of
which is itself a persisted Condition AST), target is the sign of the excess-
net return. Distiller chain, first available wins (recorded in the manifest):

    GOSDT (optimal sparse tree)  ->  imodels GreedyRuleListClassifier
    ->  sklearn DecisionTree(depth<=3, pruned)

CORELS sits between GOSDT and the greedy list when importable (its Windows
build needs GMP and frequently is not). Whatever ran is named in the output —
a verdict must never hide which distiller produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from edgestack.exceptions import ValidationError
from edgestack.logging import get_logger, log_event

log = get_logger("distill")

TIER_NONE, TIER_SMALL, TIER_NORMAL = 0.0, 0.5, 1.0
_P_SMALL, _P_NORMAL = 0.52, 0.58


@dataclass
class PositionPolicy:
    distiller: str
    rule_signatures: tuple[str, ...]
    model: Any
    text: str
    fallback_notes: tuple[str, ...] = field(default_factory=tuple)

    def tier(self, indicators: pd.DataFrame) -> np.ndarray:
        """Map rule-indicator rows to position tiers {0, 0.5, 1}."""
        x = indicators[list(self.rule_signatures)].astype(int).to_numpy()
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(x))
            p = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
        else:
            p = np.asarray(self.model.predict(x), dtype=float)
        tiers = np.full(len(x), TIER_NONE)
        tiers[p >= _P_SMALL] = TIER_SMALL
        tiers[p >= _P_NORMAL] = TIER_NORMAL
        return tiers


def distill_policy(
    indicators: pd.DataFrame,
    excess_returns: np.ndarray,
    *,
    seed: int = 42,
) -> PositionPolicy:
    """Fit the smallest adequate policy over stable-rule indicators."""
    if indicators.empty or indicators.shape[1] == 0:
        raise ValidationError("no stable rules to distill")
    y = (np.asarray(excess_returns, dtype=float) > 0).astype(int)
    ok = np.isfinite(excess_returns)
    x_frame = indicators.loc[ok].astype(int)
    y = y[ok]
    signatures = tuple(indicators.columns)
    notes: list[str] = []

    # --- GOSDT ---------------------------------------------------------------
    try:
        import gosdt as _gosdt_mod

        _Gosdt = getattr(_gosdt_mod, "GOSDTClassifier", None) or getattr(_gosdt_mod, "GOSDT")  # noqa: B009 - older API name
        model = _Gosdt(regularization=0.01, depth_budget=4, time_limit=60)
        model.fit(x_frame, pd.Series(y))
        text = getattr(model, "tree_", None)
        policy = PositionPolicy(
            "gosdt",
            signatures,
            model,
            text=str(text) if text is not None else str(model),
            fallback_notes=tuple(notes),
        )
        policy.tier(indicators.head(5))  # smoke: predictions must work
        log_event(log, 20, "distilled with GOSDT", rules=len(signatures))
        return policy
    except Exception as exc:  # any failure -> next distiller, reason recorded
        notes.append(f"gosdt unavailable/failed: {type(exc).__name__}: {exc}")

    # --- CORELS ---------------------------------------------------------------
    try:
        from corels import CorelsClassifier

        model = CorelsClassifier(max_card=2, c=0.01, n_iter=100_000)
        model.fit(x_frame.to_numpy(), y, features=list(signatures))
        policy = PositionPolicy(
            "corels", signatures, model, text=str(model.rl()), fallback_notes=tuple(notes)
        )
        policy.tier(indicators.head(5))
        log_event(log, 20, "distilled with CORELS", rules=len(signatures))
        return policy
    except Exception as exc:
        notes.append(f"corels unavailable/failed: {type(exc).__name__}: {exc}")

    # --- imodels greedy rule list ----------------------------------------------
    try:
        from imodels import GreedyRuleListClassifier

        model = GreedyRuleListClassifier(max_depth=3)
        model.fit(x_frame.to_numpy(), y, feature_names=list(signatures))
        policy = PositionPolicy(
            "greedy_rule_list", signatures, model, text=str(model), fallback_notes=tuple(notes)
        )
        policy.tier(indicators.head(5))
        log_event(log, 20, "distilled with greedy rule list", rules=len(signatures))
        return policy
    except Exception as exc:
        notes.append(f"greedy_rule_list failed: {type(exc).__name__}: {exc}")

    # --- sklearn pruned tree (always available) --------------------------------
    from sklearn.tree import DecisionTreeClassifier, export_text

    model = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=200, ccp_alpha=1e-4, random_state=seed
    )
    model.fit(x_frame.to_numpy(), y)
    text = export_text(model, feature_names=list(signatures))
    log_event(log, 20, "distilled with sklearn tree", rules=len(signatures))
    return PositionPolicy(
        "sklearn_tree_d3", signatures, model, text=text, fallback_notes=tuple(notes)
    )
