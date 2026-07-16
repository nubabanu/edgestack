"""Controlled compound-edge fitting, complementarity, and ablation analysis."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from edgestack.exceptions import ValidationError
from edgestack.recommendation.hashing import stable_hash


class CompoundKind(enum.StrEnum):
    GATED = "GATED"
    ADDITIVE = "ADDITIVE"
    INTERACTION = "INTERACTION"
    STACKING = "STACKING"
    VOTING = "VOTING"


@dataclass(frozen=True)
class CompoundFit:
    kind: CompoundKind
    horizon_sessions: int
    component_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    oof_prediction: np.ndarray
    fold_assignments: np.ndarray
    parameters: dict[str, float]

    @property
    def artifact_hash(self) -> str:
        return stable_hash(
            {
                "kind": self.kind,
                "horizon": self.horizon_sessions,
                "components": self.component_ids,
                "features": self.feature_names,
                "coefficients": self.coefficients,
                "intercept": self.intercept,
                "parameters": self.parameters,
            }
        )


def _validate_inputs(
    predictions: pd.DataFrame,
    target: pd.Series,
    fold_assignments: np.ndarray,
    component_horizons: dict[str, int],
) -> int:
    if predictions.empty or len(predictions) != len(target):
        raise ValidationError("compound predictions and target must be non-empty and aligned")
    if len(fold_assignments) != len(target) or len(np.unique(fold_assignments)) < 2:
        raise ValidationError("compound fitting requires at least two aligned inner folds")
    if set(predictions.columns) != set(component_horizons):
        raise ValidationError("every compound component requires a declared horizon")
    horizons = set(component_horizons.values())
    if len(horizons) != 1:
        raise ValidationError("compound edges cannot mix holding horizons")
    return next(iter(horizons))


def fit_compound(
    kind: CompoundKind,
    predictions: pd.DataFrame,
    target: pd.Series,
    fold_assignments: np.ndarray,
    *,
    component_horizons: dict[str, int],
    alpha: float = 1.0,
    gate_component: str | None = None,
    gate_threshold: float = 0.5,
    vote_threshold: float = 0.5,
) -> CompoundFit:
    horizon = _validate_inputs(predictions, target, fold_assignments, component_horizons)
    x = predictions.astype(float).to_numpy()
    y = target.to_numpy(dtype=float)
    feature_names = tuple(str(c) for c in predictions.columns)
    if kind is CompoundKind.INTERACTION:
        interactions = []
        interaction_names = []
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                interactions.append(x[:, i] * x[:, j])
                interaction_names.append(f"{feature_names[i]}*{feature_names[j]}")
        if interactions:
            x = np.column_stack([x, *interactions])
            feature_names = (*feature_names, *interaction_names)
    if kind is CompoundKind.GATED:
        if gate_component not in predictions:
            raise ValidationError("gated compound requires a named gate component")
        gate_idx = predictions.columns.get_loc(gate_component)
        x = np.column_stack([x.mean(axis=1), (x[:, gate_idx] >= gate_threshold).astype(float)])
        feature_names = ("base_mean", f"gate:{gate_component}")
    if kind is CompoundKind.VOTING:
        votes = (x >= vote_threshold).mean(axis=1)
        return CompoundFit(
            kind=kind,
            horizon_sessions=horizon,
            component_ids=tuple(str(c) for c in predictions.columns),
            feature_names=("vote_fraction",),
            coefficients=(1.0,),
            intercept=0.0,
            oof_prediction=votes,
            fold_assignments=np.asarray(fold_assignments, dtype=int),
            parameters={"vote_threshold": vote_threshold},
        )

    folds = np.asarray(fold_assignments, dtype=int)
    oof = np.full(len(y), np.nan)
    classification = kind is CompoundKind.STACKING
    for fold in np.unique(folds):
        train = folds != fold
        test = ~train
        if classification:
            estimator = LogisticRegression(C=1.0 / max(alpha, 1e-9), max_iter=2_000)
            estimator.fit(x[train], (y[train] > 0).astype(int))
            oof[test] = estimator.predict_proba(x[test])[:, 1]
        else:
            estimator = Ridge(alpha=alpha)
            estimator.fit(x[train], y[train])
            oof[test] = estimator.predict(x[test])
    if not np.isfinite(oof).all():
        raise ValidationError("compound cross-fitting did not cover all rows")
    if classification:
        final = LogisticRegression(C=1.0 / max(alpha, 1e-9), max_iter=2_000).fit(
            x, (y > 0).astype(int)
        )
        coefficients = tuple(float(v) for v in final.coef_[0])
        intercept = float(final.intercept_[0])
    else:
        final = Ridge(alpha=alpha).fit(x, y)
        coefficients = tuple(float(v) for v in final.coef_)
        intercept = float(final.intercept_)
    return CompoundFit(
        kind=kind,
        horizon_sessions=horizon,
        component_ids=tuple(str(c) for c in predictions.columns),
        feature_names=feature_names,
        coefficients=coefficients,
        intercept=intercept,
        oof_prediction=oof,
        fold_assignments=folds,
        parameters={
            "alpha": alpha,
            "gate_threshold": gate_threshold,
            "vote_threshold": vote_threshold,
        },
    )


def complementarity_metrics(
    predictions: pd.DataFrame,
    realized_contributions: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    """Measure dependence; economic family names are deliberately ignored."""
    if not predictions.columns.equals(realized_contributions.columns):
        raise ValidationError("prediction and contribution components must match")
    residuals = predictions.sub(target, axis=0)
    residual_corr = residuals.corr().to_numpy()
    contribution_corr = realized_contributions.corr().to_numpy()

    def max_off_diagonal(matrix: np.ndarray) -> float:
        if len(matrix) < 2:
            return 0.0
        return float(np.max(np.abs(matrix[np.triu_indices_from(matrix, k=1)])))

    base = predictions.mean(axis=1)
    base_error = float(np.mean((target - base) ** 2))
    best_error = min(float(np.mean((target - predictions[c]) ** 2)) for c in predictions)
    return {
        "max_abs_residual_correlation": max_off_diagonal(residual_corr),
        "max_abs_contribution_correlation": max_off_diagonal(contribution_corr),
        "incremental_mse_improvement": best_error - base_error,
    }


def compound_ablations(
    fit: CompoundFit,
    component_predictions: pd.DataFrame,
    target: pd.Series,
    *,
    uncertainty_penalty: float = 1.96,
) -> dict[str, float]:
    """Compare full compound with components, equal weight, voting, and removals."""
    y = target.to_numpy(dtype=float)

    def utility(prediction: np.ndarray) -> float:
        contribution = prediction * y
        standard_error = contribution.std(ddof=1) / np.sqrt(max(1, len(contribution)))
        return float(contribution.mean() - uncertainty_penalty * standard_error)

    out = {"full": utility(fit.oof_prediction)}
    for component in component_predictions:
        out[f"component:{component}"] = utility(component_predictions[component].to_numpy())
    out["equal_weight"] = utility(component_predictions.mean(axis=1).to_numpy())
    out["simple_vote"] = utility((component_predictions.to_numpy() >= 0.5).mean(axis=1))
    if len(fit.component_ids) > 1:
        for component in fit.component_ids:
            remaining = component_predictions.drop(columns=[component]).mean(axis=1).to_numpy()
            out[f"remove:{component}"] = utility(remaining)
    return out


def ablations_pass(ablations: dict[str, float], *, tolerance: float = 1e-12) -> bool:
    full = ablations["full"]
    comparisons = [value for name, value in ablations.items() if name != "full"]
    return bool(comparisons) and full > max(comparisons) + tolerance
