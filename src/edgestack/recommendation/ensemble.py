"""Cross-validated robust allocation across independently promoted sleeves.

This layer never turns rejected or watchlist candidates into capital. It only
chooses among three pre-declared allocation rules after every input sleeve has
already passed promotion and independent shadow eligibility.
"""

from __future__ import annotations

import enum
import math
from itertools import pairwise

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from edgestack.exceptions import ValidationError
from edgestack.recommendation.hashing import stable_hash
from edgestack.recommendation.manifests import TrialKind, TrialRecordV2, TrialStatus
from edgestack.recommendation.portfolio import estimate_covariance
from edgestack.recommendation.schemas import EvidenceGrade, SleeveContributionV2
from edgestack.validation.metrics import effective_sample_size


class EnsembleMethod(enum.StrEnum):
    EQUAL = "EQUAL"
    INVERSE_VOL = "INVERSE_VOL"
    MIN_VARIANCE = "MIN_VARIANCE"


class PromotedEnsembleV1(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    ensemble_id: str
    eligible: bool
    selected_method: EnsembleMethod | None = None
    sleeve_ids: tuple[str, ...]
    optimizer_trial_ids: tuple[str, ...]
    weights: dict[str, float] = Field(default_factory=dict)
    family_weights: dict[str, float] = Field(default_factory=dict)
    covariance_version: str | None = None
    cross_validated_log_growth_lower_95: float = 0.0
    method_log_growth_lower_95: dict[str, float] = Field(default_factory=dict)
    stressed_volatility: float = 0.0
    expected_shortfall_95: float = 0.0
    max_pairwise_correlation: float = 0.0
    effective_breadth: float = 0.0
    trial_charge: int = Field(default=3, ge=3, le=3)
    failure_reasons: tuple[str, ...] = ()


def _data_hash(frame: pd.DataFrame) -> str:
    hashes = pd.util.hash_pandas_object(frame, index=True).astype("uint64")
    return stable_hash(
        {
            "columns": tuple(str(column) for column in frame.columns),
            "dtypes": tuple(str(dtype) for dtype in frame.dtypes),
            "row_hashes": tuple(int(value) for value in hashes),
        }
    )


def _constraints(
    families: tuple[str, ...],
    *,
    max_family_weight: float,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = [
        {"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}
    ]
    distinct = tuple(sorted(set(families)))
    if len(distinct) <= 1:
        return output
    for family in distinct:
        indices = np.asarray(
            [index for index, value in enumerate(families) if value == family],
            dtype=int,
        )
        output.append(
            {
                "type": "ineq",
                "fun": lambda weights, idx=indices: float(max_family_weight - np.sum(weights[idx])),
            }
        )
    return output


def _project_weights(
    target: np.ndarray,
    families: tuple[str, ...],
    *,
    max_weight: float,
    max_family_weight: float,
) -> np.ndarray:
    count = len(target)
    if count * max_weight < 1.0 - 1e-10:
        raise ValidationError("per-sleeve cap makes a fully invested ensemble impossible")
    result = minimize(
        lambda weights: float(np.square(weights - target).sum()),
        np.repeat(1.0 / count, count),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * count,
        constraints=_constraints(families, max_family_weight=max_family_weight),
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not result.success:
        raise ValidationError(f"ensemble constraints are infeasible: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    return weights / weights.sum()


def _fit_method(
    returns: pd.DataFrame,
    method: EnsembleMethod,
    families: tuple[str, ...],
    *,
    max_weight: float,
    max_family_weight: float,
) -> np.ndarray:
    values = returns.to_numpy(dtype=float)
    count = values.shape[1]
    if method is EnsembleMethod.EQUAL:
        target = np.repeat(1.0 / count, count)
        return _project_weights(
            target,
            families,
            max_weight=max_weight,
            max_family_weight=max_family_weight,
        )
    covariance = LedoitWolf().fit(values).covariance_
    if method is EnsembleMethod.INVERSE_VOL:
        volatility = np.sqrt(np.maximum(np.diag(covariance), 1e-18))
        target = 1.0 / volatility
        target /= target.sum()
        return _project_weights(
            target,
            families,
            max_weight=max_weight,
            max_family_weight=max_family_weight,
        )
    result = minimize(
        lambda weights: float(weights @ covariance @ weights),
        np.repeat(1.0 / count, count),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * count,
        constraints=_constraints(families, max_family_weight=max_family_weight),
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not result.success:
        raise ValidationError(f"minimum-variance ensemble failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    return weights / weights.sum()


def _rolling_log_growth(
    returns: pd.DataFrame,
    method: EnsembleMethod,
    families: tuple[str, ...],
    *,
    max_weight: float,
    max_family_weight: float,
) -> np.ndarray:
    first_test = max(40, len(returns) // 2)
    boundaries = np.linspace(first_test, len(returns), 4, dtype=int)
    pieces: list[np.ndarray] = []
    for start, end in pairwise(boundaries):
        if end <= start:
            continue
        train = returns.iloc[:start]
        test = returns.iloc[start:end]
        weights = _fit_method(
            train,
            method,
            families,
            max_weight=max_weight,
            max_family_weight=max_family_weight,
        )
        portfolio_returns = test.to_numpy(dtype=float) @ weights
        if np.any(portfolio_returns <= -1.0):
            raise ValidationError("ensemble path contains a return at or below -100%")
        pieces.append(np.log1p(portfolio_returns))
    if not pieces:
        raise ValidationError("ensemble cross-validation produced no out-of-sample returns")
    return np.concatenate(pieces)


def _annualized_lower(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    information = effective_sample_size(values)
    standard_error = float(values.std(ddof=1)) / math.sqrt(max(1.0, information))
    return float((values.mean() - 1.96 * standard_error) * 252.0)


def ensemble_optimizer_trials(
    returns: pd.DataFrame,
    *,
    capital_eligible_sleeves: tuple[SleeveContributionV2, ...],
    max_weight: float = 0.60,
    max_family_weight: float = 0.70,
) -> tuple[TrialRecordV2, ...]:
    """Return the exact optimizer variants that must be registered pre-evaluation."""
    sleeve_ids = tuple(sleeve.sleeve_id for sleeve in capital_eligible_sleeves)
    if len(sleeve_ids) < 2 or len(set(sleeve_ids)) != len(sleeve_ids):
        raise ValidationError("optimizer trial registration needs unique ensemble sleeves")
    if {str(column) for column in returns.columns} != set(sleeve_ids):
        raise ValidationError("optimizer trial return columns do not match the sleeve registry")
    clean = returns.loc[:, list(sleeve_ids)].astype(float).replace([np.inf, -np.inf], np.nan)
    identity = {
        "sleeves": tuple(
            (sleeve.sleeve_id, sleeve.artifact_hash) for sleeve in capital_eligible_sleeves
        ),
        "returns_hash": _data_hash(clean),
        "methods": tuple(EnsembleMethod),
        "max_weight": max_weight,
        "max_family_weight": max_family_weight,
        "policy": "expanding-3-fold-lower-95-v1",
    }
    experiment_id = stable_hash(identity)[:24]
    horizon = max(sleeve.horizon_sessions for sleeve in capital_eligible_sleeves)
    return tuple(
        TrialRecordV2(
            trial_id=stable_hash({"ensemble": experiment_id, "method": method.value})[:24],
            experiment_id=experiment_id,
            kind=TrialKind.OPTIMIZER_VARIANT,
            family="promoted_ensemble_allocation",
            horizon_sessions=horizon,
            parameters={
                "method": method.value,
                "sleeve_ids": list(sleeve_ids),
                "max_weight": max_weight,
                "max_family_weight": max_family_weight,
            },
            status=TrialStatus.REGISTERED,
        )
        for method in EnsembleMethod
    )


def build_promoted_ensemble(
    returns: pd.DataFrame,
    *,
    capital_eligible_sleeves: tuple[SleeveContributionV2, ...],
    registered_optimizer_trials: tuple[TrialRecordV2, ...] = (),
    max_weight: float = 0.60,
    max_family_weight: float = 0.70,
) -> PromotedEnsembleV1:
    """Cross-validate fixed allocators over promoted, shadow-qualified sleeves only."""
    if not 0 < max_weight <= 1 or not 0 < max_family_weight <= 1:
        raise ValidationError("ensemble weight caps must be in (0, 1]")
    if len(capital_eligible_sleeves) < 2:
        raise ValidationError("an ensemble requires at least two capital-eligible sleeves")
    if any(
        sleeve.evidence_grade is not EvidenceGrade.PROMOTED for sleeve in capital_eligible_sleeves
    ):
        raise ValidationError("an ensemble cannot include an unpromoted sleeve")
    if any(sleeve.expected_return_lower_95 <= 0 for sleeve in capital_eligible_sleeves):
        raise ValidationError("ensemble allocation cannot rescue a non-positive sleeve")
    sleeve_ids = tuple(sleeve.sleeve_id for sleeve in capital_eligible_sleeves)
    if len(set(sleeve_ids)) != len(sleeve_ids):
        raise ValidationError("ensemble sleeve ids must be unique")
    if {str(column) for column in returns.columns} != set(sleeve_ids):
        raise ValidationError(
            "return columns must exactly match the capital-eligible sleeve registry"
        )
    clean = returns.loc[:, list(sleeve_ids)].astype(float).replace([np.inf, -np.inf], np.nan)
    if len(clean) < 80 or clean.isna().any().any():
        raise ValidationError("ensemble selection needs 80 aligned, finite sessions")
    expected_optimizer_trials = ensemble_optimizer_trials(
        clean,
        capital_eligible_sleeves=capital_eligible_sleeves,
        max_weight=max_weight,
        max_family_weight=max_family_weight,
    )
    if registered_optimizer_trials != expected_optimizer_trials:
        raise ValidationError(
            "all three optimizer variants must be registered before ensemble evaluation"
        )
    families = tuple(sleeve.family for sleeve in capital_eligible_sleeves)
    methods = tuple(EnsembleMethod)
    method_paths = {
        method: _rolling_log_growth(
            clean,
            method,
            families,
            max_weight=max_weight,
            max_family_weight=max_family_weight,
        )
        for method in methods
    }
    lower_bounds = {method: _annualized_lower(path) for method, path in method_paths.items()}
    selected = max(methods, key=lambda method: (lower_bounds[method], method.value))
    ensemble_id = expected_optimizer_trials[0].experiment_id
    optimizer_trial_ids = tuple(trial.trial_id for trial in expected_optimizer_trials)
    if lower_bounds[selected] <= 0:
        return PromotedEnsembleV1(
            ensemble_id=ensemble_id,
            eligible=False,
            selected_method=selected,
            sleeve_ids=sleeve_ids,
            optimizer_trial_ids=optimizer_trial_ids,
            method_log_growth_lower_95={
                method.value: value for method, value in lower_bounds.items()
            },
            cross_validated_log_growth_lower_95=lower_bounds[selected],
            failure_reasons=(
                "No pre-declared allocation method has positive lower-confidence "
                "out-of-sample log growth.",
            ),
        )

    weights = _fit_method(
        clean,
        selected,
        families,
        max_weight=max_weight,
        max_family_weight=max_family_weight,
    )
    covariance = estimate_covariance(clean)
    stressed_volatility = float(
        np.sqrt(max(0.0, weights @ covariance.stressed_annualized @ weights))
    )
    selected_path = method_paths[selected]
    threshold = float(np.quantile(selected_path, 0.05))
    tail = selected_path[selected_path <= threshold]
    expected_shortfall = float(tail.mean()) if len(tail) else threshold
    correlation = clean.corr().to_numpy(dtype=float)
    off_diagonal = correlation[~np.eye(len(correlation), dtype=bool)]
    max_correlation = float(np.nanmax(off_diagonal)) if len(off_diagonal) else 0.0
    breadth_denominator = float(weights @ np.nan_to_num(correlation, nan=1.0) @ weights)
    breadth = min(float(len(weights)), 1.0 / max(breadth_denominator, 1e-12))
    family_weights = {
        family: float(
            sum(weight for weight, value in zip(weights, families, strict=True) if value == family)
        )
        for family in sorted(set(families))
    }
    return PromotedEnsembleV1(
        ensemble_id=ensemble_id,
        eligible=True,
        selected_method=selected,
        sleeve_ids=sleeve_ids,
        optimizer_trial_ids=optimizer_trial_ids,
        weights=dict(zip(sleeve_ids, (float(value) for value in weights), strict=True)),
        family_weights=family_weights,
        covariance_version=covariance.version,
        cross_validated_log_growth_lower_95=lower_bounds[selected],
        method_log_growth_lower_95={method.value: value for method, value in lower_bounds.items()},
        stressed_volatility=stressed_volatility,
        expected_shortfall_95=expected_shortfall,
        max_pairwise_correlation=max_correlation,
        effective_breadth=breadth,
    )
