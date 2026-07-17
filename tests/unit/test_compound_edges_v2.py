"""Compound-edge fitting, ablation, and promotion-gate tests."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from edgestack.exceptions import ValidationError
from edgestack.recommendation.compounds import (
    CompoundKind,
    ablations_pass,
    complementarity_metrics,
    compound_ablations,
    fit_compound,
)
from edgestack.recommendation.manifests import ProspectiveEvidenceV2
from edgestack.recommendation.promotion import PromotionInputs, evaluate_promotion
from edgestack.recommendation.schemas import AssetKind


def _compound_data(seed: int = 4):
    rng = np.random.default_rng(seed)
    n = 900
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    predictions = pd.DataFrame({"momentum": x1, "friday": x2})
    target = pd.Series(0.2 * x1 + 0.15 * x2 + 1.5 * x1 * x2 + rng.normal(0, 0.1, n))
    folds = np.repeat(np.arange(3), n // 3)
    return predictions, target, folds


def test_interaction_is_cross_fitted_and_passes_incremental_ablations() -> None:
    predictions, target, folds = _compound_data()
    fit = fit_compound(
        CompoundKind.INTERACTION,
        predictions,
        target,
        folds,
        component_horizons={"momentum": 10, "friday": 10},
        alpha=0.1,
    )
    assert "momentum*friday" in fit.feature_names
    assert np.isfinite(fit.oof_prediction).all()
    ablations = compound_ablations(fit, predictions, target)
    assert ablations_pass(ablations)


def test_stacking_consumes_inner_fold_predictions_and_is_frozen() -> None:
    predictions, target, folds = _compound_data()
    binary_target = pd.Series((target > 0).astype(float))
    a = fit_compound(
        CompoundKind.STACKING,
        predictions,
        binary_target,
        folds,
        component_horizons={"momentum": 5, "friday": 5},
    )
    b = fit_compound(
        CompoundKind.STACKING,
        predictions,
        binary_target,
        folds,
        component_horizons={"momentum": 5, "friday": 5},
    )
    assert a.artifact_hash == b.artifact_hash
    np.testing.assert_allclose(a.oof_prediction, b.oof_prediction)


def test_voting_is_explicit_and_horizons_cannot_mix() -> None:
    predictions, target, folds = _compound_data()
    vote = fit_compound(
        CompoundKind.VOTING,
        predictions,
        target,
        folds,
        component_horizons={"momentum": 10, "friday": 10},
        vote_threshold=0.0,
    )
    assert vote.parameters["vote_threshold"] == 0.0
    with pytest.raises(ValidationError, match="cannot mix"):
        fit_compound(
            CompoundKind.ADDITIVE,
            predictions,
            target,
            folds,
            component_horizons={"momentum": 5, "friday": 20},
        )


def test_family_labels_do_not_imply_independence() -> None:
    predictions, target, _ = _compound_data()
    duplicate = pd.DataFrame(
        {"economic_momentum": predictions["momentum"], "economic_calendar": predictions["momentum"]}
    )
    metrics = complementarity_metrics(duplicate, duplicate, target)
    assert metrics["max_abs_residual_correlation"] > 0.99
    assert metrics["max_abs_contribution_correlation"] > 0.99


def _promotion_inputs(asset_kind: AssetKind, prospective=None) -> PromotionInputs:
    rng = np.random.default_rng(12)
    index = pd.bdate_range("2018-01-01", periods=1_100)
    strategy = pd.Series(rng.normal(0.0015, 0.004, len(index)), index=index)
    spy = pd.Series(rng.normal(0.0001, 0.004, len(index)), index=index)
    baseline = pd.Series(rng.normal(0.0001, 0.0035, len(index)), index=index)
    folds = {
        str(year): strategy.iloc[i * 220 : (i + 1) * 220]
        for i, year in enumerate(range(2020, 2025))
    }
    return PromotionInputs(
        sleeve_id="candidate",
        artifact_hash="artifact",
        asset_kind=asset_kind,
        fold_returns=folds,
        strategy_returns=strategy,
        risk_matched_spy=spy,
        diversified_baseline=baseline,
        spa_consistent_pvalue=0.01,
        stepm_superior_ids=("candidate",),
        stress_scenarios={
            "conservative": True,
            "stress": True,
            "financing_200bps": True,
            "financing_400bps": True,
            "financing_800bps": True,
            "delayed_fill": True,
            "liquidity": True,
            "participation": True,
            "adverse_execution": True,
        },
        prospective_evidence=prospective,
        compound_ablations_pass=True,
    )


def test_stock_cannot_promote_before_both_prospective_gates() -> None:
    incomplete = ProspectiveEvidenceV2(
        sleeve_id="candidate",
        frozen_artifact_hash="artifact",
        prospective_start=date(2026, 1, 1),
        prospective_sessions=252,
        effective_resolved_outcomes=99.9,
    )
    rejected = evaluate_promotion(_promotion_inputs(AssetKind.STOCK, incomplete))
    assert not rejected.promoted
    assert any("prospective" in reason for reason in rejected.failure_reasons)

    complete = incomplete.model_copy(update={"effective_resolved_outcomes": 100.0})
    promoted = evaluate_promotion(_promotion_inputs(AssetKind.STOCK, complete))
    assert promoted.promoted


def test_compound_fails_when_any_incremental_ablation_gate_fails() -> None:
    rejected = evaluate_promotion(
        _promotion_inputs(AssetKind.ETF),
    ).model_copy()
    assert rejected.promoted
    inputs = _promotion_inputs(AssetKind.ETF)
    inputs = PromotionInputs(**{**inputs.__dict__, "compound_ablations_pass": False})
    decision = evaluate_promotion(inputs)
    assert not decision.promoted
    assert any("ablation" in reason for reason in decision.failure_reasons)
