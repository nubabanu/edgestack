"""Frozen sleeve-promotion gates over annual outer-test portfolio returns."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from edgestack.recommendation.manifests import PromotionDecisionV2, ProspectiveEvidenceV2
from edgestack.recommendation.schemas import AssetKind, EvidenceGrade
from edgestack.validation.clustered import paired_sharpe_improvement

MIN_VALID_OUTER_FOLDS = 5
POSITIVE_FOLD_FRACTION = 0.70
REQUIRED_STRESS_SCENARIOS = frozenset(
    {
        "conservative",
        "stress",
        "financing_200bps",
        "financing_400bps",
        "financing_800bps",
        "delayed_fill",
        "liquidity",
        "participation",
        "adverse_execution",
    }
)


@dataclass(frozen=True)
class PromotionInputs:
    sleeve_id: str
    artifact_hash: str
    asset_kind: AssetKind
    fold_returns: dict[str, pd.Series]
    strategy_returns: pd.Series
    risk_matched_spy: pd.Series
    diversified_baseline: pd.Series
    spa_consistent_pvalue: float
    stepm_superior_ids: tuple[str, ...]
    stress_scenarios: dict[str, bool]
    prospective_evidence: ProspectiveEvidenceV2 | None = None
    compound_ablations_pass: bool | None = None


def risk_match_benchmark(training_strategy: pd.Series, benchmark: pd.Series) -> pd.Series:
    """Scale a benchmark with volatility learned solely from the training period."""
    joined = pd.concat([training_strategy.rename("s"), benchmark.rename("b")], axis=1).dropna()
    strategy_vol = float(joined["s"].std(ddof=1))
    benchmark_vol = float(joined["b"].std(ddof=1))
    scale = strategy_vol / benchmark_vol if benchmark_vol > 0 else 0.0
    return benchmark * scale


def evaluate_promotion(inputs: PromotionInputs, *, seed: int = 42) -> PromotionDecisionV2:
    valid_folds = {name: values.dropna() for name, values in inputs.fold_returns.items()}
    valid_folds = {name: values for name, values in valid_folds.items() if len(values) >= 200}
    positive = sum(float(np.prod(1.0 + values) - 1.0) > 0 for values in valid_folds.values())
    required_positive = math.ceil(POSITIVE_FOLD_FRACTION * len(valid_folds))
    spy_test = paired_sharpe_improvement(
        inputs.strategy_returns, inputs.risk_matched_spy, seed=seed
    )
    baseline_test = paired_sharpe_improvement(
        inputs.strategy_returns, inputs.diversified_baseline, seed=seed + 1
    )
    reasons = []
    if len(valid_folds) < MIN_VALID_OUTER_FOLDS:
        reasons.append(f"valid outer folds {len(valid_folds)} < {MIN_VALID_OUTER_FOLDS}")
    if positive < required_positive:
        reasons.append(f"positive outer folds {positive} < required {required_positive}")
    if inputs.spa_consistent_pvalue > 0.05:
        reasons.append("failed SPA consistent p-value")
    if inputs.sleeve_id not in inputs.stepm_superior_ids:
        reasons.append("not selected by StepM")
    if spy_test["ci_low"] <= 0:
        reasons.append("Sharpe lower bound versus risk-matched SPY is not positive")
    if baseline_test["ci_low"] <= 0:
        reasons.append("Sharpe lower bound versus baseline is not positive")
    failed_stress = sorted(
        name for name in REQUIRED_STRESS_SCENARIOS if not inputs.stress_scenarios.get(name, False)
    )
    if failed_stress:
        reasons.append(f"failed stress scenarios: {', '.join(failed_stress)}")
    if inputs.compound_ablations_pass is False:
        reasons.append("compound failed incremental ablations")
    if inputs.asset_kind is AssetKind.STOCK and (
        inputs.prospective_evidence is None
        or not inputs.prospective_evidence.stock_promotion_clock_satisfied
    ):
        reasons.append("stock prospective evidence clock incomplete")
    promoted = not reasons
    return PromotionDecisionV2(
        sleeve_id=inputs.sleeve_id,
        artifact_hash=inputs.artifact_hash,
        promoted=promoted,
        evidence_grade=EvidenceGrade.PROMOTED if promoted else EvidenceGrade.WATCHLIST,
        valid_outer_folds=len(valid_folds),
        positive_outer_folds=positive,
        spa_consistent_pvalue=inputs.spa_consistent_pvalue,
        stepm_superior=inputs.sleeve_id in inputs.stepm_superior_ids,
        sharpe_lower_bound_vs_spy=spy_test["ci_low"],
        sharpe_lower_bound_vs_baseline=baseline_test["ci_low"],
        stress_scenarios_passed=tuple(
            sorted(name for name, passed in inputs.stress_scenarios.items() if passed)
        ),
        failure_reasons=tuple(reasons),
    )
