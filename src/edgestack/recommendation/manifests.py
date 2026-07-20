"""Immutable experiment, trial, artifact, and promotion records."""

from __future__ import annotations

import enum
from datetime import date, datetime
from typing import Any, Literal

from pydantic import Field

from edgestack.recommendation.hashing import stable_hash
from edgestack.recommendation.schemas import EvidenceGrade, V2Model


class TrialKind(enum.StrEnum):
    STANDALONE = "STANDALONE"
    GATED = "GATED"
    ADDITIVE = "ADDITIVE"
    INTERACTION = "INTERACTION"
    STACKING = "STACKING"
    VOTING = "VOTING"
    EXECUTION_VARIANT = "EXECUTION_VARIANT"
    OPTIMIZER_VARIANT = "OPTIMIZER_VARIANT"
    MANUAL_LEGACY = "MANUAL_LEGACY"


class TrialStatus(enum.StrEnum):
    REGISTERED = "REGISTERED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    CACHED = "CACHED"


class OuterFoldV2(V2Model):
    fold_id: str
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_sessions: int = Field(ge=756)
    test_sessions: int = Field(ge=0)
    valid: bool
    invalid_reason: str | None = None


class TrialRecordV2(V2Model):
    schema_version: Literal[2] = 2
    trial_id: str
    experiment_id: str
    kind: TrialKind
    family: str
    horizon_sessions: int = Field(ge=1)
    parent_trial_ids: tuple[str, ...] = ()
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: TrialStatus = TrialStatus.REGISTERED
    failure_reason: str | None = None
    cached_from_trial_id: str | None = None
    manually_inspected: bool = False


class ExperimentManifestV2(V2Model):
    schema_version: Literal[2] = 2
    experiment_id: str
    code_revision: str
    data_version: str
    data_hashes: dict[str, str]
    universe_definition: dict[str, Any]
    point_in_time_coverage: dict[str, float]
    feature_definitions: tuple[dict[str, Any], ...]
    labels: tuple[dict[str, Any], ...]
    horizons: tuple[int, ...]
    candidate_family: tuple[str, ...]
    execution_policies: tuple[dict[str, Any], ...]
    transaction_costs: dict[str, Any]
    financing: dict[str, Any]
    calibration_method: str
    outer_folds: tuple[OuterFoldV2, ...]
    complete_trial_ids: tuple[str, ...]
    random_seeds: tuple[int, ...]
    policy_version: str
    generated_artifacts: tuple[str, ...] = ()
    evaluation_dates: tuple[date, ...] = ()
    promotion_result: str = "PENDING"
    legacy_period_accessed: bool = True
    incomplete_manual_search_history: bool = False

    @property
    def manifest_hash(self) -> str:
        return stable_hash(self.model_dump(mode="json", exclude={"manifest_hash"}))


class FrozenArtifactV2(V2Model):
    schema_version: Literal[2] = 2
    artifact_type: str
    artifact_version: str
    content_hash: str
    manifest_hash: str
    data_version: str
    feature_version: str
    policy_version: str
    horizon_sessions: int | None = Field(default=None, ge=1)
    payload_file: str
    payload_hash: str


class PromotionDecisionV2(V2Model):
    schema_version: Literal[2] = 2
    sleeve_id: str
    artifact_hash: str
    promoted: bool
    evidence_grade: EvidenceGrade
    valid_outer_folds: int = Field(ge=0)
    positive_outer_folds: int = Field(ge=0)
    spa_consistent_pvalue: float = Field(ge=0, le=1)
    stepm_superior: bool
    sharpe_lower_bound_vs_spy: float
    sharpe_lower_bound_vs_baseline: float
    log_growth_lower_bounds: dict[str, float] = Field(default_factory=dict)
    stress_scenarios_passed: tuple[str, ...]
    failure_reasons: tuple[str, ...] = ()


class ProspectiveEvidenceV2(V2Model):
    schema_version: Literal[2] = 2
    sleeve_id: str
    frozen_artifact_hash: str
    prospective_start: date
    resolved_through: date | None = None
    prospective_sessions: int = Field(default=0, ge=0)
    effective_resolved_outcomes: float = Field(default=0, ge=0)

    @property
    def stock_promotion_clock_satisfied(self) -> bool:
        return self.prospective_sessions >= 252 and self.effective_resolved_outcomes >= 100


class PublicationRecordV2(V2Model):
    schema_version: Literal[2] = 2
    run_id: str
    published_at: datetime
    bundle_hash: str
    data_version: str
    artifact_version: str
    policy_version: str
    file_hashes: dict[str, str]
