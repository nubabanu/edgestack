"""Typed public contracts for the continuous edge factory."""

from __future__ import annotations

import enum
from datetime import date, datetime
from functools import reduce
from operator import mul
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgestack.recommendation.hashing import stable_hash


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CoverageState(enum.StrEnum):
    READY = "READY"
    NEEDS_DATA = "NEEDS_DATA"
    BLOCKED_FREE_TIER = "BLOCKED_FREE_TIER"
    BLOCKED_LICENSE = "BLOCKED_LICENSE"
    BLOCKED_POINT_IN_TIME = "BLOCKED_POINT_IN_TIME"
    BLOCKED_COMPLIANCE = "BLOCKED_COMPLIANCE"
    BLOCKED_COMPONENTS = "BLOCKED_COMPONENTS"
    QUALITY_FAILED = "QUALITY_FAILED"


class JobState(enum.StrEnum):
    PENDING = "PENDING"
    LEASED = "LEASED"
    SUCCEEDED = "SUCCEEDED"
    RETRY = "RETRY"
    FAILED = "FAILED"
    BLOCKED_FREE_TIER = "BLOCKED_FREE_TIER"


class CampaignLifecycle(enum.StrEnum):
    NEEDS_DATA = "NEEDS_DATA"
    READY = "READY"
    RUNNING = "RUNNING"
    REJECTED = "REJECTED"
    SHADOW_ELIGIBLE = "SHADOW_ELIGIBLE"
    PAPER_SHADOW = "PAPER_SHADOW"
    PROMOTED = "PROMOTED"
    DEGRADED = "DEGRADED"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"
    BLOCKED_FREE_TIER = "BLOCKED_FREE_TIER"
    BLOCKED_LICENSE = "BLOCKED_LICENSE"
    BLOCKED_POINT_IN_TIME = "BLOCKED_POINT_IN_TIME"
    BLOCKED_COMPLIANCE = "BLOCKED_COMPLIANCE"
    BLOCKED_COMPONENTS = "BLOCKED_COMPONENTS"


class WorkerState(enum.StrEnum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    NIGHTLY_PAUSE = "NIGHTLY_PAUSE"
    STORAGE_BLOCKED = "STORAGE_BLOCKED"
    ERROR = "ERROR"


class ProposalStage(enum.StrEnum):
    """Append-only stages accepted from human or machine research assistants."""

    HYPOTHESIS = "HYPOTHESIS"
    DATA_QUERY = "DATA_QUERY"
    IMPLEMENTATION = "IMPLEMENTATION"
    TRIAGE = "TRIAGE"
    EVALUATION = "EVALUATION"
    REVISION = "REVISION"
    FINAL_HOLDOUT = "FINAL_HOLDOUT"


class ResearchDataQueryV1(_Contract):
    """Frozen, point-in-time data need attached to one proposed hypothesis."""

    schema_version: Literal[1] = 1
    dataset: str
    symbols: tuple[str, ...]
    frequency: str
    start: date
    end: date
    fields: tuple[str, ...]
    required_observations: int = Field(ge=1)
    preferred_providers: tuple[str, ...]
    point_in_time_required: bool = True

    @model_validator(mode="after")
    def _valid_query(self) -> ResearchDataQueryV1:
        if self.end < self.start:
            raise ValueError("data-query end cannot precede start")
        if not self.symbols:
            raise ValueError("data query must name at least one symbol or series")
        if not self.fields:
            raise ValueError("data query must name its required fields")
        if not self.preferred_providers:
            raise ValueError("data query must name at least one preferred provider")
        return self

    def requirement_id(self, proposal_id: str) -> str:
        return stable_hash({"proposal_id": proposal_id, "query": self.model_dump(mode="json")})[:24]


class CandidateProposalV1(_Contract):
    """A finite hypothesis family registered before any result is evaluated.

    External generators are untrusted idea sources. ``previously_accessed``
    therefore defaults to true and ``result_blind`` defaults to false.
    """

    schema_version: Literal[1] = 1
    proposal_id: str
    created_at: datetime
    source: str
    source_uri: str | None = None
    family: str
    name: str
    hypothesis: str
    mechanism: str
    feature_expressions: tuple[str, ...]
    parameter_grid: dict[str, tuple[str | int | float | bool, ...]] = Field(default_factory=dict)
    data_queries: tuple[ResearchDataQueryV1, ...]
    max_trials: int = Field(default=1_500, ge=1, le=1_500)
    result_blind: bool = False
    previously_accessed: bool = True
    notes: tuple[str, ...] = ()

    @property
    def trial_count(self) -> int:
        sizes = [len(values) for values in self.parameter_grid.values()]
        return reduce(mul, sizes, 1)

    @property
    def manifest_hash(self) -> str:
        return stable_hash(
            self.model_dump(
                mode="json",
                exclude={"proposal_id", "created_at"},
            )
        )

    @model_validator(mode="after")
    def _finite_and_honest(self) -> CandidateProposalV1:
        if not self.proposal_id.strip():
            raise ValueError("proposal_id cannot be empty")
        if not self.feature_expressions:
            raise ValueError("proposal must pre-register at least one feature expression")
        if not self.data_queries:
            raise ValueError("proposal must pre-register its data queries")
        if any(not values for values in self.parameter_grid.values()):
            raise ValueError("parameter axes cannot be empty")
        if self.trial_count > self.max_trials:
            raise ValueError(
                f"expanded parameter grid has {self.trial_count} trials; cap is {self.max_trials}"
            )
        if self.result_blind and self.previously_accessed:
            raise ValueError("a previously accessed proposal cannot be declared result blind")
        return self


class ProposalAttemptV1(_Contract):
    """One immutable step in a proposal's complete search-history ledger."""

    schema_version: Literal[1] = 1
    attempt_id: str
    proposal_id: str
    sequence: int = Field(ge=1)
    occurred_at: datetime
    stage: ProposalStage
    code_hash: str | None = None
    data_query_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    result_summary: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None
    viewed_guarded_data: bool = False
    revision_of_attempt_id: str | None = None


class ProposalAuditV1(_Contract):
    schema_version: Literal[1] = 1
    proposal_id: str
    manifest_hash: str
    expected_trials: int = Field(ge=1, le=1_500)
    registered_trials: int = Field(ge=0)
    attempts: int = Field(ge=0)
    final_holdout_accesses: int = Field(ge=0)
    complete_trial_registry: bool
    complete_attempt_sequence: bool
    guarded_data_accessed: bool
    intake_eligible_for_fresh_campaign: bool
    reasons: tuple[str, ...] = ()


class FactorDiagnosticsV1(_Contract):
    """Alphalens-style diagnostics that remain explicitly non-promotional."""

    schema_version: Literal[1] = 1
    diagnostics_id: str
    factor_name: str
    data_hash: str
    observations: int = Field(ge=0)
    sessions: int = Field(ge=0)
    quantiles: int = Field(ge=2, le=20)
    horizons: tuple[int, ...]
    mean_return_by_quantile: dict[str, dict[str, float]] = Field(default_factory=dict)
    top_minus_bottom_mean: dict[str, float] = Field(default_factory=dict)
    rank_ic_mean: dict[str, float] = Field(default_factory=dict)
    rank_ic_t_stat: dict[str, float] = Field(default_factory=dict)
    group_neutral_rank_ic_mean: dict[str, float] = Field(default_factory=dict)
    monotonicity: dict[str, float] = Field(default_factory=dict)
    quantile_turnover: dict[str, float] = Field(default_factory=dict)
    trial_charge: int = Field(ge=1)
    promotion_evidence: Literal[False] = False
    warnings: tuple[str, ...] = ()


class DataCoverageV1(_Contract):
    schema_version: Literal[1] = 1
    dataset_id: str
    dataset: str
    provider: str
    feed: str
    symbol: str
    frequency: str
    start: datetime | None = None
    end: datetime | None = None
    observation_count: int = Field(default=0, ge=0)
    missing_sessions: int = Field(default=0, ge=0)
    quality_status: str = "UNKNOWN"
    point_in_time: bool = False
    content_hash: str = ""
    raw_content_hashes: tuple[str, ...] = ()
    retrieved_at: datetime
    quality_results: dict[str, float | int | str | bool] = Field(default_factory=dict)
    limitations: tuple[str, ...] = ()


class EvidenceGapV1(_Contract):
    schema_version: Literal[1] = 1
    requirement_id: str
    campaign_id: str
    dataset: str
    symbols: tuple[str, ...]
    frequency: str
    start: date
    end: date
    required_observations: int = Field(ge=1)
    observed_observations: int = Field(default=0, ge=0)
    state: CoverageState
    provider: str | None = None
    next_action: str
    job_id: str | None = None

    @model_validator(mode="after")
    def _ready_has_enough_data(self) -> EvidenceGapV1:
        if self.state is CoverageState.READY and (
            self.observed_observations < self.required_observations
        ):
            raise ValueError("READY evidence gap must meet its observation requirement")
        return self


class AcquisitionJobV1(_Contract):
    schema_version: Literal[1] = 1
    job_id: str
    requirement_id: str
    campaign_id: str
    kind: Literal["ACQUIRE_DATA", "EVALUATE_CAMPAIGN", "UPDATE_SHADOW"]
    priority: int = Field(default=100, ge=0, le=10_000)
    state: JobState = JobState.PENDING
    attempts: int = Field(default=0, ge=0)
    not_before: datetime | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    payload: dict[str, str | int | float | bool | list[str] | None] = Field(default_factory=dict)
    last_error: str | None = None


class CampaignSummaryV1(_Contract):
    schema_version: Literal[1] = 1
    campaign_id: str
    manifest_hash: str
    name: str
    family: str
    lifecycle: CampaignLifecycle
    created_at: datetime
    updated_at: datetime
    trial_count: int = Field(default=0, ge=0, le=50_000)
    completed_trials: int = Field(default=0, ge=0)
    data_requirements: tuple[str, ...] = ()
    failure_reasons: tuple[str, ...] = ()
    next_action: str = ""
    artifact_hash: str | None = None
    promotion_eligible: bool = False
    previously_accessed: bool = True
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _completed_not_above_registered(self) -> CampaignSummaryV1:
        if self.completed_trials > self.trial_count:
            raise ValueError("completed trials cannot exceed registered trials")
        if self.promotion_eligible and self.previously_accessed:
            raise ValueError("previously accessed campaigns cannot be promotion eligible")
        return self


class WorkerHealthV1(_Contract):
    schema_version: Literal[1] = 1
    state: WorkerState = WorkerState.IDLE
    paused: bool = False
    current_job_id: str | None = None
    process_priority: str = "below_normal"
    max_workers: int = Field(default=4, ge=1)
    storage_used_gb: float = Field(default=0.0, ge=0)
    storage_cap_gb: float = Field(default=75.0, gt=0)
    heartbeat_at: datetime | None = None
    last_error: str | None = None


class ResearchOverviewV1(_Contract):
    schema_version: Literal[1] = 1
    generated_at: datetime
    worker: WorkerHealthV1
    proposal_count: int = Field(default=0, ge=0)
    fresh_proposal_count: int = Field(default=0, ge=0)
    campaign_counts: dict[str, int] = Field(default_factory=dict)
    acquisition_job_counts: dict[str, int] = Field(default_factory=dict)
    coverage_counts: dict[str, int] = Field(default_factory=dict)
    provider_health: dict[str, str] = Field(default_factory=dict)
    current_campaign: CampaignSummaryV1 | None = None
    evidence_gaps: tuple[EvidenceGapV1, ...] = ()
    next_jobs: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()


class ShadowStrategyV1(_Contract):
    schema_version: Literal[1] = 1
    strategy_id: str
    campaign_id: str
    artifact_hash: str
    status: CampaignLifecycle
    started_at: datetime
    resolved_through: date | None = None
    sessions: int = Field(default=0, ge=0)
    trades: int = Field(default=0, ge=0)
    effective_resolved_outcomes: float = Field(default=0.0, ge=0)
    equity: float = Field(gt=0)
    net_return: float = 0.0
    benchmark_returns: dict[str, float] = Field(default_factory=dict)
    realized_slippage_bps: float | None = None
    warnings: tuple[str, ...] = ()


class GrowthDiagnosticsV1(_Contract):
    schema_version: Literal[1] = 1
    canonical_run_id: str | None = None
    as_of: datetime
    action: Literal["ENTER_NEXT_OPEN", "WAIT", "REDUCE", "NO_ALLOCATION"]
    expected_log_growth: float
    expected_log_growth_lower_95: float
    comparator_log_growth: dict[str, float] = Field(default_factory=dict)
    quarter_kelly_limit: float = Field(ge=0, le=5)
    effective_leverage: float = Field(ge=0, le=5)
    constraint_limits: dict[str, float] = Field(default_factory=dict)
    binding_constraints: tuple[str, ...] = ()
    evidence_state: str
    warnings: tuple[str, ...] = ()
