"""Versioned public contracts for the canonical recommendation bundle."""

from __future__ import annotations

import enum
from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgestack.recommendation.hashing import stable_hash


class V2Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RecommendationStatus(enum.StrEnum):
    ACTIVE = "ACTIVE"
    BASELINE_ONLY = "BASELINE_ONLY"
    WATCHLIST_ONLY = "WATCHLIST_ONLY"
    NO_ALLOCATION = "NO_ALLOCATION"


class DrawdownState(enum.StrEnum):
    NORMAL = "NORMAL"
    DELEVERAGING = "DELEVERAGING"
    CASH_LATCHED = "CASH_LATCHED"
    RESET_ELIGIBLE = "RESET_ELIGIBLE"


class AssetKind(enum.StrEnum):
    ETF = "ETF"
    STOCK = "STOCK"
    CASH = "CASH"


class EvidenceGrade(enum.StrEnum):
    PROMOTED = "PROMOTED"
    PROSPECTIVE = "PROSPECTIVE"
    WATCHLIST = "WATCHLIST"
    POLICY = "POLICY"
    INSUFFICIENT = "INSUFFICIENT"


class WeightV2(V2Model):
    symbol: str
    weight: float
    asset_kind: AssetKind
    sector: str = "unknown"


class BaselinePolicyV2(V2Model):
    schema_version: Literal[2] = 2
    policy_version: str
    effective_date: date
    weights: tuple[WeightV2, ...]
    rebalance_schedule: str
    drift_threshold: float = Field(gt=0, le=1)
    execution_policy: str
    total_return_treatment: str
    one_way_cost_bps: float = Field(ge=0)
    participation_limit: float = Field(gt=0, le=1)
    alpha_claimed: Literal[False] = False

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> BaselinePolicyV2:
        if abs(sum(w.weight for w in self.weights) - 1.0) > 1e-9:
            raise ValueError("baseline weights must sum to 1")
        if any(w.weight < 0 for w in self.weights):
            raise ValueError("baseline is long-only")
        return self


class SleeveContributionV2(V2Model):
    sleeve_id: str
    artifact_hash: str
    horizon_sessions: int = Field(ge=1)
    compound: bool = False
    family: str
    symbol_weights: tuple[WeightV2, ...]
    expected_net_return: float
    expected_return_lower_95: float
    effective_sample_size: float = Field(ge=0)
    evidence_grade: EvidenceGrade
    signal_parameters: dict[str, Any] = Field(default_factory=dict)


class WatchlistEntryV2(V2Model):
    symbol: str
    asset_kind: AssetKind
    horizon_sessions: int = Field(ge=1)
    family: str
    thesis: str
    evidence_grade: EvidenceGrade = EvidenceGrade.WATCHLIST
    prospective_sessions: int = Field(default=0, ge=0)
    effective_resolved_outcomes: float = Field(default=0, ge=0)
    zero_weight_reason: str


class FreshnessV2(V2Model):
    as_of: datetime
    expected_session: date
    is_fresh: bool
    age_business_days: int = Field(ge=0)
    complete: bool
    compatible: bool
    reasons: tuple[str, ...] = ()


class BaseRecommendationV2(V2Model):
    schema_version: Literal[2] = 2
    status: RecommendationStatus
    as_of: datetime
    execution_at: datetime
    artifact_version: str
    data_version: str
    policy_version: str
    baseline_weights: tuple[WeightV2, ...]
    promoted_sleeves: tuple[SleeveContributionV2, ...] = ()
    promoted_compound_sleeves: tuple[SleeveContributionV2, ...] = ()
    unlevered_base_weights: tuple[WeightV2, ...]
    expected_returns: dict[str, float] = Field(default_factory=dict)
    expected_net_return: float = 0.0
    expected_volatility: float = Field(default=0.0, ge=0)
    covariance_version: str
    turnover_estimate: float = Field(default=0.0, ge=0)
    evidence_grades: dict[str, EvidenceGrade] = Field(default_factory=dict)
    watchlist: tuple[WatchlistEntryV2, ...] = ()
    freshness: FreshnessV2
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _base_invariants(self) -> BaseRecommendationV2:
        gross = sum(abs(w.weight) for w in self.unlevered_base_weights)
        if self.status is not RecommendationStatus.NO_ALLOCATION and abs(gross - 1.0) > 1e-8:
            raise ValueError("actionable unlevered base weights must have unit gross exposure")
        if self.status is RecommendationStatus.ACTIVE and not (
            self.promoted_sleeves or self.promoted_compound_sleeves
        ):
            raise ValueError("ACTIVE requires at least one promoted sleeve")
        return self


class RiskProfileV2(V2Model):
    schema_version: Literal[2] = 2
    account_equity: float = Field(default=100_000.0, gt=0)
    target_volatility: float = Field(default=0.12, gt=0, le=0.30)
    maximum_drawdown: float = Field(default=0.15, gt=0, le=0.50)
    maximum_gross_leverage: float = Field(default=1.0, ge=0, le=5.0)
    funding_spread_bps: float = Field(default=200.0, ge=0, le=2_000)
    per_stock_cap: float = Field(default=0.03, gt=0, le=0.10)
    sector_cap: float = Field(default=0.20, gt=0, le=1.0)

    @property
    def profile_hash(self) -> str:
        return stable_hash(self.model_dump(mode="json", exclude={"profile_hash"}))


class RiskStateV2(V2Model):
    schema_version: Literal[2] = 2
    state_version: int = Field(default=0, ge=0)
    previous_effective_leverage: float = Field(default=0.0, ge=0, le=5)
    peak_equity: float = Field(gt=0)
    current_equity: float = Field(gt=0)
    current_drawdown: float = Field(default=0.0, ge=0, le=1)
    drawdown_state: DrawdownState = DrawdownState.NORMAL
    cash_latched: bool = False
    reset_eligible: bool = False
    sessions_since_latch: int = Field(default=0, ge=0)
    last_session: date | None = None
    previous_target_weights: tuple[WeightV2, ...] = ()

    @model_validator(mode="after")
    def _state_consistency(self) -> RiskStateV2:
        expected = max(0.0, 1.0 - self.current_equity / self.peak_equity)
        if abs(expected - self.current_drawdown) > 1e-6:
            raise ValueError("current_drawdown must match peak and current equity")
        if (
            self.drawdown_state in {DrawdownState.CASH_LATCHED, DrawdownState.RESET_ELIGIBLE}
            and not self.cash_latched
        ):
            raise ValueError("latched drawdown states require cash_latched=true")
        if self.reset_eligible != (self.drawdown_state is DrawdownState.RESET_ELIGIBLE):
            raise ValueError("reset_eligible must match RESET_ELIGIBLE state")
        return self

    @classmethod
    def initial(cls, equity: float) -> RiskStateV2:
        return cls(peak_equity=equity, current_equity=equity)


class ConstraintResultV2(V2Model):
    name: str
    leverage_limit: float = Field(ge=0)
    binding: bool = False
    detail: str = ""


class PortfolioRecommendationV2(V2Model):
    schema_version: Literal[2] = 2
    status: RecommendationStatus
    as_of: datetime
    execution_at: datetime
    artifact_version: str
    data_version: str
    policy_version: str
    risk_profile_hash: str
    input_risk_state_version: int = Field(ge=0)
    output_risk_state: RiskStateV2
    baseline_weights: tuple[WeightV2, ...]
    base_recommendation_weights: tuple[WeightV2, ...]
    personalized_target_weights: tuple[WeightV2, ...]
    effective_leverage: float = Field(ge=0, le=5)
    constraints: tuple[ConstraintResultV2, ...]
    binding_constraints: tuple[str, ...]
    expected_net_return: float
    expected_volatility: float = Field(ge=0)
    funding_cost: float = Field(ge=0)
    turnover: float = Field(ge=0)
    one_day_stress_loss: float = Field(ge=0)
    multi_session_stress_loss: float = Field(ge=0)
    evidence_grade: EvidenceGrade
    freshness: FreshnessV2
    warnings: tuple[str, ...] = ()
    compatibility_metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _uniform_sizing_invariants(self) -> PortfolioRecommendationV2:
        cash = [
            weight
            for weight in self.personalized_target_weights
            if weight.asset_kind is AssetKind.CASH
        ]
        if len(cash) != 1 or abs(cash[0].weight - (1.0 - self.effective_leverage)) > 1e-8:
            raise ValueError("personalized recommendation requires one balancing cash weight")
        risky = {
            weight.symbol: weight.weight
            for weight in self.personalized_target_weights
            if weight.asset_kind is not AssetKind.CASH and abs(weight.weight) > 1e-12
        }
        if abs(sum(abs(weight) for weight in risky.values()) - self.effective_leverage) > 1e-8:
            raise ValueError("risky target gross must equal effective leverage")
        if self.compatibility_metadata.get("stale_preservation") != "true":
            base = {
                weight.symbol: weight.weight
                for weight in self.base_recommendation_weights
                if abs(weight.weight) > 1e-12
            }
            expected = {
                symbol: weight * self.effective_leverage
                for symbol, weight in base.items()
                if abs(weight * self.effective_leverage) > 1e-12
            }
            if set(risky) != set(expected) or any(
                abs(risky[symbol] - expected[symbol]) > 1e-8 for symbol in expected
            ):
                raise ValueError("risk preferences may only scale base weights uniformly")
        return self


class CanonicalRecommendationBundleV2(V2Model):
    schema_version: Literal[2] = 2
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session: date
    as_of: datetime
    execution_at: datetime
    data_version: str
    artifact_version: str
    policy_version: str
    baseline_policy: BaselinePolicyV2
    default_risk_profile: RiskProfileV2
    base_recommendation: BaseRecommendationV2
    default_recommendation: PortfolioRecommendationV2
    disclaimer: str = (
        "Research and paper-trading output only. Not investment advice; "
        "leverage can cause losses exceeding invested capital."
    )

    @model_validator(mode="after")
    def _versions_are_atomic(self) -> CanonicalRecommendationBundleV2:
        children = (self.base_recommendation, self.default_recommendation)
        for child in children:
            if (
                child.as_of != self.as_of
                or child.execution_at != self.execution_at
                or child.data_version != self.data_version
                or child.artifact_version != self.artifact_version
                or child.policy_version != self.policy_version
            ):
                raise ValueError("bundle children must share the bundle version set")
        if self.baseline_policy.policy_version != self.policy_version:
            raise ValueError("baseline policy version does not match bundle")
        if self.default_risk_profile.profile_hash != self.default_recommendation.risk_profile_hash:
            raise ValueError("default risk profile does not match default recommendation")
        if (
            self.base_recommendation.unlevered_base_weights
            != self.default_recommendation.base_recommendation_weights
            or self.base_recommendation.baseline_weights
            != self.default_recommendation.baseline_weights
        ):
            raise ValueError("default recommendation must size the bundled frozen base")
        return self

    @property
    def bundle_hash(self) -> str:
        return stable_hash(self.model_dump(mode="json", exclude={"generated_at", "bundle_hash"}))
