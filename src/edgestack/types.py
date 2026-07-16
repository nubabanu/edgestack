"""Core domain models shared across EdgeStack.

Everything here is a frozen Pydantic model or enum so that domain objects are
immutable, validated on construction and JSON-serializable via ``model_dump_json``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Side(enum.StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class EdgeStatus(enum.StrEnum):
    CANDIDATE = "CANDIDATE"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    VALIDATED = "VALIDATED"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


class CostScenario(enum.StrEnum):
    OPTIMISTIC = "OPTIMISTIC"
    BASE = "BASE"
    CONSERVATIVE = "CONSERVATIVE"
    STRESS = "STRESS"


class Family(enum.StrEnum):
    """Evidence families used to prevent correlated indicators double counting."""

    TREND = "trend"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    STRUCTURE = "structure"
    CALENDAR = "calendar"
    EVENT = "event"
    REGIME = "regime"
    SECTOR = "sector"
    BREADTH = "breadth"
    CROSS_ASSET = "cross_asset"
    LIQUIDITY = "liquidity"
    DEFENSIVE = "defensive"


class EntryMethod(enum.StrEnum):
    NEXT_OPEN = "NEXT_OPEN"
    NEXT_CLOSE = "NEXT_CLOSE"
    PULLBACK = "PULLBACK"
    BREAKOUT = "BREAKOUT"
    SUPPORT_BOUNCE = "SUPPORT_BOUNCE"
    MEAN_REVERSION = "MEAN_REVERSION"
    LIMIT_ZONE = "LIMIT_ZONE"


class CandidateStatus(enum.StrEnum):
    RESEARCH_CANDIDATE = "RESEARCH_CANDIDATE"
    SHORT_RESEARCH_CANDIDATE = "SHORT_RESEARCH_CANDIDATE"
    NO_TRADE = "NO_TRADE"


# ---------------------------------------------------------------------------
# Rule AST — JSON-serializable conditional definitions
# ---------------------------------------------------------------------------


class Predicate(FrozenModel):
    """A single condition over a registered feature.

    ``value`` compares against a literal; ``quantile`` compares against a
    training-set quantile of the feature (resolved per fold, never globally).
    Exactly one of the two must be provided.
    """

    kind: Literal["predicate"] = "predicate"
    feature: str
    op: Literal["<", "<=", ">", ">=", "==", "!="]
    value: float | int | bool | str | None = None
    quantile: float | None = Field(default=None, ge=0.0, le=1.0)

    def describe(self) -> str:
        if self.quantile is not None:
            return f"{self.feature} {self.op} q{self.quantile:g}(train)"
        return f"{self.feature} {self.op} {self.value}"


class AllOf(FrozenModel):
    kind: Literal["all"] = "all"
    conditions: tuple[Condition, ...]

    def describe(self) -> str:
        return " AND ".join(c.describe() for c in self.conditions)


class AnyOf(FrozenModel):
    kind: Literal["any"] = "any"
    conditions: tuple[Condition, ...]

    def describe(self) -> str:
        return "(" + " OR ".join(c.describe() for c in self.conditions) + ")"


Condition = Annotated[Predicate | AllOf | AnyOf, Field(discriminator="kind")]
AllOf.model_rebuild()
AnyOf.model_rebuild()


def condition_depth(cond: Condition) -> int:
    if isinstance(cond, Predicate):
        return 1
    return sum(condition_depth(c) for c in cond.conditions)


def condition_features(cond: Condition) -> tuple[str, ...]:
    if isinstance(cond, Predicate):
        return (cond.feature,)
    out: list[str] = []
    for c in cond.conditions:
        out.extend(condition_features(c))
    return tuple(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniverseSnapshot:
    """Set of tradeable symbols as of a date, with honest provenance."""

    as_of_date: date
    symbols: tuple[str, ...]
    source: str
    methodology: str
    is_point_in_time: bool
    limitations: tuple[str, ...]


# ---------------------------------------------------------------------------
# Edge records
# ---------------------------------------------------------------------------


class EdgeIdentity(FrozenModel):
    edge_id: str
    edge_version: int = 1
    name: str
    description: str
    direction: Side
    condition: Condition
    feature_dependencies: tuple[str, ...]
    family: Family
    asset_scope: str = "universe"
    sector_scope: str = "all"
    regime_scope: str = "all"
    entry_rule: str = "next_session_open"
    exit_rule: str = "time_stop"
    holding_horizon: int


class EdgeStats(FrozenModel):
    sample_size: int
    effective_sample_size: float
    gross_mean_return: float
    net_mean_return: float
    median_return: float
    return_std: float
    downside_deviation: float
    probability_of_profit: float
    probability_of_positive_net_return: float
    expected_shortfall: float
    value_at_risk: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float | None = None
    profit_factor: float | None = None
    win_loss_ratio: float | None = None
    turnover: float | None = None
    average_holding_period: float | None = None
    p_value: float
    adjusted_p_value: float
    q_value: float
    bayesian_posterior_probability: float
    bayesian_credible_interval: tuple[float, float]
    bootstrap_confidence_interval: tuple[float, float]
    deflated_sharpe_ratio: float


class EdgeRobustness(FrozenModel):
    stability_score: float = Field(ge=0.0, le=1.0)
    regime_stability_score: float = Field(ge=0.0, le=1.0)
    cost_robustness_score: float = Field(ge=0.0, le=1.0)
    parameter_robustness_score: float = Field(ge=0.0, le=1.0)
    out_of_sample_score: float = Field(ge=0.0, le=1.0)
    decay_score: float = Field(ge=0.0, le=1.0)
    cost_scenario_survival: dict[str, bool] = Field(default_factory=dict)
    regime_performance: dict[str, float] = Field(default_factory=dict)
    fold_net_means: tuple[float, ...] = ()


class EdgeLifecycle(FrozenModel):
    status: EdgeStatus
    discovery_start: date
    discovery_end: date
    validation_start: date
    validation_end: date
    test_start: date | None = None
    test_end: date | None = None
    last_validated_at: datetime | None = None
    failure_reasons: tuple[str, ...] = ()
    discovery_batch_id: str
    experiment_id: str


class Edge(FrozenModel):
    identity: EdgeIdentity
    stats: EdgeStats
    robustness: EdgeRobustness
    lifecycle: EdgeLifecycle


class CandidateEdge(FrozenModel):
    """A rule evaluated during discovery, before independent validation.

    Every evaluated rule is persisted — including rejected ones — so the
    batch trial count used for FDR and deflated-Sharpe is honest.
    """

    candidate_id: str
    discovery_batch_id: str
    experiment_id: str
    name: str
    direction: Side
    condition: Condition
    family: Family
    horizon: int
    in_sample_n: int
    in_sample_gross_mean: float
    in_sample_net_mean: float
    in_sample_hit_rate: float
    in_sample_p_value: float


# ---------------------------------------------------------------------------
# Signal / ranking records (spec sections 25 and 43)
# ---------------------------------------------------------------------------


class EntryPlan(FrozenModel):
    method: EntryMethod
    earliest_timestamp: datetime
    ideal_low: float | None = None
    ideal_high: float | None = None
    do_not_chase_above: float | None = None
    entry_expiration_sessions: int | None = None


class RiskPlan(FrozenModel):
    stop_price: float
    target_1: float
    target_2: float | None = None
    reward_to_risk: float


class RegimeContext(FrozenModel):
    market: str
    volatility: str
    stock_trend: str
    similarity_score: float = Field(ge=0.0, le=1.0)


class EvidenceItem(FrozenModel):
    edge: str
    family: Family
    contribution: float
    validated_sample_size: int
    q_value: float
    net_mean_return: float


class SignalCandidate(FrozenModel):
    as_of_date: date
    symbol: str
    side: Side
    status: CandidateStatus
    conviction_score: float = Field(ge=0.0, le=100.0)
    calibrated_probability_of_positive_net_return: float = Field(ge=0.0, le=1.0)
    probability_is_calibrated: bool
    expected_gross_return: float
    expected_net_return: float
    expected_volatility: float
    expected_shortfall_95: float
    recommended_holding_sessions: int
    entry: EntryPlan
    risk: RiskPlan
    regime: RegimeContext
    evidence: tuple[EvidenceItem, ...]
    warnings: tuple[str, ...] = ()
    explanation: str = ""
    liquidity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    data_quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    model_version: str = ""


class Abstention(FrozenModel):
    as_of_date: date
    symbol: str
    side: Side | None
    reasons: tuple[str, ...]


class SignalReport(FrozenModel):
    """Full machine-readable output of one analysis date."""

    as_of_date: date
    generated_at: datetime
    config_hash: str
    long_candidates: tuple[SignalCandidate, ...]
    short_candidates: tuple[SignalCandidate, ...]
    abstentions: tuple[Abstention, ...]
    universe_warnings: tuple[str, ...]
    disclaimer: str = (
        "Research output only. Not investment advice. Past patterns do not "
        "guarantee future results."
    )
