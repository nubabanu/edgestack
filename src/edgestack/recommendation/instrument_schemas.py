"""Public V2 contracts for user-selected instrument analysis.

Instrument analysis is deliberately separate from portfolio selection.  It may
describe unpromoted observations, but only a frozen, promoted timing artifact
can make an entry/exit window actionable.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from edgestack.recommendation.schemas import EvidenceGrade, V2Model


class InstrumentKind(enum.StrEnum):
    STOCK = "STOCK"
    ETF = "ETF"
    COMMODITY_PROXY = "COMMODITY_PROXY"


class TimingHorizon(enum.StrEnum):
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    YEAR = "YEAR"


class InstrumentAnalysisStatus(enum.StrEnum):
    ACTIONABLE = "ACTIONABLE"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class DirectionalRating(enum.StrEnum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    NOT_RATED = "NOT_RATED"


class EffectDirection(enum.StrEnum):
    TAILWIND = "TAILWIND"
    HEADWIND = "HEADWIND"
    MIXED = "MIXED"
    NEUTRAL = "NEUTRAL"


class CalendarResolution(enum.StrEnum):
    MINUTE_15 = "MINUTE_15"
    HOUR = "HOUR"
    DAY = "DAY"
    MONTH = "MONTH"
    YEAR = "YEAR"


class ChoiceRating(enum.StrEnum):
    STRONG = "STRONG"
    ABOVE_AVERAGE = "ABOVE_AVERAGE"
    AVERAGE = "AVERAGE"
    BELOW_AVERAGE = "BELOW_AVERAGE"
    WEAK = "WEAK"
    NOT_RATED = "NOT_RATED"


class InstrumentResolutionV2(V2Model):
    requested_symbol: str
    resolved_symbol: str
    instrument_kind: InstrumentKind
    proxy_for: str | None = None
    tradeable_instrument: bool = True
    notes: tuple[str, ...] = ()


class EdgeEffectV2(V2Model):
    edge_id: str
    family: str
    horizon_sessions: int = Field(ge=1)
    direction: EffectDirection
    observation: str
    positive_contribution: float = Field(default=0.0, ge=0)
    negative_contribution: float = Field(default=0.0, le=0)
    net_contribution: float = 0.0
    protective_avoidance_value: float = Field(default=0.0, ge=0)
    lower_95: float | None = None
    adverse_counter_effect: str
    protective_counter_effect: str
    evidence_grade: EvidenceGrade
    compound: bool = False
    promoted: bool = False
    artifact_hash: str | None = None
    invalidation: str

    @model_validator(mode="after")
    def _contributions_reconcile(self) -> EdgeEffectV2:
        expected = self.positive_contribution + self.negative_contribution
        if abs(expected - self.net_contribution) > 1e-10:
            raise ValueError("edge positive and negative contributions must reconcile to net")
        if self.promoted and self.evidence_grade is not EvidenceGrade.PROMOTED:
            raise ValueError("a promoted edge requires PROMOTED evidence")
        return self


class TimingWindowV2(V2Model):
    horizon: TimingHorizon
    label: str
    entry_window: str
    exit_window: str
    holding_sessions: int = Field(ge=1)
    expected_net_return: float | None = None
    lower_95: float | None = None
    upper_95: float | None = None
    multiple_testing_adjusted_pvalue: float | None = Field(default=None, ge=0, le=1)
    observations: int = Field(default=0, ge=0)
    effective_sample_size: float = Field(default=0.0, ge=0)
    evidence_grade: EvidenceGrade
    actionable: bool = False
    artifact_hash: str | None = None
    rationale: str
    what_invalidates_it: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _actionable_requires_promotion(self) -> TimingWindowV2:
        if self.actionable and (
            self.evidence_grade is not EvidenceGrade.PROMOTED or not self.artifact_hash
        ):
            raise ValueError("actionable timing requires a frozen promoted artifact")
        return self


class WinScoreV2(V2Model):
    """Transparent descriptive score; never a guaranteed win probability."""

    net_win_rate: float = Field(ge=0, le=1)
    shrunk_win_rate: float = Field(ge=0, le=1)
    win_score: float = Field(ge=0, le=100)
    expected_net_return: float
    lower_95: float
    observations: int = Field(ge=0)
    effective_sample_size: float = Field(ge=0)
    rank: int = Field(ge=1)
    candidates_ranked: int = Field(ge=1)
    multiple_testing_adjusted_pvalue: float = Field(ge=0, le=1)
    evidence_grade: EvidenceGrade
    actionable: bool = False
    explanation: str = (
        "Cost-adjusted historical win frequency shrunk toward 50% and confidence-weighted; "
        "not a forecast or guarantee."
    )

    @model_validator(mode="after")
    def _actionable_score_requires_promotion(self) -> WinScoreV2:
        if self.actionable and self.evidence_grade is not EvidenceGrade.PROMOTED:
            raise ValueError("an actionable win score requires promoted evidence")
        return self


class TailwindCalendarCellV2(V2Model):
    slot_key: str
    display_label: str
    entry_window: str
    exit_window: str
    horizon: TimingHorizon
    score: WinScoreV2
    rank_percentile: float = Field(ge=0, le=1)


class TailwindCalendarV2(V2Model):
    resolution: CalendarResolution
    timezone: str
    horizon: TimingHorizon
    data_start: datetime | None = None
    data_end: datetime | None = None
    cells: tuple[TailwindCalendarCellV2, ...] = ()
    warning: str


class ChosenTimeRatingV2(V2Model):
    resolution: CalendarResolution
    horizon: TimingHorizon
    requested_time: datetime
    matched_slot: str | None = None
    rating: ChoiceRating
    score: WinScoreV2 | None = None
    better_alternative: TailwindCalendarCellV2 | None = None
    score_improvement: float | None = Field(default=None, ge=0)
    recommendation: str
    actionable: bool = False

    @model_validator(mode="after")
    def _choice_actionability(self) -> ChosenTimeRatingV2:
        if self.actionable and (self.score is None or not self.score.actionable):
            raise ValueError("an actionable choice requires an actionable score")
        return self


class ExitPlanV2(V2Model):
    horizon: TimingHorizon
    entry_slot: str
    preferred_exit: str | None = None
    holding_sessions: int = Field(ge=0)
    data_resolution: CalendarResolution
    score: WinScoreV2 | None = None
    alternatives: tuple[TailwindCalendarCellV2, ...] = ()
    actionable: bool = False
    rationale: str
    warning: str | None = None


class RecheckPlanV2(V2Model):
    enabled: bool
    intended_entry_at: datetime | None = None
    next_check_at: datetime | None = None
    cadence_minutes: int | None = Field(default=None, ge=15)
    required_resolution: CalendarResolution | None = None
    automatic_recheck_supported: bool = True
    reason: str


class HorizonTimingAnalysisV2(V2Model):
    horizon: TimingHorizon
    data_resolution: str
    best_window: TimingWindowV2 | None = None
    worst_window: TimingWindowV2 | None = None
    alternatives: tuple[TimingWindowV2, ...] = ()
    intended_entry_assessment: str | None = None
    actionable: bool = False
    searched_variants: int = Field(default=0, ge=0)
    warning: str | None = None

    @model_validator(mode="after")
    def _horizon_actionability(self) -> HorizonTimingAnalysisV2:
        if self.actionable and (self.best_window is None or not self.best_window.actionable):
            raise ValueError("an actionable horizon requires an actionable best window")
        return self


class NewsEvidenceV2(V2Model):
    news_id: str
    symbol: str
    headline: str
    source: str
    published_at: datetime
    url: str | None = None
    summary: str = ""
    sentiment_label: Literal["POSITIVE", "NEGATIVE", "MIXED", "NEUTRAL", "UNSCORED"] = "UNSCORED"
    relevance: float | None = Field(default=None, ge=0, le=1)
    is_fresh: bool
    age_hours: float = Field(ge=0)
    actionable_contribution: float = Field(default=0.0, ge=0, le=0)
    warning: str = "News is context only unless a promoted artifact explicitly validates it."


class FrozenTimingArtifactV2(V2Model):
    """A live timing rule that already passed the V2 promotion process."""

    schema_version: Literal[2] = 2
    artifact_hash: str
    sleeve_id: str
    symbol: str
    horizon: TimingHorizon
    compound: bool = False
    component_artifact_hashes: tuple[str, ...] = ()
    label: str
    entry_window: str
    exit_window: str
    holding_sessions: int = Field(ge=1)
    expected_net_return: float
    incremental_expected_net_return: float
    lower_95: float
    incremental_lower_95: float
    upper_95: float
    multiple_testing_adjusted_pvalue: float = Field(ge=0, le=0.05)
    observations: int = Field(ge=1)
    effective_sample_size: float = Field(ge=1)
    data_version: str
    artifact_version: str
    policy_version: str
    evidence_grade: Literal[EvidenceGrade.PROMOTED] = EvidenceGrade.PROMOTED
    complete_trial_family_hash: str
    selection_frozen_inside_outer_training: Literal[True] = True
    ablations_passed: Literal[True] = True
    stress_scenarios_passed: tuple[str, ...]
    what_invalidates_it: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _incremental_evidence_is_positive(self) -> FrozenTimingArtifactV2:
        if self.lower_95 <= 0 or self.incremental_lower_95 <= 0:
            raise ValueError("promoted timing requires positive total and incremental lower bounds")
        if self.compound and not self.component_artifact_hashes:
            raise ValueError("a promoted compound timing artifact requires frozen components")
        return self


class AlignmentSummaryV2(V2Model):
    aligned_trade: bool
    promoted_tailwinds: int = Field(ge=0)
    promoted_headwinds: int = Field(ge=0)
    observational_tailwinds: int = Field(ge=0)
    observational_headwinds: int = Field(ge=0)
    actionable_horizons: tuple[TimingHorizon, ...] = ()
    missing_inputs: tuple[str, ...] = ()
    explanation: str


class InstrumentAnalysisV2(V2Model):
    schema_version: Literal[2] = 2
    analysis_id: str
    resolution: InstrumentResolutionV2
    as_of: datetime
    intended_entry_at: datetime | None = None
    data_version: str
    artifact_version: str
    policy_version: str
    current_price: float | None = Field(default=None, gt=0)
    status: InstrumentAnalysisStatus
    overall_rating: DirectionalRating
    overall_score: float | None = Field(default=None, ge=-100, le=100)
    canonical_portfolio_weight: float = 0.0
    alignment: AlignmentSummaryV2
    horizon_analyses: tuple[HorizonTimingAnalysisV2, ...]
    chosen_time_ratings: tuple[ChosenTimeRatingV2, ...] = ()
    exit_plans: tuple[ExitPlanV2, ...] = ()
    tailwind_calendars: tuple[TailwindCalendarV2, ...] = ()
    recheck_plan: RecheckPlanV2 = RecheckPlanV2(
        enabled=False, reason="No intended entry time was supplied."
    )
    tailwinds: tuple[EdgeEffectV2, ...] = ()
    headwinds: tuple[EdgeEffectV2, ...] = ()
    mixed_effects: tuple[EdgeEffectV2, ...] = ()
    news: tuple[NewsEvidenceV2, ...] = ()
    what_to_watch: tuple[str, ...] = ()
    current_year_notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    disclaimer: str = (
        "Research and paper-trading output only. Timing estimates are uncertain, "
        "not guarantees or investment advice."
    )

    @model_validator(mode="after")
    def _rating_requires_promoted_evidence(self) -> InstrumentAnalysisV2:
        promoted = any(effect.promoted for effect in (*self.tailwinds, *self.headwinds))
        if self.overall_rating is not DirectionalRating.NOT_RATED and not promoted:
            raise ValueError("a directional rating requires promoted evidence")
        if self.status is InstrumentAnalysisStatus.ACTIONABLE and not self.alignment.aligned_trade:
            raise ValueError("ACTIONABLE requires an all-promoted aligned setup")
        return self


class InstrumentRecheckV2(V2Model):
    schema_version: Literal[2] = 2
    previous_analysis_id: str
    analysis: InstrumentAnalysisV2
    recommendation_still_holds: bool
    better_alternative_emerged: bool
    changes: tuple[str, ...] = ()


class PatternLeaderV2(V2Model):
    rank: int = Field(ge=1)
    symbol: str
    resolution: CalendarResolution
    horizon: TimingHorizon
    strongest_slot: TailwindCalendarCellV2
    actionable: Literal[False] = False


class PatternLeaderBoardV2(V2Model):
    schema_version: Literal[2] = 2
    as_of: datetime
    resolution: CalendarResolution
    horizon: TimingHorizon
    searched_symbols: tuple[str, ...]
    skipped_symbols: tuple[str, ...] = ()
    searched_cells: int = Field(ge=0)
    leaders: tuple[PatternLeaderV2, ...]
    warning: str = (
        "Cross-instrument historical pattern ranking is watchlist research only. The symbol/slot "
        "search expands the multiple-testing family and cannot promote or authorize a trade."
    )
