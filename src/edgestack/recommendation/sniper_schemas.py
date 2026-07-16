"""Typed contracts for the loss-aversion-first sniper research workflow."""

from __future__ import annotations

import enum
from datetime import date, datetime
from typing import Literal

from pydantic import Field, model_validator

from edgestack.recommendation.schemas import EvidenceGrade, V2Model


class SniperStrategyId(enum.StrEnum):
    C1_C2_PRIMARY = "C1_C2_PRIMARY"
    C1_RSI2_DIP = "C1_RSI2_DIP"
    C2_THREE_DOWN = "C2_THREE_DOWN"
    A1_SANTA = "A1_SANTA"
    A2_PRE_FOMC = "A2_PRE_FOMC"
    E1_TLT_MONTH_END = "E1_TLT_MONTH_END"
    C3_VIX_CONTANGO = "C3_VIX_CONTANGO"
    C4_PUTCALL_BREADTH = "C4_PUTCALL_BREADTH"
    A6_OPEX_TILT = "A6_OPEX_TILT"
    E2_GOLD_JANUARY = "E2_GOLD_JANUARY"
    A3_MINOR_HOLIDAY = "A3_MINOR_HOLIDAY"
    A4_SMALL_CAP_JANUARY = "A4_SMALL_CAP_JANUARY"
    A5_WEEKEND_MONDAY = "A5_WEEKEND_MONDAY"
    B_NAIVE_OVERNIGHT = "B_NAIVE_OVERNIGHT"
    D_SHORT_VOL_INCOME = "D_SHORT_VOL_INCOME"


class SniperStage(enum.IntEnum):
    STAGE_1 = 1
    STAGE_2 = 2
    STAGE_3 = 3
    EXCLUDED = 99


class SniperRole(enum.StrEnum):
    PRIMARY_ENGINE = "PRIMARY_ENGINE"
    FIXED_CALENDAR_TRADE = "FIXED_CALENDAR_TRADE"
    DEFINED_RISK_TRADE = "DEFINED_RISK_TRADE"
    MODEST_CROSS_ASSET_TRADE = "MODEST_CROSS_ASSET_TRADE"
    VETO_FILTER = "VETO_FILTER"
    SIZE_CONFIRMATION = "SIZE_CONFIRMATION"
    MICRO_TILT = "MICRO_TILT"
    LOW_CONVICTION_WATCHLIST = "LOW_CONVICTION_WATCHLIST"
    FORBIDDEN = "FORBIDDEN"


class SniperActivation(enum.StrEnum):
    SHADOW_READY = "SHADOW_READY"
    BLOCKED_VALIDATION = "BLOCKED_VALIDATION"
    BLOCKED_DATA = "BLOCKED_DATA"
    FILTER_ONLY = "FILTER_ONLY"
    EXCLUDED = "EXCLUDED"


class SniperCandidateStatus(enum.StrEnum):
    TRIGGERED_SHADOW = "TRIGGERED_SHADOW"
    NOT_TRIGGERED = "NOT_TRIGGERED"
    SCHEDULED = "SCHEDULED"
    VETOED = "VETOED"
    BLOCKED = "BLOCKED"


class OverlayState(enum.StrEnum):
    PASS = "PASS"
    VETO = "VETO"
    CONFIRM = "CONFIRM"
    NEUTRAL = "NEUTRAL"
    UNAVAILABLE = "UNAVAILABLE"


class SniperPolicyItemV2(V2Model):
    rank: int = Field(ge=1)
    strategy_id: SniperStrategyId
    stage: SniperStage
    role: SniperRole
    activation: SniperActivation
    conviction: Literal["HIGHEST", "HIGH", "MEDIUM", "MODEST", "LOW", "REJECTED"]
    rule: str
    reason: str


class SniperSizingV2(V2Model):
    account_equity: float = Field(gt=0)
    max_tolerable_loss: float = Field(gt=0)
    adverse_move_p05: float = Field(lt=0)
    risk_notional: float = Field(ge=0)
    capped_notional: float = Field(ge=0)
    portfolio_weight: float = Field(ge=0, le=1)
    estimated_shares: int = Field(ge=0)
    cap_applied: bool
    resolved_signal_outcomes: int = Field(ge=0)
    estimate_source: str
    warning: str


class SniperOutcomeEvidenceV2(V2Model):
    observations: int = Field(ge=0)
    effective_sample_size: float = Field(ge=0)
    cost_adjusted_win_rate: float | None = Field(default=None, ge=0, le=1)
    mean_net_return: float | None = None
    adverse_move_p05: float | None = Field(default=None, le=0)
    evidence_grade: EvidenceGrade = EvidenceGrade.INSUFFICIENT
    promoted: Literal[False] = False
    warning: str = "Descriptive previously accessed history only; not nested V2 promotion evidence."


class SniperCandidateV2(V2Model):
    strategy_id: SniperStrategyId
    component_triggers: tuple[SniperStrategyId, ...] = ()
    symbol: str
    status: SniperCandidateStatus
    signal_session: date | None = None
    entry_window: str | None = None
    exit_rule: str
    maximum_holding_sessions: int = Field(ge=1)
    sizing: SniperSizingV2 | None = None
    evidence: SniperOutcomeEvidenceV2
    veto_reasons: tuple[str, ...] = ()
    cautions: tuple[str, ...] = ()
    actionable: Literal[False] = False
    paper_only: Literal[True] = True


class SniperOverlayV2(V2Model):
    strategy_id: SniperStrategyId
    state: OverlayState
    value: float | None = None
    threshold: str
    can_initiate: Literal[False] = False
    effect: str
    warning: str | None = None


class SniperPlanV2(V2Model):
    schema_version: Literal[2] = 2
    generated_at: datetime
    session: date
    data_version: str
    artifact_version: str
    policy_version: str
    account_equity: float = Field(gt=0)
    max_tolerable_loss: float = Field(gt=0)
    requested_vehicle: str
    policy_ranking: tuple[SniperPolicyItemV2, ...]
    stage_1_candidates: tuple[SniperCandidateV2, ...]
    stage_2_candidates: tuple[SniperCandidateV2, ...]
    overlays: tuple[SniperOverlayV2, ...]
    excluded_strategy_ids: tuple[SniperStrategyId, ...]
    stage_1_promotion_satisfied: bool = False
    warnings: tuple[str, ...]
    disclaimer: str = (
        "Research and paper-trading shadow plan only. No candidate is a live order or investment "
        "advice. Historical hit rates do not guarantee future wins."
    )

    @model_validator(mode="after")
    def _roles_cannot_bypass_stages(self) -> SniperPlanV2:
        excluded = {
            SniperStrategyId.A3_MINOR_HOLIDAY,
            SniperStrategyId.A4_SMALL_CAP_JANUARY,
            SniperStrategyId.A5_WEEKEND_MONDAY,
            SniperStrategyId.B_NAIVE_OVERNIGHT,
            SniperStrategyId.D_SHORT_VOL_INCOME,
        }
        if set(self.excluded_strategy_ids) != excluded:
            raise ValueError("sniper hard exclusions must remain complete")
        if any(
            candidate.actionable
            for candidate in (*self.stage_1_candidates, *self.stage_2_candidates)
        ):
            raise ValueError("unpromoted sniper candidates cannot be actionable")
        if any(overlay.can_initiate for overlay in self.overlays):
            raise ValueError("sniper overlays cannot initiate")
        return self
