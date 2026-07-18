"""Paper-only broker-aware oil decision contracts.

These contracts deliberately contain no order, quantity, notional, or live-sizing
fields.  The oil workflow can describe evidence and model losses, but it can never
authorize or place a trade.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from edgestack.recommendation.instrument_schemas import InstrumentAnalysisV2
from edgestack.recommendation.schemas import V2Model


class OilDecisionStatus(enum.StrEnum):
    BLOCKED = "BLOCKED"
    OBSERVE = "OBSERVE"
    PAPER_ONLY = "PAPER_ONLY"


class OilEventFlag(enum.StrEnum):
    WEEKEND_SUPPLY_ESCALATION = "WEEKEND_SUPPLY_ESCALATION"
    SHIPPING_DISRUPTION = "SHIPPING_DISRUPTION"
    WTI_ROLLOVER_EXPIRY = "WTI_ROLLOVER_EXPIRY"
    BROKER_MAINTENANCE = "BROKER_MAINTENANCE"


class OilBrokerQuoteV2(V2Model):
    observed_at: datetime
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    offered_leverage: float = Field(default=10.0, ge=1, le=10)

    @model_validator(mode="after")
    def _valid_quote(self) -> OilBrokerQuoteV2:
        if self.observed_at.tzinfo is None:
            raise ValueError("oil quote observed_at must include a timezone offset")
        if self.ask <= self.bid:
            raise ValueError("oil quote ask must be greater than bid")
        return self

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_bps(self) -> float:
        return (self.ask - self.bid) / self.midpoint * 10_000.0


class OilBrokerProfileV2(V2Model):
    broker: Literal["ETORO"] = "ETORO"
    broker_symbol: Literal["OIL"] = "OIL"
    product_type: Literal["NON_EXPIRING_CFD"] = "NON_EXPIRING_CFD"
    max_modeled_leverage: Literal[10] = 10
    market_timezone: Literal["GMT"] = "GMT"
    weekly_session: str = "Sunday 22:00 through Friday 20:30"
    daily_break: str = "21:00-22:00"
    overnight_fee_cutoff: str = "21:00 GMT (22:00 during UK daylight saving time)"
    weekend_fee_timing: str = "Oil weekend fee is charged on Friday"
    rollover_warning: str = (
        "The CFD is non-expiring, but its underlying oil reference and broker financing "
        "can still roll or change; confirm the broker notice manually."
    )
    specification_url: str = "https://www.etoro.com/trading/market-hours-and-events/"
    fee_url: str = "https://www.etoro.com/trading/fees/cfd-overnight-fees/"


class OilDecisionRequestV2(V2Model):
    broker_symbol: Literal["OIL"] = "OIL"
    intended_entry_at: datetime
    quote: OilBrokerQuoteV2
    modeled_leverage: float = Field(default=10.0, ge=1, le=10)
    event_flags: tuple[OilEventFlag, ...] = ()
    include_news: bool = True

    @model_validator(mode="after")
    def _entry_has_timezone(self) -> OilDecisionRequestV2:
        if self.intended_entry_at.tzinfo is None:
            raise ValueError("intended_entry_at must include a timezone offset")
        if self.modeled_leverage > self.quote.offered_leverage:
            raise ValueError("modeled leverage cannot exceed the broker quote's offered leverage")
        if len(set(self.event_flags)) != len(self.event_flags):
            raise ValueError("oil event flags must be unique")
        return self


class OilSourceObservationV2(V2Model):
    symbol: str
    role: str
    available: bool
    daily_through: datetime | None = None
    hourly_through: datetime | None = None
    fifteen_minute_through: datetime | None = None
    input_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    warning: str | None = None


class OilDataFreshnessV2(V2Model):
    canonical_matches_catalog: bool
    bundle_as_of: datetime
    quote_age_seconds: float = Field(ge=0)
    quote_fresh: bool
    all_required_sources_present: bool
    required_sources_fresh: bool
    sources: tuple[OilSourceObservationV2, ...]
    warnings: tuple[str, ...] = ()


class OilFrictionScenarioV2(V2Model):
    name: Literal["LOW", "BASE", "STRESS"]
    round_trip_cost_bps: float = Field(ge=0)
    matched_slot: str | None = None
    expected_net_return: float | None = None
    lower_95: float | None = None
    multiple_testing_adjusted_pvalue: float | None = Field(default=None, ge=0, le=1)
    observations: int = Field(default=0, ge=0)
    survives: bool = False
    warning: str


class OilEventVetoV2(V2Model):
    code: str
    label: str
    active: bool
    hard_veto: bool
    source: str


class OilSourceAlignmentV2(V2Model):
    state: Literal["ALIGNED", "MIXED", "UNAVAILABLE"]
    observations: tuple[str, ...]
    directional_contribution: Literal[0] = 0
    warning: str = "Cross-market alignment is context only and cannot initiate a trade."


class OilStressPointV2(V2Model):
    leverage: Literal[1, 5, 10]
    adverse_move_fraction: float = Field(gt=0, le=0.1)
    equity_loss_fraction: float = Field(ge=0)
    catastrophic: bool
    liquidation_possible: bool
    warning: str

    @model_validator(mode="after")
    def _fixed_stress_grid(self) -> OilStressPointV2:
        if self.adverse_move_fraction not in {0.01, 0.05, 0.1}:
            raise ValueError("oil stress moves must be 1%, 5%, or 10%")
        return self


class OilDecisionSnapshotV2(V2Model):
    schema_version: Literal[2] = 2
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: datetime
    broker: Literal["ETORO"] = "ETORO"
    broker_symbol: Literal["OIL"] = "OIL"
    product_description: str
    broker_profile: OilBrokerProfileV2
    canonical_bundle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_portfolio_weight: float = Field(default=0.0, ge=0, le=0)
    actionable: Literal[False] = False
    status: OilDecisionStatus
    intended_entry_at: datetime
    quote: OilBrokerQuoteV2
    quote_input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    modeled_leverage_cap: float = Field(ge=1, le=10)
    decision_reasons: tuple[str, ...]
    hard_block_reasons: tuple[str, ...]
    analysis: InstrumentAnalysisV2
    data_freshness: OilDataFreshnessV2
    friction_sensitivity: tuple[OilFrictionScenarioV2, ...]
    event_vetoes: tuple[OilEventVetoV2, ...]
    source_alignment: OilSourceAlignmentV2
    stress_table: tuple[OilStressPointV2, ...]
    next_recheck_at: datetime | None = None
    manual_inputs_required: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    disclaimer: str = (
        "Research and paper-trading output only. No live order, position size, or "
        "investment recommendation is produced."
    )

    @model_validator(mode="after")
    def _paper_only_invariants(self) -> OilDecisionSnapshotV2:
        if self.canonical_portfolio_weight != 0:
            raise ValueError("oil decision canonical weight must remain zero")
        if self.analysis.canonical_portfolio_weight != 0:
            raise ValueError("oil decision analysis must retain zero canonical weight")
        if self.status is OilDecisionStatus.BLOCKED and not self.hard_block_reasons:
            raise ValueError("blocked oil decisions require at least one hard-block reason")
        if self.status is not OilDecisionStatus.BLOCKED and self.hard_block_reasons:
            raise ValueError("non-blocked oil decisions cannot contain hard-block reasons")
        if len(self.friction_sensitivity) != 3:
            raise ValueError("oil decisions require low/base/stress friction scenarios")
        if len(self.stress_table) != 9:
            raise ValueError("oil decisions require the complete 3x3 stress table")
        return self
