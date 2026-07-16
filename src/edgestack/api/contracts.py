"""Versioned API request contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgestack.recommendation.instrument_schemas import InstrumentKind
from edgestack.recommendation.schemas import RiskProfileV2, RiskStateV2


class RecommendationPreviewRequestV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: RiskProfileV2
    risk_state: RiskStateV2 | None = None
    equity_override: float | None = Field(default=None, gt=0)
    reset_requested: bool = False


class InstrumentAnalysisRequestV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1, max_length=24)
    instrument_kind: InstrumentKind | None = None
    intended_entry_at: datetime | None = None
    round_trip_cost_bps: float = Field(default=10.0, ge=0, le=1_000)
    include_news: bool = True

    @model_validator(mode="after")
    def _entry_time_has_timezone(self) -> InstrumentAnalysisRequestV2:
        if self.intended_entry_at is not None and self.intended_entry_at.tzinfo is None:
            raise ValueError("intended_entry_at must include a timezone offset")
        return self
