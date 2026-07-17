"""Versioned API request contracts."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgestack.recommendation.instrument_schemas import (
    CalendarResolution,
    InstrumentAnalysisV2,
    InstrumentKind,
    TimingHorizon,
)
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
    intended_entry_date: date | None = None
    round_trip_cost_bps: float = Field(default=10.0, ge=0, le=1_000)
    include_news: bool = True

    @model_validator(mode="after")
    def _entry_time_has_timezone(self) -> InstrumentAnalysisRequestV2:
        if self.intended_entry_at is not None and self.intended_entry_date is not None:
            raise ValueError("provide intended_entry_at or intended_entry_date, not both")
        if self.intended_entry_at is not None and self.intended_entry_at.tzinfo is None:
            raise ValueError("intended_entry_at must include a timezone offset")
        return self


class InstrumentRecheckRequestV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    previous_analysis: InstrumentAnalysisV2
    request: InstrumentAnalysisRequestV2


class PatternLeaderRequestV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbols: tuple[str, ...] = (
        "SPY",
        "QQQ",
        "GLD",
        "USO",
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "GOOGL",
        "TSLA",
    )
    resolution: CalendarResolution = CalendarResolution.DAY
    horizon: TimingHorizon = TimingHorizon.WEEK
    limit: int = Field(default=10, ge=1, le=25)

    @model_validator(mode="after")
    def _bounded_unique_symbols(self) -> PatternLeaderRequestV2:
        normalized = tuple(
            dict.fromkeys(item.strip().upper() for item in self.symbols if item.strip())
        )
        if not normalized or len(normalized) > 50:
            raise ValueError("pattern leader scans require 1-50 unique symbols")
        object.__setattr__(self, "symbols", normalized)
        return self
