"""Versioned API request contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from edgestack.recommendation.schemas import RiskProfileV2, RiskStateV2


class RecommendationPreviewRequestV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: RiskProfileV2
    risk_state: RiskStateV2 | None = None
    equity_override: float | None = Field(default=None, gt=0)
    reset_requested: bool = False
