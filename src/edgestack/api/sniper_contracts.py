"""API request contracts for sniper shadow-plan sizing."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SniperPreviewRequestV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_equity: float = Field(default=100_000, gt=0)
    max_tolerable_loss: float = Field(default=250, gt=0)
    vehicle: str = Field(default="SPY", min_length=1, max_length=24)
