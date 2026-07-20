"""Deterministic event replay and explicit execution-data capability gates.

The three clocks prevent a strategy from observing an exchange event before
the feed received it or before the simulator processed it. Queue and latency
models are refused when the underlying data cannot support them.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edgestack.exceptions import ExecutionModelError
from edgestack.recommendation.hashing import stable_hash


class EventKind(enum.StrEnum):
    BAR = "BAR"
    QUOTE = "QUOTE"
    TRADE = "TRADE"
    BOOK = "BOOK"
    ORDER = "ORDER"
    EXECUTION = "EXECUTION"
    MACRO = "MACRO"
    EDGAR = "EDGAR"
    EARNINGS = "EARNINGS"
    AUCTION_IMBALANCE = "AUCTION_IMBALANCE"
    LULD = "LULD"
    CUSTOM = "CUSTOM"


class ExecutionFidelity(enum.StrEnum):
    BAR = "BAR"
    QUOTE = "QUOTE"
    TRADE = "TRADE"
    L2 = "L2"
    L3 = "L3"


class QueueModel(enum.StrEnum):
    NONE = "NONE"
    ESTIMATED = "ESTIMATED"
    EXACT = "EXACT"


class InstrumentKeyV1(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    venue: str
    symbol: str
    asset_class: str


class MarketEventV1(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    event_id: str
    kind: EventKind
    instrument: InstrumentKeyV1 | None = None
    event_time: datetime
    receive_time: datetime
    process_time: datetime
    sequence: int = Field(ge=0)
    source_hash: str
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_time", "receive_time", "process_time")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event clocks must be timezone aware")
        return value

    @model_validator(mode="after")
    def _causal_clocks(self) -> MarketEventV1:
        if self.event_time > self.receive_time:
            raise ValueError("event_time cannot follow receive_time")
        if self.receive_time > self.process_time:
            raise ValueError("receive_time cannot follow process_time")
        if not self.event_id.strip() or not self.source_hash.strip():
            raise ValueError("event_id and source_hash cannot be empty")
        return self


class ExecutionDataCapabilitiesV1(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    bars: bool = False
    quotes: bool = False
    trades: bool = False
    level2_book: bool = False
    level3_orders: bool = False
    feed_sequence: bool = False
    event_timestamps: bool = True
    receive_timestamps: bool = False
    process_timestamps: bool = False
    auction_imbalance: bool = False
    limitations: tuple[str, ...] = ()


def validate_execution_capabilities(
    capabilities: ExecutionDataCapabilitiesV1,
    *,
    fidelity: ExecutionFidelity,
    queue_model: QueueModel = QueueModel.NONE,
    model_latency: bool = False,
) -> None:
    """Reject a simulation request that is more precise than its evidence."""
    available = {
        ExecutionFidelity.BAR: capabilities.bars,
        ExecutionFidelity.QUOTE: capabilities.quotes,
        ExecutionFidelity.TRADE: capabilities.trades,
        ExecutionFidelity.L2: capabilities.level2_book,
        ExecutionFidelity.L3: capabilities.level3_orders,
    }
    if not available[fidelity]:
        raise ExecutionModelError(
            f"{fidelity.value} execution requested without {fidelity.value} data"
        )
    if queue_model is QueueModel.ESTIMATED and not (
        capabilities.level2_book and capabilities.feed_sequence
    ):
        raise ExecutionModelError(
            "estimated queue position requires sequenced level-2 book updates"
        )
    if queue_model is QueueModel.EXACT and not (
        capabilities.level3_orders and capabilities.feed_sequence
    ):
        raise ExecutionModelError("exact queue position requires sequenced level-3 order events")
    if model_latency and not (capabilities.receive_timestamps and capabilities.process_timestamps):
        raise ExecutionModelError(
            "latency modeling requires recorded receive_time and process_time"
        )


class DeterministicEventReplay:
    """Immutable replay ordered by information availability, then source identity."""

    def __init__(self, events: Iterable[MarketEventV1]) -> None:
        materialized = tuple(events)
        identifiers = [event.event_id for event in materialized]
        if len(set(identifiers)) != len(identifiers):
            raise ExecutionModelError("event replay contains duplicate event_id values")
        self._events = tuple(
            sorted(
                materialized,
                key=lambda event: (
                    event.process_time,
                    event.receive_time,
                    event.event_time,
                    event.sequence,
                    event.event_id,
                ),
            )
        )
        self.content_hash = stable_hash(
            tuple(event.model_dump(mode="json") for event in self._events)
        )

    @property
    def events(self) -> tuple[MarketEventV1, ...]:
        return self._events

    def through(self, process_time: datetime) -> tuple[MarketEventV1, ...]:
        if process_time.tzinfo is None or process_time.utcoffset() is None:
            raise ExecutionModelError("replay cutoff must be timezone aware")
        return tuple(event for event in self._events if event.process_time <= process_time)

    def run(self, handler: Callable[[MarketEventV1], Any]) -> tuple[Any, ...]:
        return tuple(handler(event) for event in self._events)
