"""Order and fill records for the session-driven backtester."""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field

import pandas as pd


class OrderType(enum.StrEnum):
    MARKET_ON_OPEN = "MARKET_ON_OPEN"
    MARKET_ON_CLOSE = "MARKET_ON_CLOSE"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(enum.StrEnum):
    NEW = "NEW"
    ACCEPTED = "ACCEPTED"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"


_ids = itertools.count(1)


@dataclass
class Order:
    symbol: str
    quantity: float  # positive = buy, negative = sell
    order_type: OrderType
    created_session: pd.Timestamp
    limit_price: float | None = None
    stop_price: float | None = None
    expires_after_sessions: int | None = None
    tag: str = ""  # entry / stop_loss / target / time_exit
    order_id: int = field(default_factory=lambda: next(_ids))
    status: OrderStatus = OrderStatus.NEW
    sessions_open: int = 0

    @property
    def is_buy(self) -> bool:
        return self.quantity > 0


@dataclass(frozen=True)
class Fill:
    order_id: int
    symbol: str
    session: pd.Timestamp
    quantity: float  # signed like the order
    price: float  # execution price, always within the bar range
    cost: float  # explicit cash cost of the leg (>= 0)
    tag: str
