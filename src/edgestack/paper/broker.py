"""Broker abstraction.

Any future real-broker integration must implement :class:`BrokerAdapter` and
be explicitly enabled — and the bundled configuration refuses
``live_trading_enabled: true`` outright, because this repository ships no
live implementation. Only :class:`SimulatedBroker` exists.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from edgestack.config import EdgeStackConfig
from edgestack.exceptions import LiveTradingDisabledError
from edgestack.execution.costs import CostModel
from edgestack.execution.fills import Bar, FillSimulator
from edgestack.execution.orders import Fill, Order, OrderStatus


@dataclass(frozen=True)
class AccountSnapshot:
    observed_at: datetime
    cash: float
    equity: float
    buying_power: float
    positions: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BrokerClock:
    observed_at: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime


@dataclass(frozen=True)
class OrderPreview:
    symbol: str
    requested_quantity: float
    maximum_fill_quantity: float
    estimated_cost: float
    accepted: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubmittedOrder:
    idempotency_key: str
    order_id: int
    status: OrderStatus


@dataclass(frozen=True)
class ReconciliationReport:
    observed_at: datetime
    matched_order_ids: tuple[int, ...]
    missing_order_ids: tuple[int, ...]
    unexpected_order_ids: tuple[int, ...]

    @property
    def clean(self) -> bool:
        return not self.missing_order_ids and not self.unexpected_order_ids


class BrokerAdapter(abc.ABC):
    """Order-submission surface. Implementations must be explicitly configured."""

    is_live: bool = False

    @abc.abstractmethod
    def account(self) -> AccountSnapshot: ...

    @abc.abstractmethod
    def clock(self, *, now: datetime | None = None) -> BrokerClock: ...

    @abc.abstractmethod
    def preview_order(
        self,
        order: Order,
        *,
        reference_price: float,
        reference_volume: float,
    ) -> OrderPreview: ...

    @abc.abstractmethod
    def submit_order(self, order: Order, *, idempotency_key: str) -> SubmittedOrder: ...

    @abc.abstractmethod
    def cancel_order(self, order_id: int) -> bool: ...

    @abc.abstractmethod
    def reconcile(self, expected_open_order_ids: tuple[int, ...]) -> ReconciliationReport: ...

    @abc.abstractmethod
    def execute_against_bar(
        self, order: Order, bar: Bar, *, at_open_phase: bool
    ) -> Fill | None: ...


class SimulatedBroker(BrokerAdapter):
    """Fills orders against historical/synthetic bars via the FillSimulator."""

    is_live = False

    def __init__(self, cost_model: CostModel, *, initial_cash: float = 100_000.0) -> None:
        self._simulator = FillSimulator(cost_model)
        self._cost_model = cost_model
        self._initial_cash = initial_cash
        self._orders: dict[int, Order] = {}
        self._idempotency: dict[str, int] = {}

    def account(self) -> AccountSnapshot:
        return AccountSnapshot(
            observed_at=datetime.now(UTC),
            cash=self._initial_cash,
            equity=self._initial_cash,
            buying_power=self._initial_cash * 5.0,
        )

    def clock(self, *, now: datetime | None = None) -> BrokerClock:
        eastern = ZoneInfo("America/New_York")
        observed = (now or datetime.now(UTC)).astimezone(eastern)
        session_open = datetime.combine(observed.date(), time(9, 30), eastern)
        session_close = datetime.combine(observed.date(), time(16, 0), eastern)
        is_weekday = observed.weekday() < 5
        is_open = is_weekday and session_open <= observed < session_close
        next_date = observed.date()
        if observed >= session_close or not is_weekday:
            next_date = next_date.fromordinal(next_date.toordinal() + 1)
        while next_date.weekday() >= 5:
            next_date = next_date.fromordinal(next_date.toordinal() + 1)
        next_open = datetime.combine(next_date, time(9, 30), eastern)
        if observed < session_open and is_weekday:
            next_open = session_open
        next_close = (
            session_close if is_open else datetime.combine(next_open.date(), time(16), eastern)
        )
        return BrokerClock(
            observed_at=observed,
            is_open=is_open,
            next_open=next_open,
            next_close=next_close,
        )

    def preview_order(
        self,
        order: Order,
        *,
        reference_price: float,
        reference_volume: float,
    ) -> OrderPreview:
        reasons = []
        if reference_price <= 0:
            reasons.append("reference price must be positive")
        if reference_volume <= 0:
            reasons.append("reference volume must be positive")
        maximum = max(0.0, 0.10 * reference_volume)
        quantity = min(abs(order.quantity), maximum)
        participation = quantity / reference_volume if reference_volume > 0 else 1.0
        cost_bps = self._cost_model.one_way_cost_bps(
            is_sell=order.quantity < 0,
            participation=participation,
        )
        return OrderPreview(
            symbol=order.symbol,
            requested_quantity=order.quantity,
            maximum_fill_quantity=quantity,
            estimated_cost=quantity * max(0.0, reference_price) * cost_bps / 1e4,
            accepted=not reasons,
            reasons=tuple(reasons),
        )

    def submit_order(self, order: Order, *, idempotency_key: str) -> SubmittedOrder:
        if not idempotency_key.strip():
            raise ValueError("idempotency_key is required")
        existing_id = self._idempotency.get(idempotency_key)
        if existing_id is not None:
            existing = self._orders[existing_id]
            return SubmittedOrder(idempotency_key, existing.order_id, existing.status)
        order.status = OrderStatus.ACCEPTED
        self._orders[order.order_id] = order
        self._idempotency[idempotency_key] = order.order_id
        return SubmittedOrder(idempotency_key, order.order_id, order.status)

    def cancel_order(self, order_id: int) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status in {OrderStatus.FILLED, OrderStatus.CANCELLED}:
            return False
        order.status = OrderStatus.CANCELLED
        return True

    def reconcile(self, expected_open_order_ids: tuple[int, ...]) -> ReconciliationReport:
        expected = set(expected_open_order_ids)
        observed = {
            order_id
            for order_id, order in self._orders.items()
            if order.status in {OrderStatus.NEW, OrderStatus.ACCEPTED, OrderStatus.PARTIAL}
        }
        return ReconciliationReport(
            observed_at=datetime.now(UTC),
            matched_order_ids=tuple(sorted(expected & observed)),
            missing_order_ids=tuple(sorted(expected - observed)),
            unexpected_order_ids=tuple(sorted(observed - expected)),
        )

    def execute_against_bar(self, order: Order, bar: Bar, *, at_open_phase: bool) -> Fill | None:
        fill = self._simulator.try_fill(order, bar, at_open_phase=at_open_phase)
        if order.order_id in self._orders:
            self._orders[order.order_id] = order
        return fill


def get_broker(cfg: EdgeStackConfig) -> BrokerAdapter:
    """The only broker EdgeStack will hand out is the simulated one."""
    if cfg.paper.live_trading_enabled:  # unreachable: config validation refuses it
        raise LiveTradingDisabledError("live trading is not implemented in EdgeStack")
    return SimulatedBroker(CostModel.from_config(cfg), initial_cash=cfg.paper.initial_cash)
