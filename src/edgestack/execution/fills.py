"""Fill simulation with hard execution-realism invariants.

Rules (asserted, not just tested):

- every fill price lies within the session's [low, high];
- a stop gapped through at the open fills AT THE OPEN, not the stop price;
- a limit order never fills at a price worse than its limit, and never fills
  when the bar's range didn't touch it;
- friction is charged as an explicit cash cost per leg (spread + slippage +
  impact via the CostModel), keeping fill prices verifiable against bars.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from edgestack.exceptions import ExecutionModelError
from edgestack.execution.costs import CostModel
from edgestack.execution.orders import Fill, Order, OrderStatus, OrderType

#: Fraction of a session's volume one order may consume.
MAX_PARTICIPATION = 0.10


@dataclass(frozen=True)
class Bar:
    session: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class FillSimulator:
    def __init__(self, cost_model: CostModel) -> None:
        self.cost_model = cost_model

    def try_fill(self, order: Order, bar: Bar, *, at_open_phase: bool) -> Fill | None:
        """Attempt to fill ``order`` against ``bar``.

        ``at_open_phase`` distinguishes the opening auction (MOO + gap logic)
        from the intraday/close phase (LIMIT/STOP touches, MOC).
        """
        price = self._execution_price(order, bar, at_open_phase)
        if price is None:
            return None

        if not (bar.low - 1e-9 <= price <= bar.high + 1e-9):
            raise ExecutionModelError(
                f"fill price {price} outside session range [{bar.low}, {bar.high}] "
                f"for order {order.order_id} ({order.order_type})"
            )

        quantity = order.quantity
        if bar.volume > 0:
            cap = MAX_PARTICIPATION * bar.volume
            if abs(quantity) > cap:
                quantity = cap if quantity > 0 else -cap
        participation = abs(quantity) / bar.volume if bar.volume > 0 else 1.0

        notional = abs(quantity) * price
        cost_bps = self.cost_model.one_way_cost_bps(
            is_sell=quantity < 0, participation=participation
        )
        order.status = (
            OrderStatus.FILLED
            if abs(quantity) >= abs(order.quantity) - 1e-9
            else OrderStatus.PARTIAL
        )
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            session=bar.session,
            quantity=quantity,
            price=price,
            cost=notional * cost_bps / 1e4,
            tag=order.tag,
        )

    def _execution_price(self, order: Order, bar: Bar, at_open_phase: bool) -> float | None:
        kind = order.order_type
        if kind is OrderType.MARKET_ON_OPEN:
            return bar.open if at_open_phase else None
        if kind is OrderType.MARKET_ON_CLOSE:
            return None if at_open_phase else bar.close

        if kind is OrderType.LIMIT:
            limit = order.limit_price
            if limit is None:
                raise ExecutionModelError("LIMIT order without limit_price")
            if order.is_buy:
                if bar.open <= limit:
                    return bar.open  # opened at or below the limit
                if bar.low <= limit:
                    return limit  # touched intraday
            else:
                if bar.open >= limit:
                    return bar.open
                if bar.high >= limit:
                    return limit
            return None

        if kind is OrderType.STOP:
            stop = order.stop_price
            if stop is None:
                raise ExecutionModelError("STOP order without stop_price")
            if order.is_buy:  # buy stop (covers a short)
                if bar.open >= stop:
                    return bar.open  # gapped through: worse than stop
                if bar.high >= stop:
                    return stop
            else:  # sell stop (protects a long)
                if bar.open <= stop:
                    return bar.open  # gapped through: worse than stop
                if bar.low <= stop:
                    return stop
            return None

        raise ExecutionModelError(f"unsupported order type: {kind}")
