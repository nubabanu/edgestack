"""Portfolio accounting: cash, positions, mark-to-market, borrow accrual."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from edgestack.execution.costs import SESSIONS_PER_YEAR, CostModel
from edgestack.execution.orders import Fill


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0          # negative = short
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    entry_session: pd.Timestamp | None = None
    sessions_held: int = 0

    @property
    def is_open(self) -> bool:
        return abs(self.quantity) > 1e-9


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    def position(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position(symbol=symbol))

    def apply_fill(self, fill: Fill, session: pd.Timestamp) -> None:
        pos = self.position(fill.symbol)
        self.cash -= fill.quantity * fill.price   # buy consumes, sell releases
        self.cash -= fill.cost                    # friction is explicit cash

        old_qty = pos.quantity
        new_qty = old_qty + fill.quantity
        if old_qty == 0 or (old_qty > 0) == (fill.quantity > 0):
            # opening or adding: weighted average entry
            total = abs(old_qty) + abs(fill.quantity)
            pos.avg_price = (
                (abs(old_qty) * pos.avg_price + abs(fill.quantity) * fill.price) / total
            )
            if old_qty == 0:
                pos.entry_session = session
                pos.sessions_held = 0
        else:
            # reducing or closing: realize P&L on the closed part
            closed = min(abs(old_qty), abs(fill.quantity))
            direction = 1.0 if old_qty > 0 else -1.0
            pos.realized_pnl += direction * closed * (fill.price - pos.avg_price)
            if abs(new_qty) < 1e-9:
                new_qty = 0.0
                pos.entry_session = None
        pos.quantity = new_qty

    def accrue_daily_costs(self, prices: dict[str, float], cost_model: CostModel) -> float:
        """Charge short-borrow fees on open shorts; returns the cash charged."""
        borrow_rate = cost_model.short_borrow_annualized * cost_model.multiplier
        charged = 0.0
        for pos in self.positions.values():
            if pos.quantity < 0 and pos.symbol in prices:
                notional = abs(pos.quantity) * prices[pos.symbol]
                fee = notional * borrow_rate / SESSIONS_PER_YEAR
                self.cash -= fee
                charged += fee
            if pos.is_open:
                pos.sessions_held += 1
        return charged

    def equity(self, prices: dict[str, float]) -> float:
        value = self.cash
        for pos in self.positions.values():
            if pos.is_open and pos.symbol in prices:
                value += pos.quantity * prices[pos.symbol]
        return value

    def gross_exposure(self, prices: dict[str, float]) -> float:
        return sum(
            abs(pos.quantity) * prices.get(pos.symbol, pos.avg_price)
            for pos in self.positions.values() if pos.is_open
        )

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.is_open]
