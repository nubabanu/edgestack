"""Trade, order and equity ledgers — the replayable audit trail of a run."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from edgestack.execution.orders import Fill, Order


@dataclass
class Ledger:
    fills: list[Fill] = field(default_factory=list)
    order_rows: list[dict] = field(default_factory=list)
    equity_rows: list[dict] = field(default_factory=list)
    trade_rows: list[dict] = field(default_factory=list)

    def record_order(self, order: Order, session: pd.Timestamp) -> None:
        self.order_rows.append(
            {
                "order_id": order.order_id,
                "session": session,
                "symbol": order.symbol,
                "quantity": order.quantity,
                "type": order.order_type.value,
                "status": order.status.value,
                "limit_price": order.limit_price,
                "stop_price": order.stop_price,
                "tag": order.tag,
            }
        )

    def record_fill(self, fill: Fill) -> None:
        self.fills.append(fill)

    def record_equity(
        self,
        session: pd.Timestamp,
        cash: float,
        equity: float,
        gross: float,
        n_positions: int,
        borrow_paid: float,
    ) -> None:
        self.equity_rows.append(
            {
                "session": session,
                "cash": cash,
                "equity": equity,
                "gross_exposure": gross,
                "n_positions": n_positions,
                "borrow_paid": borrow_paid,
            }
        )

    def record_trade(
        self,
        *,
        symbol: str,
        side: str,
        entry_session: pd.Timestamp,
        exit_session: pd.Timestamp,
        entry_price: float,
        exit_price: float,
        quantity: float,
        net_pnl: float,
        exit_reason: str,
    ) -> None:
        self.trade_rows.append(
            {
                "symbol": symbol,
                "side": side,
                "entry_session": entry_session,
                "exit_session": exit_session,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "net_pnl": net_pnl,
                "exit_reason": exit_reason,
            }
        )

    def equity_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_rows)

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_rows)

    def orders_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.order_rows)

    def fills_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.__dict__ for f in self.fills])
