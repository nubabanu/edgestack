"""Session-driven backtest engine.

Daily bars make sessions the only event, so the loop is explicit:

    for each session:
        1. OPEN phase: protective stops / targets checked against the open
           (gaps fill at the open), scheduled market-on-open entries and
           time exits fill at the open;
        2. INTRADAY phase: stop and target touches inside the bar — when a
           bar touches both, the STOP is assumed to hit first (conservative);
        3. CLOSE phase: market-on-close orders;
        4. accrue borrow on shorts, mark to market, record equity;
        5. AFTER the close: intents signaled at this session's close become
           next-session market-on-open entries (signals can never execute at
           the close that produced them).

The engine replays *trade intents* (symbol, signal session, side, horizon,
stop, target) so it stays decoupled from how signals were produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from edgestack.backtest.ledger import Ledger
from edgestack.backtest.portfolio import PortfolioState
from edgestack.config import EdgeStackConfig
from edgestack.exceptions import DataError
from edgestack.execution.costs import CostModel
from edgestack.execution.fills import Bar, FillSimulator
from edgestack.execution.orders import Fill, Order, OrderStatus, OrderType
from edgestack.types import CostScenario, Side


@dataclass(frozen=True)
class TradeIntent:
    symbol: str
    signal_session: pd.Timestamp
    side: Side
    horizon: int
    stop_price: float
    target_price: float


@dataclass
class _OpenTrade:
    side: Side
    stop_price: float
    target_price: float
    exit_session_pos: int  # position index of the scheduled time exit
    entry_price: float = 0.0
    entry_session: pd.Timestamp | None = None
    quantity: float = 0.0
    entry_cost: float = 0.0  # friction paid on the entry leg


class BacktestEngine:
    def __init__(
        self, panel: pd.DataFrame, cfg: EdgeStackConfig, scenario: CostScenario | None = None
    ) -> None:
        self.cfg = cfg
        self.cost_model = CostModel.from_config(cfg, scenario)
        self.simulator = FillSimulator(self.cost_model)
        self.sessions = sorted(pd.to_datetime(panel["date"]).unique())
        self.session_pos = {s: i for i, s in enumerate(self.sessions)}
        self.bars: dict[pd.Timestamp, dict[str, Bar]] = {}
        for row in panel.itertuples(index=False):
            record = cast(Any, row)
            session = pd.Timestamp(record.date)
            symbol = str(record.symbol)
            bar = Bar(
                session=session,
                open=float(record.open),
                high=float(record.high),
                low=float(record.low),
                close=float(record.close),
                volume=float(record.volume),
            )
            self.bars.setdefault(session, {})[symbol] = bar

    def run(self, intents: list[TradeIntent], initial_cash: float = 100_000.0) -> Ledger:
        if not self.sessions:
            raise DataError("empty panel")
        intents_by_session: dict[pd.Timestamp, list[TradeIntent]] = {}
        for intent in intents:
            if intent.signal_session in self.session_pos:
                intents_by_session.setdefault(intent.signal_session, []).append(intent)

        ledger = Ledger()
        portfolio = PortfolioState(cash=initial_cash)
        open_trades: dict[str, _OpenTrade] = {}
        pending_entries: list[tuple[Order, TradeIntent]] = []
        last_prices: dict[str, float] = {}

        for pos_idx, session in enumerate(self.sessions):
            day_bars = self.bars.get(session, {})

            # 1+2. exits first (frees capital), then entries at the open.
            self._process_exits(session, pos_idx, day_bars, portfolio, open_trades, ledger)
            pending_entries = self._process_entries(
                session, day_bars, portfolio, open_trades, pending_entries, ledger
            )

            # 4. daily accounting.
            for symbol, bar in day_bars.items():
                last_prices[symbol] = bar.close
            borrow = portfolio.accrue_daily_costs(last_prices, self.cost_model)
            equity = portfolio.equity(last_prices)
            ledger.record_equity(
                session,
                portfolio.cash,
                equity,
                portfolio.gross_exposure(last_prices),
                len(portfolio.open_positions()),
                borrow,
            )

            # 5. tonight's signals become tomorrow's market-on-open entries.
            for intent in intents_by_session.get(session, []):
                order = self._entry_order(
                    intent, session, equity, last_prices, portfolio, open_trades, pending_entries
                )
                if order is not None:
                    ledger.record_order(order, session)
                    pending_entries.append((order, intent))

        self._force_close_remaining(portfolio, open_trades, last_prices, ledger)
        return ledger

    # -- phases ---------------------------------------------------------------

    def _process_exits(
        self,
        session: pd.Timestamp,
        pos_idx: int,
        day_bars: dict[str, Bar],
        portfolio: PortfolioState,
        open_trades: dict[str, _OpenTrade],
        ledger: Ledger,
    ) -> None:
        for symbol in list(open_trades):
            trade = open_trades[symbol]
            bar = day_bars.get(symbol)
            if bar is None:
                continue
            qty = -trade.quantity  # exit reverses the position
            exit_fill: Fill | None = None
            reason = ""

            if pos_idx >= trade.exit_session_pos:
                order = Order(
                    symbol=symbol,
                    quantity=qty,
                    order_type=OrderType.MARKET_ON_OPEN,
                    created_session=session,
                    tag="time_exit",
                )
                exit_fill = self.simulator.try_fill(order, bar, at_open_phase=True)
                reason = "time_exit"

            if exit_fill is None:
                stop_order = Order(
                    symbol=symbol,
                    quantity=qty,
                    order_type=OrderType.STOP,
                    created_session=session,
                    stop_price=trade.stop_price,
                    tag="stop_loss",
                )
                exit_fill = self.simulator.try_fill(stop_order, bar, at_open_phase=False)
                reason = "stop_loss"

            if exit_fill is None:
                target_order = Order(
                    symbol=symbol,
                    quantity=qty,
                    order_type=OrderType.LIMIT,
                    created_session=session,
                    limit_price=trade.target_price,
                    tag="target",
                )
                exit_fill = self.simulator.try_fill(target_order, bar, at_open_phase=False)
                reason = "target"

            if exit_fill is not None:
                portfolio.apply_fill(exit_fill, session)
                ledger.record_fill(exit_fill)
                direction = 1.0 if trade.side is Side.LONG else -1.0
                net_pnl = (
                    direction * abs(trade.quantity) * (exit_fill.price - trade.entry_price)
                    - exit_fill.cost
                    - trade.entry_cost
                )
                ledger.record_trade(
                    symbol=symbol,
                    side=trade.side.value,
                    entry_session=trade.entry_session or session,
                    exit_session=session,
                    entry_price=trade.entry_price,
                    exit_price=exit_fill.price,
                    quantity=trade.quantity,
                    net_pnl=net_pnl,
                    exit_reason=reason,
                )
                del open_trades[symbol]

    def _process_entries(
        self,
        session: pd.Timestamp,
        day_bars: dict[str, Bar],
        portfolio: PortfolioState,
        open_trades: dict[str, _OpenTrade],
        pending: list[tuple[Order, TradeIntent]],
        ledger: Ledger,
    ) -> list[tuple[Order, TradeIntent]]:
        still_pending: list[tuple[Order, TradeIntent]] = []
        for order, intent in pending:
            bar = day_bars.get(order.symbol)
            if bar is None:
                order.status = OrderStatus.EXPIRED
                ledger.record_order(order, session)
                continue
            fill = self.simulator.try_fill(order, bar, at_open_phase=True)
            if fill is None:
                order.status = OrderStatus.EXPIRED
                ledger.record_order(order, session)
                continue
            portfolio.apply_fill(fill, session)
            ledger.record_fill(fill)
            ledger.record_order(order, session)
            exit_pos = min(
                self.session_pos[session] + intent.horizon,
                len(self.sessions) - 1,
            )
            open_trades[intent.symbol] = _OpenTrade(
                side=intent.side,
                stop_price=intent.stop_price,
                target_price=intent.target_price,
                exit_session_pos=exit_pos,
                entry_price=fill.price,
                entry_session=session,
                quantity=fill.quantity,
                entry_cost=fill.cost,
            )
        return still_pending

    def _entry_order(
        self,
        intent: TradeIntent,
        session: pd.Timestamp,
        equity: float,
        last_prices: dict[str, float],
        portfolio: PortfolioState,
        open_trades: dict[str, _OpenTrade],
        pending: list[tuple[Order, TradeIntent]],
    ) -> Order | None:
        # Pending (accepted, not yet filled) entries count against every
        # limit: with many signals per session the open-position view alone
        # would let the book blow straight through the caps.
        pending_symbols = {o.symbol for o, _ in pending}
        if intent.symbol in open_trades or intent.symbol in pending_symbols:
            return None
        if len(open_trades) + len(pending) >= self.cfg.risk.max_positions:
            return None
        ref_price = last_prices.get(intent.symbol)
        if not ref_price or ref_price <= 0:
            return None
        weight_cap = self.cfg.risk.max_position_weight * equity
        qty = math.floor(weight_cap / ref_price)
        if qty < 1:
            return None
        reserved = sum(abs(o.quantity) * last_prices.get(o.symbol, 0.0) for o, _ in pending)
        gross = portfolio.gross_exposure(last_prices) + reserved
        if gross + qty * ref_price > self.cfg.risk.max_gross_exposure * equity:
            return None
        signed = qty if intent.side is Side.LONG else -qty
        return Order(
            symbol=intent.symbol,
            quantity=signed,
            order_type=OrderType.MARKET_ON_OPEN,
            created_session=session,
            tag="entry",
        )

    def _force_close_remaining(
        self,
        portfolio: PortfolioState,
        open_trades: dict[str, _OpenTrade],
        last_prices: dict[str, float],
        ledger: Ledger,
    ) -> None:
        last_session = self.sessions[-1]
        for symbol in list(open_trades):
            trade = open_trades[symbol]
            bar = self.bars.get(last_session, {}).get(symbol)
            if bar is None:
                continue
            order = Order(
                symbol=symbol,
                quantity=-trade.quantity,
                order_type=OrderType.MARKET_ON_CLOSE,
                created_session=last_session,
                tag="end_of_data",
            )
            fill = self.simulator.try_fill(order, bar, at_open_phase=False)
            if fill is None:
                continue
            portfolio.apply_fill(fill, last_session)
            ledger.record_fill(fill)
            direction = 1.0 if trade.side is Side.LONG else -1.0
            net_pnl = (
                direction * abs(trade.quantity) * (fill.price - trade.entry_price)
                - fill.cost
                - trade.entry_cost
            )
            ledger.record_trade(
                symbol=symbol,
                side=trade.side.value,
                entry_session=trade.entry_session or last_session,
                exit_session=last_session,
                entry_price=trade.entry_price,
                exit_price=fill.price,
                quantity=trade.quantity,
                net_pnl=net_pnl,
                exit_reason="end_of_data",
            )
            del open_trades[symbol]
