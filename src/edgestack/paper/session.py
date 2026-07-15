"""Paper-trading sessions: simulated fills, positions, P&L, prediction tracking.

Daily flow for session S:

1. exits for open positions against S's bar (stop first, then target, then
   time exit at the open after the holding horizon);
2. entries at S's open from the signal report generated at the PREVIOUS
   session's close (long candidates only — shorts are research-only without
   borrow data and are never paper-executed);
3. mark to market at S's close, persist state, print a daily report.

State is a JSON file under the artifacts directory (atomic writes). Every
closed trade stores what was PREDICTED next to what actually happened, so
monitoring can compare forecasts with outcomes.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel

from edgestack.config import EdgeStackConfig
from edgestack.data.calendar import TradingCalendar
from edgestack.data.catalog import DataCatalog, atomic_write_bytes
from edgestack.exceptions import DataError
from edgestack.execution.fills import Bar
from edgestack.execution.orders import Order, OrderType
from edgestack.paper.broker import get_broker
from edgestack.reporting.signal_report import load_report
from edgestack.risk.allocation import AllocationRequest, allocate
from edgestack.types import Side


class PaperPosition(BaseModel):
    symbol: str
    side: Side
    quantity: float
    entry_price: float
    entry_session: date
    stop_price: float
    target_price: float
    horizon: int
    predicted_probability: float
    predicted_net_return: float
    entry_cost: float


class PaperTrade(BaseModel):
    symbol: str
    side: Side
    entry_session: date
    exit_session: date
    entry_price: float
    exit_price: float
    quantity: float
    net_pnl: float
    exit_reason: str
    predicted_probability: float
    predicted_net_return: float
    realized_net_return: float


class PaperState(BaseModel):
    cash: float
    last_session: date | None = None
    positions: list[PaperPosition] = []
    trades: list[PaperTrade] = []
    skipped_shorts: int = 0


def _state_path(catalog: DataCatalog):
    return catalog.artifacts_dir / "paper" / "state.json"


def load_state(catalog: DataCatalog, cfg: EdgeStackConfig) -> PaperState:
    path = _state_path(catalog)
    if path.exists():
        return PaperState.model_validate_json(path.read_text(encoding="utf-8"))
    return PaperState(cash=cfg.paper.initial_cash)


def save_state(catalog: DataCatalog, state: PaperState) -> None:
    atomic_write_bytes(_state_path(catalog), state.model_dump_json(indent=2).encode())


def run_session(cfg: EdgeStackConfig, as_of: date | None = None) -> str:
    """Run one paper session; returns the printed daily report."""
    catalog = DataCatalog(cfg)
    calendar = TradingCalendar(cfg.data.calendar)
    broker = get_broker(cfg)
    panel = catalog.load_panel()
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).unique()))
    session_ts = pd.Timestamp(as_of) if as_of else sessions[-1]
    if session_ts not in set(sessions):
        raise DataError(f"{session_ts.date()} is not a session with data")

    day = panel.loc[pd.to_datetime(panel["date"]) == session_ts]
    bars = {
        str(r.symbol): Bar(session=session_ts, open=r.open, high=r.high,
                           low=r.low, close=r.close, volume=r.volume)
        for r in day.itertuples(index=False)
    }

    state = load_state(catalog, cfg)
    if state.last_session is not None and session_ts.date() <= state.last_session:
        raise DataError(
            f"session {session_ts.date()} already processed "
            f"(last was {state.last_session})"
        )

    lines = [f"PAPER session {session_ts.date()} — simulated fills only, not advice"]
    _process_exits(state, bars, broker, calendar, session_ts, lines)
    _process_entries(cfg, catalog, calendar, state, bars, broker, session_ts, lines)

    closes = {s: b.close for s, b in bars.items()}
    market_value = sum(
        p.quantity * closes.get(p.symbol, p.entry_price) for p in state.positions
    )
    equity = state.cash + market_value
    state.last_session = session_ts.date()
    save_state(catalog, state)
    catalog.audit("paper_session", reason=str(session_ts.date()),
                  equity=round(equity, 2), positions=len(state.positions))

    realized = sum(t.net_pnl for t in state.trades)
    lines.append(
        f"equity {equity:,.2f} (cash {state.cash:,.2f}, "
        f"{len(state.positions)} open positions, realized P&L {realized:,.2f})"
    )
    if state.trades:
        recent = state.trades[-5:]
        lines.append("recent closed trades (predicted vs realized):")
        for t in recent:
            lines.append(
                f"  {t.symbol} {t.side.value}: predicted P={t.predicted_probability:.2f} "
                f"E[net]={t.predicted_net_return:+.4f} -> realized "
                f"{t.realized_net_return:+.4f} ({t.exit_reason})"
            )
    report = "\n".join(lines)
    print(report)
    return report


def _process_exits(state: PaperState, bars: dict[str, Bar], broker,
                   calendar: TradingCalendar, session_ts: pd.Timestamp,
                   lines: list[str]) -> None:
    still_open: list[PaperPosition] = []
    for pos in state.positions:
        bar = bars.get(pos.symbol)
        if bar is None:
            still_open.append(pos)
            continue
        held = calendar.sessions_between(pos.entry_session, session_ts.date())
        fill = None
        reason = ""
        if held >= pos.horizon:
            order = Order(symbol=pos.symbol, quantity=-pos.quantity,
                          order_type=OrderType.MARKET_ON_OPEN,
                          created_session=session_ts, tag="time_exit")
            fill = broker.execute_against_bar(order, bar, at_open_phase=True)
            reason = "time_exit"
        if fill is None:
            order = Order(symbol=pos.symbol, quantity=-pos.quantity,
                          order_type=OrderType.STOP, created_session=session_ts,
                          stop_price=pos.stop_price, tag="stop_loss")
            fill = broker.execute_against_bar(order, bar, at_open_phase=False)
            reason = "stop_loss"
        if fill is None:
            order = Order(symbol=pos.symbol, quantity=-pos.quantity,
                          order_type=OrderType.LIMIT, created_session=session_ts,
                          limit_price=pos.target_price, tag="target")
            fill = broker.execute_against_bar(order, bar, at_open_phase=False)
            reason = "target"
        if fill is None:
            still_open.append(pos)
            continue

        state.cash -= fill.quantity * fill.price + fill.cost
        direction = 1.0 if pos.side is Side.LONG else -1.0
        net_pnl = (direction * abs(pos.quantity) * (fill.price - pos.entry_price)
                   - fill.cost - pos.entry_cost)
        realized_ret = direction * (fill.price / pos.entry_price - 1.0)
        state.trades.append(PaperTrade(
            symbol=pos.symbol, side=pos.side, entry_session=pos.entry_session,
            exit_session=session_ts.date(), entry_price=pos.entry_price,
            exit_price=fill.price, quantity=pos.quantity, net_pnl=net_pnl,
            exit_reason=reason, predicted_probability=pos.predicted_probability,
            predicted_net_return=pos.predicted_net_return,
            realized_net_return=realized_ret,
        ))
        lines.append(f"closed {pos.symbol} {pos.side.value} via {reason}: "
                     f"net P&L {net_pnl:+,.2f}")
    state.positions = still_open


def _process_entries(cfg: EdgeStackConfig, catalog: DataCatalog,
                     calendar: TradingCalendar, state: PaperState,
                     bars: dict[str, Bar], broker, session_ts: pd.Timestamp,
                     lines: list[str]) -> None:
    prev_session = calendar.prev_session(session_ts.date())
    try:
        report = load_report(catalog, prev_session.date())
    except DataError:
        lines.append(f"no signal report for {prev_session.date()}: no new entries")
        return

    if report.short_candidates:
        state.skipped_shorts += len(report.short_candidates)
        lines.append(
            f"skipped {len(report.short_candidates)} short research candidates "
            "(no borrow data — research only)"
        )

    open_symbols = {p.symbol for p in state.positions}
    candidates = [c for c in report.long_candidates if c.symbol not in open_symbols]
    if not candidates:
        lines.append("no eligible long candidates: no new entries")
        return

    requests = [
        AllocationRequest(
            symbol=c.symbol, side=Side.LONG, conviction=c.conviction_score,
            annualized_vol=float(c.expected_volatility
                                 * np.sqrt(252.0 / max(1, c.recommended_holding_sessions))),
        )
        for c in candidates
    ]
    weights = allocate(requests, cfg.risk, method="score_weighted")
    closes = {s: b.close for s, b in bars.items()}
    equity = state.cash + sum(
        p.quantity * closes.get(p.symbol, p.entry_price) for p in state.positions
    )

    for candidate in candidates:
        weight = weights.get(candidate.symbol, 0.0)
        bar = bars.get(candidate.symbol)
        if weight <= 0 or bar is None:
            continue
        qty = int(weight * equity / bar.open)
        if qty < 1:
            continue
        order = Order(symbol=candidate.symbol, quantity=qty,
                      order_type=OrderType.MARKET_ON_OPEN,
                      created_session=session_ts, tag="entry")
        fill = broker.execute_against_bar(order, bar, at_open_phase=True)
        if fill is None:
            continue
        state.cash -= fill.quantity * fill.price + fill.cost
        state.positions.append(PaperPosition(
            symbol=candidate.symbol, side=Side.LONG, quantity=fill.quantity,
            entry_price=fill.price, entry_session=session_ts.date(),
            stop_price=candidate.risk.stop_price,
            target_price=candidate.risk.target_1,
            horizon=candidate.recommended_holding_sessions,
            predicted_probability=candidate.calibrated_probability_of_positive_net_return,
            predicted_net_return=candidate.expected_net_return,
            entry_cost=fill.cost,
        ))
        lines.append(f"opened {candidate.symbol} LONG {fill.quantity:.0f} @ "
                     f"{fill.price:.2f} (stop {candidate.risk.stop_price}, "
                     f"target {candidate.risk.target_1}, "
                     f"{candidate.recommended_holding_sessions} sessions)")
