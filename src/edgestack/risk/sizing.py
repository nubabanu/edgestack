"""Volatility-aware stops, profit targets and entry zones."""

from __future__ import annotations

from datetime import date

from edgestack.config import EdgeStackConfig
from edgestack.data.calendar import TradingCalendar
from edgestack.exceptions import DataError
from edgestack.types import EntryMethod, EntryPlan, RiskPlan, Side

TICK = 0.01


def _round_tick(price: float) -> float:
    return round(round(price / TICK) * TICK, 2)


def build_risk_plan(side: Side, entry_ref: float, atr: float,
                    cfg: EdgeStackConfig) -> RiskPlan:
    """ATR-multiple stop and targets around a reference entry price."""
    if entry_ref <= 0 or atr <= 0:
        raise DataError("entry price and ATR must be positive")
    stop_mult = cfg.risk.atr_stop_multiple
    t1_mult = cfg.risk.atr_target_multiples[0]
    t2_mult = cfg.risk.atr_target_multiples[1] if len(cfg.risk.atr_target_multiples) > 1 else None

    if side is Side.LONG:
        stop = entry_ref - stop_mult * atr
        target_1 = entry_ref + t1_mult * atr
        target_2 = entry_ref + t2_mult * atr if t2_mult else None
    else:
        stop = entry_ref + stop_mult * atr
        target_1 = entry_ref - t1_mult * atr
        target_2 = entry_ref - t2_mult * atr if t2_mult else None

    risk = abs(entry_ref - stop)
    reward = abs(target_1 - entry_ref)
    return RiskPlan(
        stop_price=_round_tick(max(stop, TICK)),
        target_1=_round_tick(max(target_1, TICK)),
        target_2=_round_tick(max(target_2, TICK)) if target_2 else None,
        reward_to_risk=round(reward / risk, 2) if risk > 0 else 0.0,
    )


def build_entry_plan(side: Side, close: float, atr: float, as_of: date,
                     calendar: TradingCalendar,
                     execution_delay_sessions: int = 1) -> EntryPlan:
    """Next-open entry with a volatility-scaled limit zone.

    The earliest valid execution is the open of the session
    ``execution_delay_sessions`` after the signal date — a close-observed
    signal can never execute at its own close.
    """
    entry_session = calendar.next_session(as_of, count=execution_delay_sessions)
    earliest = calendar.market_open_at(entry_session)
    if side is Side.LONG:
        ideal_low = close - 0.5 * atr
        ideal_high = close + 0.25 * atr
        do_not_chase = close + 1.0 * atr
    else:
        ideal_low = close - 0.25 * atr
        ideal_high = close + 0.5 * atr
        do_not_chase = close - 1.0 * atr
    return EntryPlan(
        method=EntryMethod.NEXT_OPEN,
        earliest_timestamp=earliest.to_pydatetime(),
        ideal_low=_round_tick(max(ideal_low, TICK)),
        ideal_high=_round_tick(max(ideal_high, TICK)),
        do_not_chase_above=_round_tick(max(do_not_chase, TICK)),
        entry_expiration_sessions=3,
    )
