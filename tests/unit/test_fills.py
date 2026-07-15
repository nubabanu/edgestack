"""Table-driven fill-simulation tests."""

from __future__ import annotations

import pandas as pd
import pytest

from edgestack.execution.costs import CostModel
from edgestack.execution.fills import Bar, FillSimulator
from edgestack.execution.orders import Order, OrderType
from edgestack.types import CostScenario

SESSION = pd.Timestamp("2020-06-01")


def _sim() -> FillSimulator:
    return FillSimulator(CostModel(
        scenario=CostScenario.BASE, commission_bps=1.0, half_spread_bps=5.0,
        base_slippage_bps=5.0, impact_coeff_bps=10.0,
        short_borrow_annualized=0.05, sec_fee_bps=0.03,
    ))


def _bar(open_=100.0, high=105.0, low=95.0, close=102.0, volume=1e6) -> Bar:
    return Bar(session=SESSION, open=open_, high=high, low=low, close=close,
               volume=volume)


def _order(qty: float, order_type: OrderType, **kw) -> Order:
    return Order(symbol="TST", quantity=qty, order_type=order_type,
                 created_session=SESSION, **kw)


def test_market_on_open_and_close_phases() -> None:
    sim = _sim()
    moo = _order(100, OrderType.MARKET_ON_OPEN)
    fill = sim.try_fill(moo, _bar(), at_open_phase=True)
    assert fill is not None and fill.price == 100.0
    assert fill.cost > 0  # friction charged explicitly
    assert sim.try_fill(_order(100, OrderType.MARKET_ON_OPEN), _bar(),
                        at_open_phase=False) is None
    moc = sim.try_fill(_order(100, OrderType.MARKET_ON_CLOSE), _bar(),
                       at_open_phase=False)
    assert moc is not None and moc.price == 102.0


@pytest.mark.parametrize(
    ("open_", "low", "limit", "expected"),
    [
        (100.0, 95.0, 97.0, 97.0),    # touched intraday -> at the limit
        (96.0, 95.0, 97.0, 96.0),     # opened below limit -> better fill at open
        (100.0, 98.0, 97.0, None),    # never traded down to the limit -> no fill
    ],
)
def test_buy_limit_rules(open_: float, low: float, limit: float,
                         expected: float | None) -> None:
    fill = _sim().try_fill(
        _order(100, OrderType.LIMIT, limit_price=limit),
        _bar(open_=open_, low=low), at_open_phase=False,
    )
    if expected is None:
        assert fill is None
    else:
        assert fill is not None and fill.price == expected


@pytest.mark.parametrize(
    ("open_", "low", "stop", "expected"),
    [
        (100.0, 95.0, 96.0, 96.0),    # touched intraday -> at the stop
        (94.0, 90.0, 96.0, 94.0),     # GAPPED THROUGH -> at the (worse) open
        (100.0, 97.0, 96.0, None),    # stop never touched
    ],
)
def test_sell_stop_rules_including_gap(open_: float, low: float, stop: float,
                                       expected: float | None) -> None:
    fill = _sim().try_fill(
        _order(-100, OrderType.STOP, stop_price=stop),
        _bar(open_=open_, low=low, close=max(low, 95.0)), at_open_phase=False,
    )
    if expected is None:
        assert fill is None
    else:
        assert fill is not None and fill.price == expected


def test_buy_stop_gap_through_fills_at_open() -> None:
    # Covering a short after an upward gap: open 108 > stop 105 -> fill at 108.
    fill = _sim().try_fill(
        _order(100, OrderType.STOP, stop_price=105.0),
        _bar(open_=108.0, high=110.0, low=107.0, close=109.0), at_open_phase=False,
    )
    assert fill is not None and fill.price == 108.0


def test_participation_cap_creates_partial_fill() -> None:
    fill = _sim().try_fill(
        _order(1_000_000, OrderType.MARKET_ON_OPEN), _bar(volume=1e6),
        at_open_phase=True,
    )
    assert fill is not None
    assert fill.quantity == pytest.approx(100_000)  # 10% of session volume


def test_every_fill_price_is_inside_the_bar_range() -> None:
    sim = _sim()
    bar = _bar()
    cases = [
        _order(10, OrderType.MARKET_ON_OPEN),
        _order(-10, OrderType.MARKET_ON_CLOSE),
        _order(10, OrderType.LIMIT, limit_price=99.0),
        _order(-10, OrderType.STOP, stop_price=96.0),
    ]
    for order in cases:
        for phase in (True, False):
            fill = sim.try_fill(order, bar, at_open_phase=phase)
            if fill is not None:
                assert bar.low <= fill.price <= bar.high
