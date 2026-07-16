"""Cost-model tests, including the monotonicity property."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from edgestack.execution.costs import SCENARIO_MULTIPLIER, CostModel
from edgestack.types import CostScenario, Side

_ORDER = [
    CostScenario.OPTIMISTIC,
    CostScenario.BASE,
    CostScenario.CONSERVATIVE,
    CostScenario.STRESS,
]


def _model(scenario: CostScenario, **overrides: float) -> CostModel:
    params: dict[str, float] = {
        "commission_bps": 1.0,
        "half_spread_bps": 5.0,
        "base_slippage_bps": 5.0,
        "impact_coeff_bps": 10.0,
        "short_borrow_annualized": 0.05,
        "sec_fee_bps": 0.03,
    }
    params.update(overrides)
    return CostModel(scenario=scenario, **params)


def test_scenario_multipliers_are_ordered() -> None:
    values = [SCENARIO_MULTIPLIER[s] for s in _ORDER]
    assert values == sorted(values)


def test_roundtrip_components() -> None:
    m = _model(CostScenario.BASE)
    # Long round trip: buy leg (no SEC fee) + sell leg (SEC fee), no borrow.
    expected = (1.0 + 10.0) / 1e4 + (1.0 + 0.03 + 10.0) / 1e4
    assert m.roundtrip_cost(Side.LONG, 10) == pytest.approx(expected)


def test_short_pays_borrow_over_holding() -> None:
    m = _model(CostScenario.BASE)
    short_10 = m.roundtrip_cost(Side.SHORT, 10)
    short_40 = m.roundtrip_cost(Side.SHORT, 40)
    assert short_40 > short_10
    assert short_40 - short_10 == pytest.approx(0.05 * 30 / 252)


def test_short_net_return_flips_direction() -> None:
    m = _model(CostScenario.BASE)
    # A -3% move earns a short ~+3% gross.
    assert m.net_return(-0.03, Side.SHORT, 5) > 0.02


@given(
    commission=st.floats(0, 5),
    spread=st.floats(0, 25),
    slippage=st.floats(0, 25),
    impact=st.floats(0, 50),
    borrow=st.floats(0, 0.3),
    gross=st.floats(-0.3, 0.3),
    holding=st.integers(1, 60),
    participation=st.floats(0, 1),
    side=st.sampled_from([Side.LONG, Side.SHORT]),
)
def test_property_increasing_costs_never_increase_net_returns(
    commission: float,
    spread: float,
    slippage: float,
    impact: float,
    borrow: float,
    gross: float,
    holding: int,
    participation: float,
    side: Side,
) -> None:
    nets = []
    for scenario in _ORDER:
        m = CostModel(
            scenario=scenario,
            commission_bps=commission,
            half_spread_bps=spread,
            base_slippage_bps=slippage,
            impact_coeff_bps=impact,
            short_borrow_annualized=borrow,
            sec_fee_bps=0.03,
        )
        nets.append(m.net_return(gross, side, holding, participation))
    assert all(a >= b - 1e-12 for a, b in itertools.pairwise(nets))
