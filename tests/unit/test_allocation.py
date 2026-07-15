"""Allocator behavior and limit-enforcement properties."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from edgestack.config import RiskConfig
from edgestack.risk.allocation import AllocationRequest, allocate
from edgestack.types import Side


def _req(symbol: str, side: Side = Side.LONG, conviction: float = 70.0,
         vol: float = 0.25) -> AllocationRequest:
    return AllocationRequest(symbol=symbol, side=side, conviction=conviction,
                             annualized_vol=vol)


def test_equal_risk_prefers_low_vol() -> None:
    risk = RiskConfig(max_position_weight=0.5, max_gross_exposure=1.0)
    weights = allocate([_req("CALM", vol=0.10), _req("WILD", vol=0.50)], risk,
                       method="equal_risk")
    assert weights["CALM"] > weights["WILD"] > 0


def test_score_weighted_ignores_sub_neutral_conviction() -> None:
    risk = RiskConfig()
    weights = allocate(
        [_req("GOOD", conviction=80), _req("MEH", conviction=50)], risk,
        method="score_weighted",
    )
    assert "MEH" not in weights
    assert weights["GOOD"] > 0
    # All-neutral book allocates nothing at all.
    assert allocate([_req("A", conviction=45)], risk, method="score_weighted") == {}


def test_vol_target_scales_gross_to_target() -> None:
    risk = RiskConfig(max_position_weight=1.0, max_gross_exposure=3.0,
                      max_net_exposure=3.0, target_annualized_volatility=0.10)
    weights = allocate([_req("A", vol=0.20), _req("B", vol=0.20)], risk,
                       method="vol_target")
    # naive portfolio vol = sqrt(sum (w*vol)^2) should be ~ target
    naive = (sum((w * 0.20) ** 2 for w in weights.values())) ** 0.5
    assert naive == pytest.approx(0.10, rel=0.05)


def test_max_positions_keeps_highest_conviction() -> None:
    risk = RiskConfig(max_positions=2)
    weights = allocate(
        [_req("A", conviction=90), _req("B", conviction=80), _req("C", conviction=70)],
        risk,
    )
    assert set(weights) == {"A", "B"}


@given(
    n=st.integers(1, 12),
    seed=st.integers(0, 10_000),
)
def test_property_all_limits_hold(n: int, seed: int) -> None:
    import numpy as np

    rng = np.random.default_rng(seed)
    risk = RiskConfig(max_position_weight=0.05, max_sector_weight=0.2,
                      max_gross_exposure=0.5, max_net_exposure=0.2,
                      max_positions=8)
    requests = [
        AllocationRequest(
            symbol=f"S{i}",
            side=Side.LONG if rng.uniform() < 0.6 else Side.SHORT,
            conviction=float(rng.uniform(40, 100)),
            annualized_vol=float(rng.uniform(0.05, 0.9)),
        )
        for i in range(n)
    ]
    for method in ("equal_risk", "vol_target", "score_weighted"):
        weights = allocate(requests, risk, method=method)
        assert len(weights) <= risk.max_positions
        assert all(abs(w) <= risk.max_position_weight + 1e-9 for w in weights.values())
        assert sum(abs(w) for w in weights.values()) <= risk.max_gross_exposure + 1e-9
        assert abs(sum(weights.values())) <= risk.max_net_exposure + 1e-9
