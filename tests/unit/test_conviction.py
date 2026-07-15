"""Conviction-score property and behavior tests."""

from __future__ import annotations

from dataclasses import replace

from hypothesis import given
from hypothesis import strategies as st

from edgestack.scoring.conviction import ConvictionInputs, conviction

unit = st.floats(0.0, 1.0, allow_nan=False)


def _inputs(**overrides: float) -> ConvictionInputs:
    base = ConvictionInputs(
        calibrated_probability=0.6,
        expected_net_return=0.004,
        expected_risk=0.02,
        effective_sample_size=400,
        stability_score=0.8,
        out_of_sample_score=0.8,
        deflated_sharpe=0.8,
        regime_similarity=1.0,
        liquidity_score=0.9,
        data_quality_score=0.95,
        cost_survival_fraction=0.75,
        ci_width=0.002,
        tail_risk=-0.02,
        model_edge_disagreement=0.0,
    )
    return replace(base, **overrides)


@given(
    p=unit, net=st.floats(-0.05, 0.05, allow_nan=False),
    risk=st.floats(0.0, 0.2, allow_nan=False),
    ess=st.floats(0.0, 10000, allow_nan=False),
    stability=unit, oos=unit, dsr=unit, regime=unit, liq=unit, dq=unit,
    cost=unit, ci=st.floats(0.0, 0.5, allow_nan=False),
    tail=st.floats(-0.5, 0.0, allow_nan=False), disagree=unit,
)
def test_property_score_always_in_bounds(
    p, net, risk, ess, stability, oos, dsr, regime, liq, dq, cost, ci, tail, disagree
) -> None:
    result = conviction(
        ConvictionInputs(
            calibrated_probability=p, expected_net_return=net, expected_risk=risk,
            effective_sample_size=ess, stability_score=stability,
            out_of_sample_score=oos, deflated_sharpe=dsr, regime_similarity=regime,
            liquidity_score=liq, data_quality_score=dq, cost_survival_fraction=cost,
            ci_width=ci, tail_risk=tail, model_edge_disagreement=disagree,
        )
    )
    assert 0.0 <= result.score <= 100.0


@given(p_low=unit, p_high=unit)
def test_property_monotone_in_probability(p_low: float, p_high: float) -> None:
    if p_low > p_high:
        p_low, p_high = p_high, p_low
    low = conviction(_inputs(calibrated_probability=p_low))
    high = conviction(_inputs(calibrated_probability=p_high))
    assert high.score >= low.score - 1e-9


def test_no_economic_edge_never_scores_above_neutral() -> None:
    assert conviction(_inputs(expected_net_return=0.0)).score <= 50.0
    assert conviction(_inputs(expected_net_return=-0.01)).score <= 50.0
    assert conviction(_inputs(calibrated_probability=0.5)).score <= 50.0


def test_small_samples_shrink_toward_neutral() -> None:
    big = conviction(_inputs(effective_sample_size=2000))
    small = conviction(_inputs(effective_sample_size=10))
    assert big.score > small.score
    assert abs(small.score - 50.0) < abs(big.score - 50.0)


def test_score_drops_with_wide_intervals_and_disagreement() -> None:
    tight = conviction(_inputs())
    wide = conviction(_inputs(ci_width=0.05))
    disagreeing = conviction(_inputs(model_edge_disagreement=0.5))
    assert wide.score < tight.score
    assert disagreeing.score < tight.score


def test_regime_mismatch_and_costs_reduce_score() -> None:
    good = conviction(_inputs())
    bad_regime = conviction(_inputs(regime_similarity=0.1))
    bad_costs = conviction(_inputs(cost_survival_fraction=0.25))
    assert bad_regime.score < good.score
    assert bad_costs.score < good.score


def test_components_are_reported() -> None:
    result = conviction(_inputs())
    for key in ("base_edge", "reliability", "applicability", "tradability",
                "risk_penalty", "shrink", "raw_shrunk"):
        assert key in result.components
