"""Family-capped evidence aggregation and risk-plan tests."""

from __future__ import annotations

from datetime import date

import pytest

from edgestack.config import EdgeStackConfig, ScoringConfig
from edgestack.data.calendar import TradingCalendar
from edgestack.risk.sizing import build_entry_plan, build_risk_plan
from edgestack.scoring.ranking import aggregate_evidence
from edgestack.types import (
    Edge,
    EdgeIdentity,
    EdgeLifecycle,
    EdgeRobustness,
    EdgeStats,
    EdgeStatus,
    Family,
    Predicate,
    Side,
)


def _edge(edge_id: str, family: Family, net: float = 0.005, horizon: int = 10) -> Edge:
    return Edge(
        identity=EdgeIdentity(
            edge_id=edge_id, name=edge_id, description=edge_id, direction=Side.LONG,
            condition=Predicate(feature="rsi_14", op="<", value=30.0),
            feature_dependencies=("rsi_14",), family=family, holding_horizon=horizon,
        ),
        stats=EdgeStats(
            sample_size=300, effective_sample_size=250.0, gross_mean_return=net + 0.002,
            net_mean_return=net, median_return=net, return_std=0.02,
            downside_deviation=0.01, probability_of_profit=0.56,
            probability_of_positive_net_return=0.55, expected_shortfall=-0.03,
            value_at_risk=-0.02, max_drawdown=-0.1, sharpe_ratio=1.0, sortino_ratio=1.2,
            p_value=0.01, adjusted_p_value=0.03, q_value=0.03,
            bayesian_posterior_probability=0.9,
            bayesian_credible_interval=(0.001, 0.008),
            bootstrap_confidence_interval=(0.001, 0.009),
            deflated_sharpe_ratio=0.8,
        ),
        robustness=EdgeRobustness(
            stability_score=0.8, regime_stability_score=0.6, cost_robustness_score=0.75,
            parameter_robustness_score=0.7, out_of_sample_score=0.8, decay_score=1.0,
        ),
        lifecycle=EdgeLifecycle(
            status=EdgeStatus.VALIDATED,
            discovery_start=date(2010, 1, 1), discovery_end=date(2019, 12, 31),
            validation_start=date(2010, 1, 1), validation_end=date(2019, 12, 31),
            discovery_batch_id="b", experiment_id="e",
        ),
    )


def test_correlated_family_cannot_stack_conviction() -> None:
    scoring = ScoringConfig()
    one = aggregate_evidence([_edge("m1", Family.MOMENTUM)], scoring)
    five_same = aggregate_evidence(
        [_edge(f"m{i}", Family.MOMENTUM) for i in range(5)], scoring
    )
    families = [Family.MOMENTUM, Family.TREND, Family.CALENDAR, Family.VOLUME,
                Family.STRUCTURE]
    five_diverse = aggregate_evidence(
        [_edge(f"d{i}", fam) for i, fam in enumerate(families)], scoring
    )
    assert one is not None and five_same is not None and five_diverse is not None
    # Five clones of the same family barely beat one edge...
    assert five_same.total_contribution <= scoring.family_cap
    # ...while five DIFFERENT families genuinely add evidence.
    assert five_diverse.total_contribution > five_same.total_contribution * 1.5
    # Contributions are normalized and reported per edge.
    assert sum(i.contribution for i in five_diverse.items) == pytest.approx(1.0, abs=0.01)


def test_recommended_horizon_follows_strongest_edge() -> None:
    scoring = ScoringConfig()
    weak = _edge("weak", Family.MOMENTUM, net=0.001, horizon=5)
    strong = _edge("strong", Family.CALENDAR, net=0.009, horizon=20)
    agg = aggregate_evidence([weak, strong], scoring)
    assert agg is not None
    assert agg.recommended_horizon == 20


def test_empty_evidence_returns_none() -> None:
    assert aggregate_evidence([], ScoringConfig()) is None


def test_risk_plan_long_and_short() -> None:
    cfg = EdgeStackConfig()
    long_plan = build_risk_plan(Side.LONG, entry_ref=100.0, atr=2.0, cfg=cfg)
    assert long_plan.stop_price == pytest.approx(96.0)   # 2 ATR below
    assert long_plan.target_1 == pytest.approx(105.0)    # 2.5 ATR above
    assert long_plan.target_2 == pytest.approx(108.0)
    assert long_plan.reward_to_risk == pytest.approx(1.25)

    short_plan = build_risk_plan(Side.SHORT, entry_ref=100.0, atr=2.0, cfg=cfg)
    assert short_plan.stop_price == pytest.approx(104.0)  # above entry for shorts
    assert short_plan.target_1 == pytest.approx(95.0)


def test_entry_plan_next_open_and_zone() -> None:
    cal = TradingCalendar("XNYS")
    plan = build_entry_plan(Side.LONG, close=100.0, atr=2.0,
                            as_of=date(2023, 4, 6), calendar=cal)
    # Signal at Thursday close before Good Friday -> earliest entry Monday open.
    assert plan.earliest_timestamp.date() == date(2023, 4, 10)
    assert plan.ideal_low == pytest.approx(99.0)
    assert plan.ideal_high == pytest.approx(100.5)
    assert plan.do_not_chase_above == pytest.approx(102.0)
