"""Statistical acceptance tests: the engine must reject noise, detect real
edges, reject train-only edges, and refuse cost-destroyed edges."""

from __future__ import annotations

from datetime import date

import pytest
from harness import (
    THURSDAY_LONG_PREFIX,
    edge_by_name_prefix,
    friday_effect,
    run_research,
    validated_names,
)

from edgestack.types import CostScenario, EdgeStatus, Side

pytestmark = pytest.mark.statistical


def test_random_walk_yields_no_validated_edges() -> None:
    _, batch, edges = run_research(effects=())
    assert batch.trial_count == 14  # 7 conditions x 2 sides: honest trial count
    assert validated_names(edges) == []
    # Every rejection carries explicit reasons.
    assert all(e.lifecycle.failure_reasons for e in edges)


def test_persistent_injected_edge_is_detected() -> None:
    _, _, edges = run_research(effects=(friday_effect(60.0),))
    names = validated_names(edges)
    assert any(n.startswith(THURSDAY_LONG_PREFIX) for n in names), names

    edge = edge_by_name_prefix(edges, THURSDAY_LONG_PREFIX)
    assert edge.lifecycle.status is EdgeStatus.VALIDATED
    assert edge.stats.net_mean_return > 0.001
    assert edge.stats.q_value <= 0.05
    assert edge.stats.deflated_sharpe_ratio >= 0.5
    assert edge.robustness.stability_score >= 2 / 3
    # The mirrored SHORT rule must NOT validate.
    short_thursday = [
        e for e in edges
        if e.identity.direction is Side.SHORT
        and e.identity.name.startswith("short_h1: cal_weekday == 3.0")
    ]
    assert short_thursday and short_thursday[0].lifecycle.status is EdgeStatus.REJECTED


def test_train_only_edge_is_rejected_out_of_sample() -> None:
    # Effect exists only 2010-2015; walk-forward test windows cover ~2018-2020.
    effect = friday_effect(60.0, start=date(2010, 1, 1), end=date(2015, 12, 31))
    _, _, edges = run_research(effects=(effect,))
    edge = edge_by_name_prefix(edges, THURSDAY_LONG_PREFIX)
    assert edge.lifecycle.status is EdgeStatus.REJECTED
    assert edge.lifecycle.failure_reasons


def test_cost_destroyed_edge_is_not_activated() -> None:
    # ~25 bps gross per trade: clears OPTIMISTIC (~10 bps) round-trip costs
    # but not CONSERVATIVE (~30 bps). The edge is real but not tradeable.
    _, _, edges = run_research(effects=(friday_effect(25.0),))
    edge = edge_by_name_prefix(edges, THURSDAY_LONG_PREFIX)
    assert edge.lifecycle.status is EdgeStatus.REJECTED
    survival = edge.robustness.cost_scenario_survival
    assert survival.get(CostScenario.OPTIMISTIC.value) is True
    assert survival.get(CostScenario.CONSERVATIVE.value) is False
    assert any(
        "conservative costs" in r or "economic floor" in r
        for r in edge.lifecycle.failure_reasons
    )
