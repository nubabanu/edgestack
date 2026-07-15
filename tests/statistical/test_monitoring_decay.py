"""Monitoring must catch an edge whose effect has vanished from the market."""

from __future__ import annotations

import pytest
from harness import (
    THURSDAY_LONG_PREFIX,
    build_frames,
    edge_by_name_prefix,
    friday_effect,
    run_research,
)

from edgestack.monitoring.lifecycle import assess_edge, decide_transition
from edgestack.types import EdgeStatus

pytestmark = pytest.mark.statistical


def test_vanished_edge_is_degraded_or_suspended() -> None:
    # Validate the Thursday edge on a market where the Friday drift is real...
    cfg, _, edges = run_research(effects=(friday_effect(60.0),))
    edge = edge_by_name_prefix(edges, THURSDAY_LONG_PREFIX)
    assert edge.lifecycle.status is EdgeStatus.VALIDATED

    # ...then monitor it against a market where the effect no longer exists.
    _, dead_features, dead_labels = build_frames(effects=(), seed=7, cfg=cfg)
    health = assess_edge(edge, dead_features, dead_labels, cfg,
                         sessions_since_status=10)
    assert health.n_recent >= 5
    assert health.recent_posterior < cfg.monitoring.degrade_posterior_threshold

    decision = decide_transition(EdgeStatus.ACTIVE, health, cfg)
    assert decision is not None
    assert decision[0] in (EdgeStatus.DEGRADED, EdgeStatus.SUSPENDED)


def test_healthy_edge_survives_monitoring() -> None:
    cfg, _, edges = run_research(effects=(friday_effect(60.0),))
    edge = edge_by_name_prefix(edges, THURSDAY_LONG_PREFIX)
    _, features, labels = build_frames(effects=(friday_effect(60.0),), cfg=cfg)
    health = assess_edge(edge, features, labels, cfg, sessions_since_status=10)
    assert health.recent_posterior >= cfg.monitoring.degrade_posterior_threshold
    # ACTIVE + healthy -> no transition.
    assert decide_transition(EdgeStatus.ACTIVE, health, cfg) is None
