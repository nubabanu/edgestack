"""Drift metric and lifecycle-rule tests."""

from __future__ import annotations

import numpy as np
import pytest

from edgestack.config import EdgeStackConfig
from edgestack.exceptions import ValidationError
from edgestack.monitoring.drift import population_stability_index
from edgestack.monitoring.lifecycle import EdgeHealth, decide_transition
from edgestack.types import EdgeStatus


def test_psi_zero_for_same_distribution() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(0, 1, 500)
    assert population_stability_index(ref, cur) < 0.05


def test_psi_large_for_shifted_distribution() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 2000)
    shifted = rng.normal(1.5, 1, 500)
    assert population_stability_index(ref, shifted) > 0.5


def test_psi_requires_enough_data() -> None:
    with pytest.raises(ValidationError):
        population_stability_index(np.arange(5.0), np.arange(5.0))


def _health(posterior: float, *, n: int = 30, psi: float = 0.0, since: int = 0) -> EdgeHealth:
    return EdgeHealth(
        edge_id="e1",
        n_recent=n,
        recent_net_mean=0.001,
        recent_hit_rate=0.55,
        recent_posterior=posterior,
        max_feature_psi=psi,
        sessions_since_status=since,
    )


def test_lifecycle_rules() -> None:
    cfg = EdgeStackConfig()  # degrade 0.40, suspend 0.25, retire 126

    # healthy VALIDATED gets promoted to ACTIVE
    decision = decide_transition(EdgeStatus.VALIDATED, _health(0.8), cfg)
    assert decision is not None and decision[0] is EdgeStatus.ACTIVE

    # decay: ACTIVE degrades, then suspends
    assert decide_transition(EdgeStatus.ACTIVE, _health(0.35), cfg)[0] is EdgeStatus.DEGRADED
    assert decide_transition(EdgeStatus.ACTIVE, _health(0.10), cfg)[0] is EdgeStatus.SUSPENDED
    assert decide_transition(EdgeStatus.DEGRADED, _health(0.10), cfg)[0] is EdgeStatus.SUSPENDED

    # recovery requires evidence
    assert decide_transition(EdgeStatus.DEGRADED, _health(0.6), cfg)[0] is EdgeStatus.ACTIVE
    assert decide_transition(EdgeStatus.SUSPENDED, _health(0.7), cfg)[0] is EdgeStatus.ACTIVE

    # feature drift alone degrades an active edge
    assert decide_transition(EdgeStatus.ACTIVE, _health(0.8, psi=0.5), cfg)[0] is (
        EdgeStatus.DEGRADED
    )

    # long suspension retires (even without recent signals)
    decision = decide_transition(EdgeStatus.SUSPENDED, _health(0.5, n=0, since=200), cfg)
    assert decision is not None and decision[0] is EdgeStatus.RETIRED

    # too little evidence -> no transition
    assert decide_transition(EdgeStatus.ACTIVE, _health(0.1, n=2), cfg) is None
    # healthy ACTIVE edge stays put
    assert decide_transition(EdgeStatus.ACTIVE, _health(0.8), cfg) is None
