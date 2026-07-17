"""Edge/candidate persistence round trips."""

from __future__ import annotations

from datetime import date

import pytest

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog
from edgestack.discovery.candidate_generation import DiscoveryBatch
from edgestack.discovery.edge_store import (
    current_statuses,
    load_batch,
    load_edges,
    record_edge_event,
    save_batch,
    save_edges,
)
from edgestack.exceptions import ValidationError
from edgestack.types import (
    CandidateEdge,
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


def _candidate(cid: str = "cand00000001") -> CandidateEdge:
    return CandidateEdge(
        candidate_id=cid,
        discovery_batch_id="batch01",
        experiment_id="exp01",
        name="long_h5: cal_is_friday == 1.0",
        direction=Side.LONG,
        condition=Predicate(feature="cal_is_friday", op="==", value=1.0),
        family=Family.CALENDAR,
        horizon=5,
        in_sample_n=500,
        in_sample_gross_mean=0.003,
        in_sample_net_mean=0.001,
        in_sample_hit_rate=0.56,
        in_sample_p_value=0.01,
    )


def _edge(status: EdgeStatus = EdgeStatus.VALIDATED) -> Edge:
    cand = _candidate()
    return Edge(
        identity=EdgeIdentity(
            edge_id=cand.candidate_id,
            name=cand.name,
            description="friday",
            direction=cand.direction,
            condition=cand.condition,
            feature_dependencies=("cal_is_friday",),
            family=cand.family,
            holding_horizon=cand.horizon,
        ),
        stats=EdgeStats(
            sample_size=400,
            effective_sample_size=350.0,
            gross_mean_return=0.003,
            net_mean_return=0.001,
            median_return=0.001,
            return_std=0.02,
            downside_deviation=0.01,
            probability_of_profit=0.55,
            probability_of_positive_net_return=0.54,
            expected_shortfall=-0.03,
            value_at_risk=-0.02,
            max_drawdown=-0.1,
            sharpe_ratio=1.1,
            sortino_ratio=1.4,
            p_value=0.004,
            adjusted_p_value=0.02,
            q_value=0.02,
            bayesian_posterior_probability=0.9,
            bayesian_credible_interval=(0.0002, 0.002),
            bootstrap_confidence_interval=(0.0001, 0.0021),
            deflated_sharpe_ratio=0.8,
        ),
        robustness=EdgeRobustness(
            stability_score=0.8,
            regime_stability_score=0.5,
            cost_robustness_score=0.75,
            parameter_robustness_score=0.7,
            out_of_sample_score=0.8,
            decay_score=1.0,
            cost_scenario_survival={"CONSERVATIVE": True},
            fold_net_means=(0.001, 0.002),
        ),
        lifecycle=EdgeLifecycle(
            status=status,
            discovery_start=date(2010, 1, 1),
            discovery_end=date(2019, 12, 31),
            validation_start=date(2010, 1, 1),
            validation_end=date(2019, 12, 31),
            discovery_batch_id="batch01",
            experiment_id="exp01",
        ),
    )


def test_batch_round_trip(cfg: EdgeStackConfig) -> None:
    catalog = DataCatalog(cfg)
    exp = catalog.record_experiment("discovery")
    batch = DiscoveryBatch(batch_id="batch01", experiment_id=exp, candidates=(_candidate(),))
    save_batch(catalog, batch)
    loaded = load_batch(catalog)
    assert loaded.batch_id == "batch01"
    assert loaded.candidates[0] == batch.candidates[0]
    # trial_count lands on the experiment record, written by the store.
    with catalog.connect() as con:
        count = con.execute(
            "SELECT trial_count FROM experiments WHERE experiment_id = ?", [exp]
        ).fetchone()[0]
    assert count == 1


def test_missing_batch_raises(cfg: EdgeStackConfig) -> None:
    catalog = DataCatalog(cfg)
    with pytest.raises(ValidationError, match="no discovery batches"):
        load_batch(catalog)


def test_edge_lifecycle_is_event_sourced(cfg: EdgeStackConfig) -> None:
    catalog = DataCatalog(cfg)
    edge = _edge()
    save_edges(catalog, [edge])
    assert [e.identity.edge_id for e in load_edges(catalog)] == [edge.identity.edge_id]

    # Status changes append events; the edge row itself is never rewritten.
    record_edge_event(
        catalog,
        edge.identity.edge_id,
        EdgeStatus.VALIDATED,
        EdgeStatus.DEGRADED,
        "rolling net mean below expectation",
    )
    statuses = current_statuses(catalog)
    assert statuses.loc[statuses["edge_id"] == edge.identity.edge_id, "status"].iloc[0] == (
        EdgeStatus.DEGRADED.value
    )
    assert load_edges(catalog, statuses=(EdgeStatus.VALIDATED, EdgeStatus.ACTIVE)) == []
    assert len(load_edges(catalog, statuses=(EdgeStatus.DEGRADED,))) == 1
