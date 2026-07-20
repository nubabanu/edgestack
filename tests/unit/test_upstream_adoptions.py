from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError as PydanticValidationError

from edgestack.api.app import create_app
from edgestack.data.catalog import DataCatalog
from edgestack.exceptions import DataError, ExecutionModelError, ValidationError
from edgestack.execution.events import (
    DeterministicEventReplay,
    EventKind,
    ExecutionDataCapabilitiesV1,
    ExecutionFidelity,
    MarketEventV1,
    QueueModel,
    validate_execution_capabilities,
)
from edgestack.recommendation.ensemble import (
    build_promoted_ensemble,
    ensemble_optimizer_trials,
)
from edgestack.recommendation.schemas import (
    AssetKind,
    EvidenceGrade,
    SleeveContributionV2,
    WeightV2,
)
from edgestack.research.factor_diagnostics import analyze_factor
from edgestack.research.proposals import ProposalRegistry
from edgestack.research.schemas import (
    CampaignLifecycle,
    CandidateProposalV1,
    ProposalAttemptV1,
    ProposalStage,
    ResearchDataQueryV1,
    ShadowStrategyV1,
)
from edgestack.research.store import ResearchStore


def _proposal(*, previously_accessed: bool = False) -> CandidateProposalV1:
    return CandidateProposalV1(
        proposal_id="agent-trend-v1",
        created_at=datetime(2026, 7, 20, tzinfo=UTC),
        source="external-agent",
        family="trend_dip",
        name="Trend pullback after breadth confirmation",
        hypothesis="A broad uptrend followed by a small dip has positive next-week net return.",
        mechanism="Slow institutional rebalancing with temporary liquidity pressure.",
        feature_expressions=("close / lag(close, 200) - 1", "close / lag(close, 5) - 1"),
        parameter_grid={"lookback": (100, 200), "dip": (-0.01, -0.02)},
        data_queries=(
            ResearchDataQueryV1(
                dataset="prices",
                symbols=("SPY", "QQQ"),
                frequency="1d",
                start=date(2012, 1, 3),
                end=date(2025, 12, 31),
                fields=("open", "high", "low", "close", "volume"),
                required_observations=756,
                preferred_providers=("alpaca", "yahoo"),
            ),
        ),
        result_blind=not previously_accessed,
        previously_accessed=previously_accessed,
    )


def test_agent_proposals_expand_atomically_and_guard_holdout_access(cfg) -> None:
    registry = ProposalRegistry(DataCatalog(cfg))
    proposal = _proposal()

    audit = registry.register(proposal)
    assert audit.expected_trials == 4
    assert audit.registered_trials == 4
    assert audit.intake_eligible_for_fresh_campaign
    assert registry.register(proposal) == audit
    with registry.catalog.connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM trial_ledger_v2 WHERE experiment_id = ?",
                [proposal.proposal_id],
            ).fetchone()[0]
            == 4
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM audit_log WHERE event = 'research_proposal_registered'"
            ).fetchone()[0]
            == 1
        )

    first = ProposalAttemptV1(
        attempt_id="attempt-1",
        proposal_id=proposal.proposal_id,
        sequence=1,
        occurred_at=datetime(2026, 7, 20, 10, tzinfo=UTC),
        stage=ProposalStage.DATA_QUERY,
        data_query_id=proposal.data_queries[0].requirement_id(proposal.proposal_id),
        result_summary={"rows": 10_000},
    )
    registry.append_attempt(first)
    with pytest.raises(DataError, match="guarded-data access"):
        registry.append_attempt(
            ProposalAttemptV1(
                attempt_id="attempt-2-invalid",
                proposal_id=proposal.proposal_id,
                sequence=2,
                occurred_at=datetime(2026, 7, 20, 11, tzinfo=UTC),
                stage=ProposalStage.FINAL_HOLDOUT,
            )
        )
    final = ProposalAttemptV1(
        attempt_id="attempt-2",
        proposal_id=proposal.proposal_id,
        sequence=2,
        occurred_at=datetime(2026, 7, 20, 11, tzinfo=UTC),
        stage=ProposalStage.FINAL_HOLDOUT,
        viewed_guarded_data=True,
    )
    guarded = registry.append_attempt(final)
    assert guarded.guarded_data_accessed
    assert not guarded.intake_eligible_for_fresh_campaign

    with pytest.raises(DataError, match="immutable"):
        registry.register(proposal.model_copy(update={"hypothesis": "changed after results"}))


def test_proposal_state_is_available_through_read_only_api(cfg) -> None:
    proposal = _proposal(previously_accessed=True)
    ProposalRegistry(DataCatalog(cfg)).register(proposal)
    client = TestClient(create_app(cfg))

    listed = client.get("/research/proposals")
    assert listed.status_code == 200
    assert listed.json()[0]["proposal_id"] == proposal.proposal_id
    audit = client.get(f"/research/proposals/{proposal.proposal_id}/audit")
    assert audit.status_code == 200
    assert not audit.json()["intake_eligible_for_fresh_campaign"]
    assert client.get("/research/proposals/missing").status_code == 404


def test_factor_diagnostics_are_deterministic_and_non_promotional() -> None:
    rows = []
    symbols = tuple(f"S{index:02d}" for index in range(20))
    for session in pd.date_range("2025-01-02", periods=30, freq="B"):
        for index, symbol in enumerate(symbols):
            factor = float(index) + 0.01 * session.day
            rows.append(
                {
                    "date": session,
                    "symbol": symbol,
                    "group": "A" if index % 2 else "B",
                    "factor": factor,
                    "forward_return_1": 0.0002 * factor + 0.00001 * ((index % 3) - 1),
                    "forward_return_5": 0.0005 * factor + 0.00002 * ((index % 5) - 2),
                }
            )
    frame = pd.DataFrame(rows)

    first = analyze_factor(frame, factor_name="synthetic_monotonic", group_column="group")
    second = analyze_factor(frame, factor_name="synthetic_monotonic", group_column="group")

    assert first == second
    assert first.diagnostics_id == second.diagnostics_id
    assert first.rank_ic_mean["1"] > 0.95
    assert first.top_minus_bottom_mean["5"] > 0
    assert first.monotonicity["1"] > 0.95
    assert first.trial_charge == 20
    assert first.promotion_evidence is False
    assert max(first.quantile_turnover.values()) == 0.0

    with pytest.raises(ValidationError, match="duplicate"):
        analyze_factor(pd.concat([frame, frame.iloc[[0]]]), factor_name="duplicate")


def _event(
    event_id: str,
    *,
    event_offset: int,
    receive_offset: int,
    process_offset: int,
    sequence: int,
) -> MarketEventV1:
    base = datetime(2026, 7, 20, 13, 30, tzinfo=UTC)
    return MarketEventV1(
        event_id=event_id,
        kind=EventKind.TRADE,
        event_time=base + timedelta(microseconds=event_offset),
        receive_time=base + timedelta(microseconds=receive_offset),
        process_time=base + timedelta(microseconds=process_offset),
        sequence=sequence,
        source_hash="a" * 64,
        payload={"price": 100.0},
    )


def test_event_replay_uses_information_time_and_refuses_fake_fidelity() -> None:
    first = _event("first", event_offset=1, receive_offset=3, process_offset=8, sequence=1)
    second = _event("second", event_offset=2, receive_offset=4, process_offset=6, sequence=2)
    replay = DeterministicEventReplay((first, second))
    reversed_input = DeterministicEventReplay((second, first))

    assert [event.event_id for event in replay.events] == ["second", "first"]
    assert replay.content_hash == reversed_input.content_hash
    assert replay.run(lambda event: event.event_id) == ("second", "first")
    with pytest.raises(ExecutionModelError, match="duplicate"):
        DeterministicEventReplay((first, first))
    with pytest.raises(PydanticValidationError, match="event_time"):
        _event("causality", event_offset=5, receive_offset=4, process_offset=6, sequence=3)

    bars = ExecutionDataCapabilitiesV1(bars=True)
    validate_execution_capabilities(bars, fidelity=ExecutionFidelity.BAR)
    with pytest.raises(ExecutionModelError, match="level-2"):
        validate_execution_capabilities(
            bars,
            fidelity=ExecutionFidelity.BAR,
            queue_model=QueueModel.ESTIMATED,
        )
    l2 = ExecutionDataCapabilitiesV1(level2_book=True, feed_sequence=True)
    validate_execution_capabilities(
        l2,
        fidelity=ExecutionFidelity.L2,
        queue_model=QueueModel.ESTIMATED,
    )
    with pytest.raises(ExecutionModelError, match="receive_time"):
        validate_execution_capabilities(l2, fidelity=ExecutionFidelity.L2, model_latency=True)


def _sleeve(sleeve_id: str, family: str, *, promoted: bool = True) -> SleeveContributionV2:
    return SleeveContributionV2(
        sleeve_id=sleeve_id,
        artifact_hash=(sleeve_id[-1] * 64),
        horizon_sessions=5,
        family=family,
        symbol_weights=(
            WeightV2(
                symbol="SPY",
                weight=1.0,
                asset_kind=AssetKind.ETF,
                sector="broad_equity",
            ),
        ),
        expected_net_return=0.08,
        expected_return_lower_95=0.01,
        effective_sample_size=200,
        evidence_grade=EvidenceGrade.PROMOTED if promoted else EvidenceGrade.WATCHLIST,
    )


def test_robust_ensemble_uses_only_positive_promoted_sleeves_and_can_abstain(cfg) -> None:
    rng = np.random.default_rng(7)
    sleeves = (
        _sleeve("sleeve-1", "trend"),
        _sleeve("sleeve-2", "trend"),
        _sleeve("sleeve-3", "mean_reversion"),
    )
    positive = pd.DataFrame(
        rng.normal(0.002, 0.002, size=(180, 3)),
        columns=[sleeve.sleeve_id for sleeve in sleeves],
    )
    trials = ensemble_optimizer_trials(positive, capital_eligible_sleeves=sleeves)
    with pytest.raises(ValidationError, match="registered"):
        build_promoted_ensemble(positive, capital_eligible_sleeves=sleeves)
    result = build_promoted_ensemble(
        positive,
        capital_eligible_sleeves=sleeves,
        registered_optimizer_trials=trials,
    )

    assert result.eligible
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert max(result.weights.values()) <= 0.60 + 1e-8
    assert result.family_weights["trend"] <= 0.70 + 1e-8
    assert result.cross_validated_log_growth_lower_95 > 0
    assert result.trial_charge == 3

    negative = positive * 0 + rng.normal(-0.002, 0.0005, size=(180, 3))
    negative_trials = ensemble_optimizer_trials(negative, capital_eligible_sleeves=sleeves)
    abstained = build_promoted_ensemble(
        negative,
        capital_eligible_sleeves=sleeves,
        registered_optimizer_trials=negative_trials,
    )
    assert not abstained.eligible
    assert abstained.weights == {}

    with pytest.raises(ValidationError, match="unpromoted"):
        build_promoted_ensemble(
            positive,
            capital_eligible_sleeves=(sleeves[0], _sleeve("sleeve-2", "trend", promoted=False)),
        )

    store = ResearchStore(DataCatalog(cfg))
    for sleeve in sleeves:
        store.save_promoted_sleeve(sleeve)
        store.upsert_shadow(
            ShadowStrategyV1(
                strategy_id=sleeve.sleeve_id,
                campaign_id=f"campaign-{sleeve.sleeve_id}",
                artifact_hash=sleeve.artifact_hash,
                status=CampaignLifecycle.PROMOTED,
                started_at=datetime(2026, 1, 2, tzinfo=UTC),
                equity=100_000,
            )
        )
    persisted = store.build_promoted_ensemble(positive)
    assert persisted.eligible
    with store.catalog.connect() as connection:
        statuses = connection.execute(
            "SELECT status FROM trial_ledger_v2 WHERE experiment_id = ? ORDER BY trial_id",
            [persisted.ensemble_id],
        ).fetchall()
    assert statuses == [("SUCCEEDED",), ("SUCCEEDED",), ("SUCCEEDED",)]
