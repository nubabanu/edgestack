from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from edgestack.api.app import create_app
from edgestack.data.catalog import DataCatalog
from edgestack.exceptions import DataError
from edgestack.recommendation.schemas import (
    AssetKind,
    EvidenceGrade,
    SleeveContributionV2,
    WeightV2,
)
from edgestack.research.coverage import (
    CoveragePlanner,
    DataRequirement,
    ProviderCapability,
)
from edgestack.research.locks import nightly_lock
from edgestack.research.schemas import (
    AcquisitionJobV1,
    CampaignLifecycle,
    CoverageState,
    JobState,
    ShadowStrategyV1,
)
from edgestack.research.store import ResearchStore
from edgestack.research.templates import registered_templates, seed_registered_campaigns
from edgestack.research.worker import ResearchWorker, _execute_job_process


def test_insufficient_coverage_creates_resumable_job_and_long_yahoo_history_parks(cfg) -> None:
    store = ResearchStore(DataCatalog(cfg))
    planner = CoveragePlanner(
        store,
        (
            ProviderCapability("yahoo", ("prices",), ("1d",)),
            ProviderCapability(
                "yahoo_intraday",
                ("intraday",),
                ("1m",),
                max_range_days={"1m": 7},
            ),
        ),
    )
    daily = DataRequirement(
        campaign_id="daily",
        dataset="prices",
        symbols=("SPY",),
        frequency="1d",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        required_observations=200,
        preferred_providers=("yahoo",),
    )
    gap = planner.plan(daily)
    assert gap.state is CoverageState.NEEDS_DATA
    assert store.jobs()[0].state is JobState.PENDING

    intraday = DataRequirement(
        campaign_id="intraday",
        dataset="intraday",
        symbols=("SPY",),
        frequency="1m",
        start=date(2020, 1, 1),
        end=date(2024, 12, 31),
        required_observations=100_000,
        preferred_providers=("yahoo_intraday",),
    )
    blocked = planner.plan(intraday)
    assert blocked.state is CoverageState.BLOCKED_FREE_TIER
    assert store.jobs()[0].campaign_id == "daily"


def test_expired_lease_is_recoverable_after_restart(cfg) -> None:
    store = ResearchStore(DataCatalog(cfg))
    planner = CoveragePlanner(
        store,
        (ProviderCapability("yahoo", ("prices",), ("1d",)),),
    )
    requirement = DataRequirement(
        campaign_id="restart",
        dataset="prices",
        symbols=("SPY",),
        frequency="1d",
        start=date(2024, 1, 1),
        end=date(2024, 2, 1),
        required_observations=20,
        preferred_providers=("yahoo",),
    )
    planner.plan(requirement)
    first = store.lease_next_job("worker-a", now=datetime(2026, 1, 1, tzinfo=UTC))
    assert first is not None and first.state is JobState.LEASED
    recovered = store.lease_next_job(
        "worker-b",
        now=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=1),
    )
    assert recovered is not None
    assert recovered.job_id == first.job_id
    assert recovered.lease_owner == "worker-b"


def test_incomplete_success_tries_the_next_free_provider_before_parking(cfg) -> None:
    store = ResearchStore(DataCatalog(cfg))
    planner = CoveragePlanner(
        store,
        (
            ProviderCapability("alpaca", ("prices",), ("1d",)),
            ProviderCapability("yahoo", ("prices",), ("1d",)),
        ),
    )
    requirement = DataRequirement(
        campaign_id="fallback",
        dataset="prices",
        symbols=("SPY",),
        frequency="1d",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        required_observations=200,
        preferred_providers=("alpaca", "yahoo"),
    )
    first_gap = planner.plan(requirement)
    first_job = store.jobs()[0]
    store.upsert_job(first_job.model_copy(update={"state": JobState.SUCCEEDED}))

    repair_gap = planner.plan(requirement)

    assert first_gap.provider == "alpaca"
    assert repair_gap.state is CoverageState.NEEDS_DATA
    assert repair_gap.provider == "yahoo"
    assert {str(job.payload["provider"]) for job in store.jobs()} == {"alpaca", "yahoo"}


def test_leased_job_executes_in_an_isolated_worker_process(cfg, tmp_path) -> None:
    now = datetime.now(UTC)
    job = AcquisitionJobV1(
        job_id="process-contract",
        requirement_id="test",
        campaign_id="test",
        kind="ACQUIRE_DATA",
        state=JobState.LEASED,
        created_at=now,
        updated_at=now,
        payload={"dataset": "unsupported"},
    )
    with ProcessPoolExecutor(max_workers=1) as pool:
        encoded = pool.submit(
            _execute_job_process,
            cfg.model_dump_json(),
            str(tmp_path),
            job.model_dump_json(),
        ).result(timeout=30)
    result = AcquisitionJobV1.model_validate_json(encoded)

    assert result.state is JobState.RETRY
    assert result.attempts == 1
    assert ResearchStore(DataCatalog(cfg)).jobs()[0].job_id == job.job_id


def test_registered_campaigns_are_frozen_and_trial_capped(cfg) -> None:
    store = ResearchStore(DataCatalog(cfg))
    first = seed_registered_campaigns(store, cfg, through=date(2026, 6, 30))
    second = registered_templates(cfg, through=date(2026, 6, 30))
    assert [item.manifest_hash for item in first] == [item.manifest_hash for item in second]
    assert all(len(item.candidate_family) <= 1_500 for item in first)
    assert all((item.evaluator == "opening_fade") == item.previously_accessed for item in first)
    with store.catalog.connect() as connection:
        count = connection.execute("SELECT COUNT(*) FROM trial_ledger_v2").fetchone()[0]
    assert count == sum(len(item.candidate_family) for item in first)


def test_read_only_edge_factory_api_is_available_without_canonical_bundle(cfg) -> None:
    client = TestClient(create_app(cfg))
    overview = client.get("/research/overview")
    assert overview.status_code == 200
    assert overview.json()["worker"]["storage_cap_gb"] == 75.0
    assert client.get("/research/campaigns").json() == []
    assert client.get("/data/coverage").json() == []
    assert client.get("/paper/strategies").json() == []
    diagnostics = client.get("/recommendations/growth-diagnostics").json()
    assert diagnostics["action"] == "NO_ALLOCATION"
    assert diagnostics["evidence_state"] == "INSUFFICIENT"


def test_campaign_lifecycle_and_promoted_registry_are_immutable(cfg) -> None:
    store = ResearchStore(DataCatalog(cfg))
    campaign = seed_registered_campaigns(store, cfg, through=date(2026, 6, 30))[0]
    summary = store.campaign(campaign.template_id)
    assert summary is not None
    ready = summary.model_copy(update={"lifecycle": CampaignLifecycle.READY})
    store.upsert_campaign(ready)
    running = ready.model_copy(update={"lifecycle": CampaignLifecycle.RUNNING})
    store.upsert_campaign(running)
    with pytest.raises(DataError, match="invalid campaign transition"):
        store.upsert_campaign(running.model_copy(update={"lifecycle": CampaignLifecycle.READY}))

    sleeve = SleeveContributionV2(
        sleeve_id="daily:qualified",
        artifact_hash="a" * 64,
        horizon_sessions=5,
        family="daily_multi_family",
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
        effective_sample_size=300.0,
        evidence_grade=EvidenceGrade.PROMOTED,
    )
    store.save_promoted_sleeve(sleeve)
    store.save_promoted_sleeve(sleeve)
    assert store.promoted_sleeves() == (sleeve,)
    assert store.capital_eligible_sleeves() == ()
    store.upsert_shadow(
        ShadowStrategyV1(
            strategy_id=sleeve.sleeve_id,
            campaign_id=campaign.template_id,
            artifact_hash=sleeve.artifact_hash,
            status=CampaignLifecycle.PROMOTED,
            started_at=datetime.now(UTC),
            equity=100_000.0,
        )
    )
    assert store.capital_eligible_sleeves() == (sleeve,)
    with pytest.raises(DataError, match="immutable"):
        store.save_promoted_sleeve(sleeve.model_copy(update={"expected_net_return": 0.09}))


def test_worker_pauses_for_nightly_and_blocks_at_storage_cap(cfg, monkeypatch) -> None:
    worker = ResearchWorker(cfg)
    with nightly_lock(cfg.paths.artifacts_dir):
        assert worker.run_cycle() == []
        assert worker.status().state.value == "NIGHTLY_PAUSE"

    monkeypatch.setattr("edgestack.research.worker._factory_storage_gb", lambda _cfg: 75.0)
    assert worker.run_cycle() == []
    assert worker.status().state.value == "STORAGE_BLOCKED"
