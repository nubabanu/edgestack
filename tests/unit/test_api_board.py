"""Canonical recommendation API and one-release compatibility contracts."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from edgestack.api.app import create_app
from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog
from edgestack.paper.canonical import CanonicalPaperStateV2, queue_recommendation_target
from edgestack.recommendation.policy import load_baseline_policy
from edgestack.recommendation.publication import AtomicRecommendationPublisher
from edgestack.recommendation.risk import RiskInputsV2, size_recommendation
from edgestack.recommendation.schemas import (
    AssetKind,
    BaseRecommendationV2,
    CanonicalRecommendationBundleV2,
    FreshnessV2,
    RecommendationStatus,
    RiskProfileV2,
    RiskStateV2,
    WatchlistEntryV2,
)

SESSION = date(2026, 7, 16)


def _publish_fixture(
    cfg: EdgeStackConfig, *, data_version: str = "data-v2"
) -> CanonicalRecommendationBundleV2:
    policy = load_baseline_policy()
    as_of = datetime(2026, 7, 16, 20, tzinfo=UTC)
    execution_at = datetime(2026, 7, 17, 13, 30, tzinfo=UTC)
    freshness = FreshnessV2(
        as_of=as_of,
        expected_session=SESSION,
        is_fresh=True,
        age_business_days=0,
        complete=True,
        compatible=True,
    )
    watchlist = (
        WatchlistEntryV2(
            symbol="AAA",
            asset_kind=AssetKind.STOCK,
            horizon_sessions=5,
            family="momentum",
            thesis="prospective shadow candidate",
            prospective_sessions=20,
            effective_resolved_outcomes=15,
            zero_weight_reason="requires 252 prospective sessions and ESS >= 100",
        ),
    )
    base = BaseRecommendationV2(
        status=RecommendationStatus.WATCHLIST_ONLY,
        as_of=as_of,
        execution_at=execution_at,
        artifact_version="artifact-v2",
        data_version=data_version,
        policy_version=policy.policy_version,
        baseline_weights=policy.weights,
        unlevered_base_weights=policy.weights,
        expected_net_return=0,
        expected_volatility=0.08,
        covariance_version="cov-v2",
        watchlist=watchlist,
        freshness=freshness,
    )
    inputs = RiskInputsV2(
        session=SESSION,
        stressed_forecast_volatility=0.10,
        parametric_995_one_day_loss=0.02,
        historical_995_one_day_loss=0.025,
        bootstrapped_99_path_drawdown=0.08,
        liquidity_position_limits={weight.symbol: 5.0 for weight in policy.weights},
        funding_rate=0.04,
        funding_rate_as_of=SESSION,
    )
    profile = RiskProfileV2()
    recommendation = size_recommendation(
        base=base,
        profile=profile,
        state=RiskStateV2.initial(profile.account_equity),
        inputs=inputs,
    )
    bundle = CanonicalRecommendationBundleV2(
        generated_at=datetime(2026, 7, 16, 20, 5, tzinfo=UTC),
        session=SESSION,
        as_of=as_of,
        execution_at=execution_at,
        data_version=base.data_version,
        artifact_version=base.artifact_version,
        policy_version=base.policy_version,
        baseline_policy=policy,
        default_risk_profile=profile,
        base_recommendation=base,
        default_recommendation=recommendation,
    )
    AtomicRecommendationPublisher(cfg.paths.artifacts_dir).publish(
        bundle=bundle,
        risk_inputs=inputs,
        paper_state=queue_recommendation_target(
            CanonicalPaperStateV2.initial(profile.account_equity, recommendation.output_risk_state),
            recommendation,
        ).model_dump(mode="json"),
        monitoring={"healthy": True},
    )
    return bundle


@pytest.fixture()
def client(cfg: EdgeStackConfig) -> TestClient:
    return TestClient(create_app(cfg))


@pytest.fixture()
def canonical_client(cfg: EdgeStackConfig) -> tuple[TestClient, CanonicalRecommendationBundleV2]:
    bundle = _publish_fixture(cfg)
    return TestClient(create_app(cfg)), bundle


def test_canonical_and_compatibility_endpoints_fail_closed_before_publication(
    client: TestClient,
) -> None:
    for path in ("/recommendations/latest", "/board", "/picks", "/master"):
        response = client.get(path)
        assert response.status_code == 404
        assert "canonical recommendation has not been published" in response.json()["detail"]


def test_paper_missing_is_404(client: TestClient) -> None:
    response = client.get("/paper")
    assert response.status_code == 404
    assert "no canonical paper state" in response.json()["detail"]


def test_latest_returns_one_verified_atomic_bundle(
    canonical_client: tuple[TestClient, CanonicalRecommendationBundleV2],
) -> None:
    client, expected = canonical_client
    response = client.get("/recommendations/latest")

    assert response.status_code == 200
    assert response.json()["data_version"] == expected.data_version
    assert response.json()["base_recommendation"]["status"] == "WATCHLIST_ONLY"
    assert response.json()["default_recommendation"]["effective_leverage"] == 0.25


def test_paper_reads_the_state_from_the_same_atomic_run(
    canonical_client: tuple[TestClient, CanonicalRecommendationBundleV2],
) -> None:
    client, _bundle = canonical_client
    response = client.get("/paper")

    assert response.status_code == 200
    assert response.json()["state"]["schema_version"] == 2
    assert response.json()["state"]["pending_target"] is not None
    assert response.json()["equity_history"] == []


def test_legacy_board_and_picks_are_empty_watchlist_only_projections(
    canonical_client: tuple[TestClient, CanonicalRecommendationBundleV2],
) -> None:
    client, bundle = canonical_client
    board = client.get("/board")
    picks = client.get("/picks")

    assert board.status_code == picks.status_code == 200
    assert board.headers["Deprecation"] == "true"
    assert board.json()["rows"] == []
    assert picks.json()["picks"] == []
    assert board.json()["watchlist_v2"][0]["symbol"] == "AAA"
    assert board.json()["canonical_bundle_hash"] == bundle.bundle_hash


def test_legacy_master_contains_only_canonical_target_weights(
    canonical_client: tuple[TestClient, CanonicalRecommendationBundleV2],
) -> None:
    client, bundle = canonical_client
    response = client.get("/master")

    assert response.status_code == 200
    assert response.json()["canonical_bundle_hash"] == bundle.bundle_hash
    assert response.json()["instruments"]["SPY"]["canonical_target_weight"] == 0.0625
    assert "ensemble_exposure_next_session" not in response.text


def test_preview_is_stateless_and_cannot_change_base_selection(
    canonical_client: tuple[TestClient, CanonicalRecommendationBundleV2],
    cfg: EdgeStackConfig,
) -> None:
    client, bundle = canonical_client
    pointer = cfg.paths.artifacts_dir / "recommendations" / "current.json"
    before = pointer.read_bytes()
    response = client.post(
        "/recommendations/preview",
        json={
            "profile": {
                "maximum_gross_leverage": 0.10,
                "target_volatility": 0.05,
                "maximum_drawdown": 0.10,
                "funding_spread_bps": 800,
                "per_stock_cap": 0.01,
                "sector_cap": 0.10,
                "account_equity": 100000,
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["effective_leverage"] == 0.10
    assert payload["base_recommendation_weights"] == [
        weight.model_dump(mode="json")
        for weight in bundle.base_recommendation.unlevered_base_weights
    ]
    assert pointer.read_bytes() == before
    assert client.get("/recommendations/latest").json()[
        "default_risk_profile"
    ] == bundle.default_risk_profile.model_dump(mode="json")


def test_legacy_routes_are_marked_deprecated_in_openapi(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/board", "/picks", "/master", "/signals/latest"):
        assert paths[path]["get"]["deprecated"] is True


def test_latest_rejects_a_tampered_bundle(
    canonical_client: tuple[TestClient, CanonicalRecommendationBundleV2],
    cfg: EdgeStackConfig,
) -> None:
    client, _bundle = canonical_client
    root = cfg.paths.artifacts_dir / "recommendations"
    pointer = json.loads((root / "current.json").read_text(encoding="utf-8"))
    path = root / "runs" / pointer["run_id"] / "recommendation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["base_recommendation"]["warnings"] = ["tampered after publication"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    response = client.get("/recommendations/latest")
    assert response.status_code == 404
    assert "publication checksum mismatch" in response.json()["detail"]


def test_instrument_analysis_is_version_bound_and_non_promotional(cfg: EdgeStackConfig) -> None:
    catalog = DataCatalog(cfg)
    dates = pd.bdate_range(end=SESSION, periods=900)
    rng = np.random.default_rng(7)
    close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, len(dates)))
    open_ = close * (1 + rng.normal(0, 0.001, len(dates)))
    catalog.write_bars(
        pd.DataFrame(
            {
                "symbol": "AAA",
                "date": dates,
                "open": open_,
                "high": np.maximum(open_, close) * 1.005,
                "low": np.minimum(open_, close) * 0.995,
                "close": close,
                "volume": 2_000_000.0,
                "adj_close": close,
            }
        ),
        provider="fixture",
    )
    _publish_fixture(cfg, data_version=catalog.data_manifest_hash())
    client = TestClient(create_app(cfg))

    response = client.post(
        "/instruments/analyze",
        json={
            "symbol": "AAA",
            "intended_entry_at": "2026-07-20T09:30:00-04:00",
            "round_trip_cost_bps": 12,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["resolution"]["resolved_symbol"] == "AAA"
    assert payload["status"] == "INSUFFICIENT_EVIDENCE"
    assert payload["overall_rating"] == "NOT_RATED"
    assert payload["alignment"]["aligned_trade"] is False
    assert payload["horizon_analyses"][0]["best_window"] is None
    assert payload["horizon_analyses"][1]["best_window"]["actionable"] is False
    assert len(catalog.audit_events("test_set_accessed")) == 1
