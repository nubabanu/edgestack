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

    recheck = client.post(
        "/instruments/recheck",
        json={
            "previous_analysis": payload,
            "request": {
                "symbol": "AAA",
                "intended_entry_at": "2026-07-20T09:30:00-04:00",
                "round_trip_cost_bps": 12,
            },
        },
    )
    assert recheck.status_code == 200
    assert recheck.json()["previous_analysis_id"] == payload["analysis_id"]
    assert recheck.json()["analysis"]["recheck_plan"]["cadence_minutes"] == 60

    leaders = client.post(
        "/instruments/pattern-leaders",
        json={
            "symbols": ["AAA", "MISSING"],
            "resolution": "DAY",
            "horizon": "WEEK",
        },
    )
    assert leaders.status_code == 200
    assert leaders.json()["leaders"][0]["symbol"] == "AAA"
    assert leaders.json()["leaders"][0]["actionable"] is False
    assert leaders.json()["skipped_symbols"] == ["MISSING"]
    assert "watchlist research only" in leaders.json()["warning"]

    day_only = client.post(
        "/instruments/analyze",
        json={"symbol": "AAA", "intended_entry_date": "2026-07-21"},
    )
    assert day_only.status_code == 200
    assert all(
        item["resolution"] not in {"MINUTE_15", "HOUR"}
        for item in day_only.json()["chosen_time_ratings"]
    )
    assert any(item["resolution"] == "DAY" for item in day_only.json()["chosen_time_ratings"])


def test_oil_decision_api_is_stateless_broker_aware_and_never_actionable(
    cfg: EdgeStackConfig,
) -> None:
    catalog = DataCatalog(cfg)
    dates = pd.bdate_range(end=SESSION, periods=900)
    rng = np.random.default_rng(72026)
    for symbol in ("CL=F", "USO"):
        close = 100 * np.cumprod(1 + rng.normal(0.0001, 0.008, len(dates)))
        close *= 100 / close[-1]
        open_ = close * (1 + rng.normal(0, 0.001, len(dates)))
        catalog.write_bars(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": dates,
                    "open": open_,
                    "high": np.maximum(open_, close) * 1.004,
                    "low": np.minimum(open_, close) * 0.996,
                    "close": close,
                    "volume": 5_000_000.0,
                    "adj_close": close,
                }
            ),
            provider="fixture",
        )
    rows = []
    for session in pd.bdate_range(end=SESSION, periods=25):
        price = 100.0
        for timestamp in pd.date_range(
            pd.Timestamp(session.date(), tz="UTC") + pd.Timedelta(hours=13, minutes=30),
            periods=26,
            freq="15min",
        ):
            next_price = price * (1 + rng.normal(0, 0.001))
            rows.append(
                {
                    "symbol": "USO",
                    "timestamp": timestamp,
                    "interval_minutes": 15,
                    "open": price,
                    "high": max(price, next_price) * 1.001,
                    "low": min(price, next_price) * 0.999,
                    "close": next_price,
                    "volume": 50_000.0,
                }
            )
            price = next_price
    catalog.write_intraday_bars(pd.DataFrame(rows), provider="fixture")
    bundle = _publish_fixture(cfg, data_version=catalog.data_manifest_hash())
    api = TestClient(create_app(cfg))
    pointer = cfg.paths.artifacts_dir / "recommendations" / "current.json"
    before = pointer.read_bytes()
    observed_at = datetime.now(UTC)

    response = api.post(
        "/oil/decision",
        json={
            "broker_symbol": "OIL",
            "intended_entry_at": "2026-07-20T09:30:00-04:00",
            "quote": {
                "observed_at": observed_at.isoformat(),
                "bid": 99.98,
                "ask": 100.02,
                "offered_leverage": 10,
            },
            "modeled_leverage": 10,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["broker_symbol"] == "OIL"
    assert payload["canonical_bundle_hash"] == bundle.bundle_hash
    assert payload["status"] in {"OBSERVE", "BLOCKED"}
    assert payload["actionable"] is False
    assert payload["canonical_portfolio_weight"] == 0
    assert payload["analysis"]["overall_rating"] == "NOT_RATED"
    assert len(payload["friction_sensitivity"]) == 3
    assert len(payload["stress_table"]) == 9
    assert payload["stress_table"][-1]["equity_loss_fraction"] == 1.0
    assert "quantity" not in response.text
    assert "notional" not in response.text
    assert "/oil/decision" in api.get("/openapi.json").json()["paths"]
    assert pointer.read_bytes() == before


def test_sniper_api_is_version_bound_sized_and_structurally_non_actionable(
    cfg: EdgeStackConfig,
) -> None:
    catalog = DataCatalog(cfg)
    dates = pd.bdate_range(end=SESSION, periods=1_000)
    rng = np.random.default_rng(83)
    returns = rng.normal(0.00035, 0.004, len(dates))
    for index in range(240, len(returns) - 10, 20):
        returns[index : index + 3] = [-0.003, -0.004, -0.005]
        returns[index + 3 : index + 6] = [0.005, 0.004, 0.003]
    returns[-3:] = [-0.006, -0.007, -0.008]
    close = 100 * np.cumprod(1 + returns)
    open_ = close * (1 + rng.normal(0, 0.001, len(dates)))
    catalog.write_bars(
        pd.DataFrame(
            {
                "symbol": "SPY",
                "date": dates,
                "open": open_,
                "high": np.maximum(open_, close) * 1.003,
                "low": np.minimum(open_, close) * 0.997,
                "close": close,
                "volume": 10_000_000.0,
                "adj_close": close,
            }
        ),
        provider="fixture",
    )
    _publish_fixture(cfg, data_version=catalog.data_manifest_hash())
    client = TestClient(create_app(cfg))
    canonical_before = client.get("/recommendations/latest").json()

    latest = client.get("/sniper/latest")
    preview = client.post(
        "/sniper/preview",
        json={"account_equity": 200_000, "max_tolerable_loss": 400, "vehicle": "SPY"},
    )

    assert latest.status_code == preview.status_code == 200
    payload = preview.json()
    primary = payload["stage_1_candidates"][0]
    assert payload["account_equity"] == 200_000
    assert payload["max_tolerable_loss"] == 400
    assert primary["strategy_id"] == "C1_C2_PRIMARY"
    assert primary["status"] == "TRIGGERED_SHADOW"
    assert primary["actionable"] is False
    assert primary["sizing"]["capped_notional"] <= 200_000
    assert all(not item["can_initiate"] for item in payload["overlays"])
    assert len(payload["excluded_strategy_ids"]) == 5
    assert all(item["status"] == "BLOCKED" for item in payload["stage_2_candidates"])
    assert client.get("/recommendations/latest").json() == canonical_before
