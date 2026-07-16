"""Atomic, content-addressed recommendation publication tests."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from edgestack.recommendation.policy import load_baseline_policy
from edgestack.recommendation.publication import AtomicRecommendationPublisher
from edgestack.recommendation.risk import RiskInputsV2, size_recommendation
from edgestack.recommendation.schemas import (
    BaseRecommendationV2,
    CanonicalRecommendationBundleV2,
    FreshnessV2,
    RecommendationStatus,
    RiskProfileV2,
    RiskStateV2,
)
from edgestack.recommendation.service import CanonicalBundleRepository


def _bundle(
    session: date,
    *,
    data_version: str,
    generated_minute: int = 5,
) -> tuple[CanonicalRecommendationBundleV2, RiskInputsV2]:
    policy = load_baseline_policy()
    as_of = datetime(session.year, session.month, session.day, 20, tzinfo=UTC)
    execution_at = as_of + timedelta(days=1)
    freshness = FreshnessV2(
        as_of=as_of,
        expected_session=session,
        is_fresh=True,
        age_business_days=0,
        complete=True,
        compatible=True,
    )
    base = BaseRecommendationV2(
        status=RecommendationStatus.BASELINE_ONLY,
        as_of=as_of,
        execution_at=execution_at,
        artifact_version="baseline-artifact-v1",
        data_version=data_version,
        policy_version=policy.policy_version,
        baseline_weights=policy.weights,
        unlevered_base_weights=policy.weights,
        covariance_version="cov-v1",
        expected_volatility=0.08,
        freshness=freshness,
    )
    inputs = RiskInputsV2(
        session=session,
        stressed_forecast_volatility=0.10,
        parametric_995_one_day_loss=0.02,
        historical_995_one_day_loss=0.025,
        bootstrapped_99_path_drawdown=0.08,
        liquidity_position_limits={weight.symbol: 5.0 for weight in policy.weights},
        funding_rate=0.04,
        funding_rate_as_of=session,
    )
    profile = RiskProfileV2()
    recommendation = size_recommendation(
        base=base,
        profile=profile,
        state=RiskStateV2.initial(profile.account_equity),
        inputs=inputs,
    )
    return (
        CanonicalRecommendationBundleV2(
            generated_at=datetime(
                session.year,
                session.month,
                session.day,
                20,
                generated_minute,
                tzinfo=UTC,
            ),
            session=session,
            as_of=as_of,
            execution_at=execution_at,
            data_version=data_version,
            artifact_version=base.artifact_version,
            policy_version=policy.policy_version,
            baseline_policy=policy,
            default_risk_profile=profile,
            base_recommendation=base,
            default_recommendation=recommendation,
        ),
        inputs,
    )


def test_atomic_publication_writes_one_verified_version_set(tmp_path: Path) -> None:
    bundle, inputs = _bundle(date(2026, 7, 16), data_version="data-v1")
    publisher = AtomicRecommendationPublisher(tmp_path)
    record = publisher.publish(
        bundle=bundle,
        risk_inputs=inputs,
        paper_state={"status": "initialized"},
        monitoring={"healthy": True},
    )
    repository = CanonicalBundleRepository(tmp_path)
    run = repository.run_dir()

    assert repository.latest() == bundle
    assert repository.risk_inputs() == inputs
    assert run.name == record.run_id
    assert (run / "checksums.json").exists()
    assert (run / "compatibility" / "board.json").exists()
    for name in ("paper_state.json", "monitoring.json"):
        payload = json.loads((run / name).read_text(encoding="utf-8"))
        assert payload["session"] == bundle.session.isoformat()
        assert payload["data_version"] == bundle.data_version
        assert payload["artifact_version"] == bundle.artifact_version


def test_failure_before_pointer_swap_leaves_previous_publication_current(tmp_path: Path) -> None:
    publisher = AtomicRecommendationPublisher(tmp_path)
    first_bundle, first_inputs = _bundle(date(2026, 7, 16), data_version="data-v1")
    publisher.publish(
        bundle=first_bundle,
        risk_inputs=first_inputs,
        paper_state={"status": "first"},
        monitoring={"healthy": True},
    )
    pointer = tmp_path / "recommendations" / "current.json"
    before = pointer.read_bytes()
    second_bundle, second_inputs = _bundle(date(2026, 7, 17), data_version="data-v2")

    def fail(_run: Path) -> None:
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        publisher.publish(
            bundle=second_bundle,
            risk_inputs=second_inputs,
            paper_state={"status": "second"},
            monitoring={"healthy": True},
            before_pointer_swap=fail,
        )

    assert pointer.read_bytes() == before
    assert CanonicalBundleRepository(tmp_path).latest() == first_bundle


def test_semantically_identical_publication_is_idempotent_across_generation_time(
    tmp_path: Path,
) -> None:
    first, inputs = _bundle(date(2026, 7, 16), data_version="data-v1", generated_minute=5)
    second, _ = _bundle(date(2026, 7, 16), data_version="data-v1", generated_minute=6)
    publisher = AtomicRecommendationPublisher(tmp_path)
    first_record = publisher.publish(
        bundle=first,
        risk_inputs=inputs,
        paper_state={"status": "same"},
        monitoring={"healthy": True},
    )
    second_record = publisher.publish(
        bundle=second,
        risk_inputs=inputs,
        paper_state={"status": "same"},
        monitoring={"healthy": True},
    )

    assert first.bundle_hash == second.bundle_hash
    assert first_record == second_record
    assert len(list((tmp_path / "recommendations" / "runs").iterdir())) == 1
