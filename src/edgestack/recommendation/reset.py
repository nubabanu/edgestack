"""Explicit persisted drawdown-latch reset."""

from __future__ import annotations

from datetime import UTC, datetime

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog
from edgestack.exceptions import DataError
from edgestack.paper.canonical import (
    CanonicalPaperStateV2,
    load_monitoring_payload,
    load_paper_state,
    queue_recommendation_target,
)
from edgestack.recommendation.manifests import PublicationRecordV2
from edgestack.recommendation.publication import AtomicRecommendationPublisher
from edgestack.recommendation.registry import RecommendationRegistry
from edgestack.recommendation.risk import size_recommendation
from edgestack.recommendation.schemas import CanonicalRecommendationBundleV2
from edgestack.recommendation.service import CanonicalBundleRepository


def reset_persisted_risk_state(cfg: EdgeStackConfig) -> PublicationRecordV2:
    catalog = DataCatalog(cfg)
    repository = CanonicalBundleRepository(catalog.artifacts_dir)
    bundle = repository.latest()
    inputs = repository.risk_inputs()
    paper = load_paper_state(
        repository,
        initial_equity=bundle.default_risk_profile.account_equity,
        risk_state=bundle.default_recommendation.output_risk_state,
    )
    if not paper.risk_state.reset_eligible:
        raise DataError("persisted drawdown latch is not reset-eligible")
    recommendation = size_recommendation(
        base=bundle.base_recommendation,
        profile=bundle.default_risk_profile,
        state=paper.risk_state,
        inputs=inputs,
        reset_requested=True,
    )
    if recommendation.output_risk_state.cash_latched:
        raise DataError("persisted drawdown latch reset was denied")
    updated_bundle = CanonicalRecommendationBundleV2.model_validate(
        {
            **bundle.model_dump(),
            "generated_at": datetime.now(UTC),
            "default_recommendation": recommendation,
        }
    )
    updated_paper = CanonicalPaperStateV2.model_validate(
        {**paper.model_dump(), "risk_state": recommendation.output_risk_state}
    )
    updated_paper = queue_recommendation_target(updated_paper, recommendation)
    publication = AtomicRecommendationPublisher(catalog.artifacts_dir).publish(
        bundle=updated_bundle,
        risk_inputs=inputs,
        paper_state=updated_paper.model_dump(mode="json"),
        monitoring=load_monitoring_payload(repository),
    )
    RecommendationRegistry(catalog).save_publication(publication)
    catalog.audit(
        "risk_latch_reset_v2",
        reason=bundle.session.isoformat(),
        run_id=publication.run_id,
    )
    return publication
