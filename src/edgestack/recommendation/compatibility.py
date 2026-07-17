"""One-release legacy projections sourced only from the canonical bundle."""

from __future__ import annotations

from typing import Any

from edgestack.recommendation.schemas import CanonicalRecommendationBundleV2

DEPRECATION_NOTICE = (
    "Deprecated compatibility projection; migrate to /recommendations/latest. "
    "No legacy stock signal is actionable."
)


def board_projection(bundle: CanonicalRecommendationBundleV2) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "legacy_schema_version": 1,
        "as_of": bundle.as_of.date().isoformat(),
        "generated_at": bundle.generated_at.isoformat(),
        "status": bundle.default_recommendation.status.value,
        "disclaimer": bundle.disclaimer,
        "deprecated": True,
        "deprecation_notice": DEPRECATION_NOTICE,
        "rows": [],
        "watchlist_v2": [
            entry.model_dump(mode="json") for entry in bundle.base_recommendation.watchlist
        ],
        "canonical_bundle_hash": bundle.bundle_hash,
    }


def picks_projection(bundle: CanonicalRecommendationBundleV2) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "legacy_schema_version": 1,
        "as_of": bundle.as_of.date().isoformat(),
        "generated_at": bundle.generated_at.isoformat(),
        "status": bundle.default_recommendation.status.value,
        "disclaimer": bundle.disclaimer,
        "deprecated": True,
        "deprecation_notice": DEPRECATION_NOTICE,
        "picks": [],
        "watchlist_v2": [
            entry.model_dump(mode="json") for entry in bundle.base_recommendation.watchlist
        ],
        "canonical_bundle_hash": bundle.bundle_hash,
    }


def master_projection(bundle: CanonicalRecommendationBundleV2) -> dict[str, Any]:
    recommendation = bundle.default_recommendation
    return {
        "schema_version": 2,
        "legacy_schema_version": 1,
        "as_of": bundle.as_of.date().isoformat(),
        "execution_at": bundle.execution_at.isoformat(),
        "status": recommendation.status.value,
        "deprecated": True,
        "deprecation_notice": DEPRECATION_NOTICE,
        "effective_leverage": recommendation.effective_leverage,
        "instruments": {
            weight.symbol: {
                "canonical_target_weight": weight.weight,
                "asset_kind": weight.asset_kind.value,
                "sector": weight.sector,
            }
            for weight in recommendation.personalized_target_weights
        },
        "canonical_bundle_hash": bundle.bundle_hash,
        "disclaimer": bundle.disclaimer,
    }


def signals_projection(bundle: CanonicalRecommendationBundleV2) -> dict[str, Any]:
    """Keep the old shape but withdraw all legacy candidate actions."""
    return {
        "schema_version": 2,
        "as_of": bundle.as_of.isoformat(),
        "execution_at": bundle.execution_at.isoformat(),
        "deprecated": True,
        "deprecation_notice": DEPRECATION_NOTICE,
        "long_candidates": [],
        "short_candidates": [],
        "abstentions": [
            {
                "symbol": item.symbol,
                "reason": item.zero_weight_reason,
                "watchlist_only": True,
            }
            for item in bundle.base_recommendation.watchlist
        ],
        "canonical_target_weights": [
            weight.model_dump(mode="json")
            for weight in bundle.default_recommendation.personalized_target_weights
        ],
        "canonical_bundle_hash": bundle.bundle_hash,
        "disclaimer": bundle.disclaimer,
    }
