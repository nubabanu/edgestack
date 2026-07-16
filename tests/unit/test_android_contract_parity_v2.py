"""The committed Android seed is a valid authoritative Python V2 bundle."""

from __future__ import annotations

from pathlib import Path

from edgestack.recommendation.schemas import CanonicalRecommendationBundleV2


def test_android_seed_validates_against_python_contract() -> None:
    path = Path("android/app/src/main/assets/seed/recommendation.json")
    bundle = CanonicalRecommendationBundleV2.model_validate_json(path.read_text(encoding="utf-8"))

    assert bundle.schema_version == 2
    assert sum(weight.weight for weight in bundle.base_recommendation.unlevered_base_weights) == 1
    assert bundle.default_recommendation.base_recommendation_weights == (
        bundle.base_recommendation.unlevered_base_weights
    )
    assert all(entry.prospective_sessions < 252 for entry in bundle.base_recommendation.watchlist)
