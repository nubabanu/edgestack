"""Canonical recommendation engine.

Everything actionable in EdgeStack is represented by a versioned
``CanonicalRecommendationBundleV2``.  Legacy board/picks/master views are
projections of that bundle; they are not independent signal engines.
"""

from edgestack.recommendation.instrument_schemas import InstrumentAnalysisV2
from edgestack.recommendation.oil_schemas import OilDecisionSnapshotV2
from edgestack.recommendation.risk import RiskInputsV2, size_recommendation
from edgestack.recommendation.schemas import (
    BaseRecommendationV2,
    CanonicalRecommendationBundleV2,
    PortfolioRecommendationV2,
    RecommendationStatus,
    RiskProfileV2,
    RiskStateV2,
)

__all__ = [
    "BaseRecommendationV2",
    "CanonicalRecommendationBundleV2",
    "InstrumentAnalysisV2",
    "OilDecisionSnapshotV2",
    "PortfolioRecommendationV2",
    "RecommendationStatus",
    "RiskInputsV2",
    "RiskProfileV2",
    "RiskStateV2",
    "size_recommendation",
]
