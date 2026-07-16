"""Evidence aggregation with family-level anti-double-counting.

Five momentum indicators agreeing is barely more evidence than one: within an
evidence family the aggregate contribution is ``max + lambda * mean(rest)``,
and each family's share of the total is capped. Only across genuinely
different families do contributions add up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edgestack.config import ScoringConfig
from edgestack.types import Edge, EvidenceItem, Family


@dataclass(frozen=True)
class AggregatedEvidence:
    items: tuple[EvidenceItem, ...]
    expected_net_return: float  # contribution-weighted mean of edge nets
    total_contribution: float  # 0..1-ish aggregate evidence mass
    stability: float  # contribution-weighted stability
    out_of_sample: float
    deflated_sharpe: float
    cost_survival: float
    effective_n: float
    ci_width: float
    tail_risk: float
    recommended_horizon: int


def _edge_strength(edge: Edge) -> float:
    """Raw evidence strength of one edge: economics x reliability, bounded."""
    econ = min(1.0, max(0.0, edge.stats.net_mean_return / 0.01))
    reliability = (
        edge.robustness.stability_score
        + edge.stats.deflated_sharpe_ratio
        + edge.stats.bayesian_posterior_probability
    ) / 3.0
    return econ * reliability


def aggregate_evidence(edges: list[Edge], scoring: ScoringConfig) -> AggregatedEvidence | None:
    """Family-capped aggregation of the matched, validated edges for one signal."""
    if not edges:
        return None

    strengths = {e.identity.edge_id: _edge_strength(e) for e in edges}

    # Within-family: max + lambda * mean(rest); then cap each family's share.
    by_family: dict[Family, list[Edge]] = {}
    for edge in edges:
        by_family.setdefault(edge.identity.family, []).append(edge)

    family_contrib: dict[Family, float] = {}
    edge_contrib: dict[str, float] = {}
    for family, members in by_family.items():
        vals = sorted((strengths[e.identity.edge_id] for e in members), reverse=True)
        best, rest = vals[0], vals[1:]
        combined = best + scoring.within_family_lambda * float(np.mean(rest)) if rest else best
        # Hard per-family cap: no single family can dominate the evidence stack.
        capped = min(combined, scoring.family_cap)
        family_contrib[family] = capped
        # distribute the CAPPED family contribution over members by strength
        total_strength = sum(strengths[e.identity.edge_id] for e in members) or 1.0
        for e in members:
            edge_contrib[e.identity.edge_id] = (
                capped * strengths[e.identity.edge_id] / total_strength
            )

    total = sum(family_contrib.values())
    if total <= 0:
        return None

    def wmean(values: list[float]) -> float:
        weights = [edge_contrib[e.identity.edge_id] for e in edges]
        return float(np.average(values, weights=weights))

    items = tuple(
        EvidenceItem(
            edge=e.identity.name,
            family=e.identity.family,
            contribution=round(edge_contrib[e.identity.edge_id] / total, 4),
            validated_sample_size=e.stats.sample_size,
            q_value=e.stats.q_value,
            net_mean_return=e.stats.net_mean_return,
        )
        for e in sorted(edges, key=lambda e: -edge_contrib[e.identity.edge_id])
    )

    # Horizon: the strongest edge's horizon wins (conservative expected
    # utility per edge is already baked into _edge_strength).
    best_edge = max(edges, key=lambda e: edge_contrib[e.identity.edge_id])

    ci_widths = [
        e.stats.bootstrap_confidence_interval[1] - e.stats.bootstrap_confidence_interval[0]
        for e in edges
    ]
    return AggregatedEvidence(
        items=items,
        expected_net_return=wmean([e.stats.net_mean_return for e in edges]),
        total_contribution=float(min(1.0, total)),
        stability=wmean([e.robustness.stability_score for e in edges]),
        out_of_sample=wmean([e.robustness.out_of_sample_score for e in edges]),
        deflated_sharpe=wmean([e.stats.deflated_sharpe_ratio for e in edges]),
        cost_survival=wmean([e.robustness.cost_robustness_score for e in edges]),
        effective_n=float(min(e.stats.effective_sample_size for e in edges)),
        ci_width=float(
            np.average(ci_widths, weights=[edge_contrib[e.identity.edge_id] for e in edges])
        ),
        tail_risk=float(min(e.stats.expected_shortfall for e in edges)),
        recommended_horizon=best_edge.identity.holding_horizon,
    )
