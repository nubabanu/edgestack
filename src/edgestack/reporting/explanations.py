"""Plain-English explanations for signal candidates.

Language discipline: everything is historical association, never causation,
never a promise. "Similar historical states produced..." — not "will produce".
"""

from __future__ import annotations

from edgestack.types import RegimeContext, SignalCandidate


def explain_candidate(candidate: SignalCandidate) -> str:
    lines = [
        f"{candidate.symbol} ranks as a {candidate.side.value} research candidate because:",
        "",
    ]
    for i, item in enumerate(candidate.evidence, start=1):
        lines.append(
            f"{i}. Validated edge `{item.edge}` ({item.family.value} family) applies: "
            f"historically {item.net_mean_return:+.2%} net per trade over "
            f"{item.validated_sample_size} out-of-sample signals (q={item.q_value:.3f}); "
            f"contributes {item.contribution:.0%} of the evidence."
        )
    lines.append("")
    lines.append(
        f"Similar historical states produced a positive net return in "
        f"{candidate.calibrated_probability_of_positive_net_return:.0%} of validated "
        f"cases over a {candidate.recommended_holding_sessions}-session horizon"
        + (" (calibrated probability)." if candidate.probability_is_calibrated
           else " (NOT independently calibrated — treat as approximate).")
    )
    lines.append(_regime_line(candidate.regime))
    if candidate.warnings:
        lines.append("")
        lines.append("Main risks and caveats:")
        for warning in candidate.warnings:
            lines.append(f"- {warning}")
    lines.append("")
    lines.append(
        "These are historical associations, not causal claims, and not "
        "investment advice."
    )
    return "\n".join(lines)


def _regime_line(regime: RegimeContext) -> str:
    return (
        f"Current market regime: {regime.market} (stock trend {regime.stock_trend}); "
        f"similarity to historically favorable regimes: {regime.similarity_score:.0%}."
    )
