"""The 0-100 conviction score.

A conviction score is NOT a probability. It summarizes whether a signal is
trustworthy and economically actionable:

    raw  = base_edge x reliability x applicability x tradability - risk_penalty
    shrunk = raw * ess / (ess + shrinkage_min_sample)        (toward neutral)
    score  = 100 * logistic(slope * shrunk)                  (0 raw -> 50)

Every component is pre-bounded to [0, 1] (penalty to [0, 0.5]); the function
is pure and returns all intermediates so explanations can show the math.
A score of 50 is neutral "no usable edge".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: expected net return (per trade) that maps the economic component to 1.0
NET_RETURN_SCALE = 0.01
#: CI width (relative to |expected net|) beyond which uncertainty is maximal
UNCERTAINTY_SCALE = 6.0
#: per-trade expected shortfall magnitude treated as maximal tail risk
TAIL_SCALE = 0.10


@dataclass(frozen=True)
class ConvictionInputs:
    calibrated_probability: float  # P(net > 0), 0..1
    expected_net_return: float  # per trade, signed
    expected_risk: float  # per-trade return std, >= 0
    effective_sample_size: float
    stability_score: float  # fraction of positive OOS folds, 0..1
    out_of_sample_score: float  # 0..1
    deflated_sharpe: float  # 0..1
    regime_similarity: float  # 0..1
    liquidity_score: float  # 0..1
    data_quality_score: float  # 0..1
    cost_survival_fraction: float  # scenarios survived / total, 0..1
    ci_width: float  # bootstrap CI width of the net mean
    tail_risk: float  # |expected shortfall| per trade
    model_edge_disagreement: float = 0.0  # |model P - edge posterior|, 0..1


@dataclass(frozen=True)
class ConvictionResult:
    score: float
    components: dict[str, float] = field(default_factory=dict)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def conviction(
    inputs: ConvictionInputs,
    *,
    shrinkage_min_sample: int = 30,
    logistic_slope: float = 6.0,
) -> ConvictionResult:
    p = _clip01(inputs.calibrated_probability)
    prob_edge = _clip01(2.0 * (p - 0.5))  # 0 at coin flip, 1 at certainty
    econ_edge = _clip01(inputs.expected_net_return / NET_RETURN_SCALE)
    base_edge = prob_edge * econ_edge

    reliability = _clip01(
        (inputs.stability_score + inputs.out_of_sample_score + inputs.deflated_sharpe) / 3.0
    )
    applicability = _clip01(inputs.regime_similarity)
    tradability = _clip01(
        (inputs.liquidity_score + inputs.data_quality_score + inputs.cost_survival_fraction) / 3.0
    )

    denom = max(abs(inputs.expected_net_return), 1e-6)
    uncertainty = _clip01(inputs.ci_width / (UNCERTAINTY_SCALE * denom))
    tail = _clip01(abs(inputs.tail_risk) / TAIL_SCALE)
    disagreement = _clip01(inputs.model_edge_disagreement)
    risk_penalty = 0.5 * _clip01(0.4 * uncertainty + 0.3 * tail + 0.3 * disagreement)

    raw = base_edge * reliability * applicability * tradability - risk_penalty

    ess = max(0.0, inputs.effective_sample_size)
    shrink = ess / (ess + max(1, shrinkage_min_sample))
    raw_shrunk = raw * shrink

    score = float(100.0 / (1.0 + np.exp(-logistic_slope * raw_shrunk)))
    score = float(min(100.0, max(0.0, score)))

    return ConvictionResult(
        score=score,
        components={
            "base_edge": base_edge,
            "prob_edge": prob_edge,
            "econ_edge": econ_edge,
            "reliability": reliability,
            "applicability": applicability,
            "tradability": tradability,
            "risk_penalty": risk_penalty,
            "uncertainty": uncertainty,
            "tail": tail,
            "disagreement": disagreement,
            "shrink": shrink,
            "raw": raw,
            "raw_shrunk": raw_shrunk,
        },
    )
