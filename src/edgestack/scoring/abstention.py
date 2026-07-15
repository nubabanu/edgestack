"""Abstention gates: the explicit NO-TRADE decision with recorded reasons.

A high-quality system abstains often. Every symbol that does not become a
candidate gets an Abstention record saying exactly why.
"""

from __future__ import annotations

from edgestack.config import EdgeStackConfig


def gate_reasons(
    *,
    cfg: EdgeStackConfig,
    price: float,
    median_dollar_volume: float,
    data_quality_score: float,
    n_matched_edges: int,
    calibrated_probability: float | None,
    expected_net_return: float | None,
    conviction_score: float | None,
    reward_to_risk: float | None,
    effective_sample_size: float | None,
) -> list[str]:
    """Every failed gate, not just the first — the user should see all of them."""
    reasons: list[str] = []
    if price < cfg.universe.min_price:
        reasons.append(f"price {price:.2f} below minimum {cfg.universe.min_price}")
    if median_dollar_volume < cfg.universe.min_median_dollar_volume:
        reasons.append(
            f"median dollar volume {median_dollar_volume:,.0f} below minimum "
            f"{cfg.universe.min_median_dollar_volume:,.0f}"
        )
    if data_quality_score < 0.7:
        reasons.append(f"data quality {data_quality_score:.0%} insufficient")
    if n_matched_edges == 0:
        reasons.append("no validated edge applies to the current state")
        return reasons  # nothing below is defined without evidence
    if effective_sample_size is not None and (
        effective_sample_size < cfg.signals.min_effective_sample_size
    ):
        reasons.append(
            f"effective sample size {effective_sample_size:.0f} below "
            f"{cfg.signals.min_effective_sample_size}"
        )
    if calibrated_probability is not None and (
        calibrated_probability < cfg.signals.min_probability_of_profit
    ):
        reasons.append(
            f"P(net>0) {calibrated_probability:.2f} below threshold "
            f"{cfg.signals.min_probability_of_profit}"
        )
    if expected_net_return is not None and (
        expected_net_return < cfg.signals.min_expected_net_return
    ):
        reasons.append(
            f"expected net return {expected_net_return:.4f} below floor "
            f"{cfg.signals.min_expected_net_return}"
        )
    if conviction_score is not None and conviction_score < cfg.signals.min_conviction_score:
        reasons.append(
            f"conviction {conviction_score:.0f} below minimum "
            f"{cfg.signals.min_conviction_score:.0f}"
        )
    if reward_to_risk is not None and reward_to_risk < cfg.signals.min_reward_to_risk:
        reasons.append(
            f"reward-to-risk {reward_to_risk:.2f} below minimum "
            f"{cfg.signals.min_reward_to_risk}"
        )
    return reasons
