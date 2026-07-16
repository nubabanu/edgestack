"""Portfolio allocators with hard limit enforcement.

Signals never become equal-sized positions automatically. Three allocators
(equal-risk, volatility-targeted, score-weighted) produce raw weights, then a
single enforcement pass guarantees:

- |w_i| <= max_position_weight
- sum |w_i| <= max_gross_exposure
- |sum w_i| <= max_net_exposure
- number of positions <= max_positions (keeps the highest-conviction ones)

Limits are guaranteed by construction and re-checked by a property test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edgestack.config import RiskConfig
from edgestack.exceptions import ValidationError
from edgestack.types import Side

SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class AllocationRequest:
    symbol: str
    side: Side
    conviction: float  # 0..100
    annualized_vol: float  # stock's annualized volatility estimate


def allocate(
    requests: list[AllocationRequest],
    risk: RiskConfig,
    method: str = "score_weighted",
) -> dict[str, float]:
    """Signed target weights per symbol (fraction of equity)."""
    if not requests:
        return {}
    if len({r.symbol for r in requests}) != len(requests):
        raise ValidationError("duplicate symbols in allocation requests")

    kept = sorted(requests, key=lambda r: -r.conviction)[: risk.max_positions]
    vols = np.array([max(r.annualized_vol, 0.05) for r in kept])
    scores = np.array([max(r.conviction - 50.0, 0.0) for r in kept])

    if method == "equal_risk":
        raw = 1.0 / vols
    elif method == "vol_target":
        # inverse-vol base scaled so the naive (independence) portfolio vol
        # estimate matches the target
        raw = 1.0 / vols
        raw = raw / raw.sum()
        naive_vol = float(np.sqrt(np.sum((raw * vols) ** 2)))
        scale = risk.target_annualized_volatility / max(naive_vol, 1e-9)
        raw = raw * scale
    elif method == "score_weighted":
        if scores.sum() <= 0:
            return {}
        raw = scores / vols
    else:
        raise ValidationError(f"unknown allocation method: {method}")

    if raw.sum() <= 0:
        return {}
    if method != "vol_target":
        raw = raw / raw.sum()  # normalize to full gross budget, capped below

    weights = {r.symbol: float(w if r.side is Side.LONG else -w) for r, w in zip(kept, raw)}
    return _enforce_limits(weights, risk)


def _enforce_limits(weights: dict[str, float], risk: RiskConfig) -> dict[str, float]:
    # per-position cap
    capped = {
        s: float(np.clip(w, -risk.max_position_weight, risk.max_position_weight))
        for s, w in weights.items()
    }
    # gross cap: scale everything down
    gross = sum(abs(w) for w in capped.values())
    if gross > risk.max_gross_exposure and gross > 0:
        factor = risk.max_gross_exposure / gross
        capped = {s: w * factor for s, w in capped.items()}
    # net cap: scale down the dominant side only
    net = sum(capped.values())
    if abs(net) > risk.max_net_exposure:
        heavy_side = 1.0 if net > 0 else -1.0
        heavy = {s: w for s, w in capped.items() if w * heavy_side > 0}
        light_sum = sum(w for w in capped.values() if w * heavy_side <= 0)
        target_heavy = heavy_side * risk.max_net_exposure - light_sum
        heavy_sum = sum(heavy.values())
        if heavy_sum != 0:
            factor = target_heavy / heavy_sum
            factor = float(np.clip(factor, 0.0, 1.0))
            for s in heavy:
                capped[s] = heavy[s] * factor
    return {s: w for s, w in capped.items() if abs(w) > 1e-9}
