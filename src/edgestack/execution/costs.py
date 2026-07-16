"""Transaction-cost model with explicit scenarios.

Components (all configurable, in basis points unless noted):
commission, half bid-ask spread, base slippage, market-impact (scaled by a
participation proxy), SEC/exchange fee on sells, and an annualized short
borrow fee. Scenario multipliers scale the market-friction components:

    OPTIMISTIC   0.5x
    BASE         1.0x
    CONSERVATIVE 1.5x   (default everywhere)
    STRESS       3.0x

Costs are monotonically non-decreasing across that ordering — enforced by a
property test — so a strategy that only works under OPTIMISTIC costs can
never look better under CONSERVATIVE ones.
"""

from __future__ import annotations

from dataclasses import dataclass

from edgestack.config import EdgeStackConfig
from edgestack.types import CostScenario, Side

SCENARIO_MULTIPLIER: dict[CostScenario, float] = {
    CostScenario.OPTIMISTIC: 0.5,
    CostScenario.BASE: 1.0,
    CostScenario.CONSERVATIVE: 1.5,
    CostScenario.STRESS: 3.0,
}

SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class CostModel:
    scenario: CostScenario
    commission_bps: float
    half_spread_bps: float
    base_slippage_bps: float
    impact_coeff_bps: float
    short_borrow_annualized: float
    sec_fee_bps: float

    @classmethod
    def from_config(cls, cfg: EdgeStackConfig, scenario: CostScenario | None = None) -> CostModel:
        return cls(
            scenario=scenario or cfg.costs.scenario,
            commission_bps=cfg.costs.commission_bps,
            half_spread_bps=cfg.costs.half_spread_bps,
            base_slippage_bps=cfg.costs.base_slippage_bps,
            impact_coeff_bps=cfg.costs.impact_coeff_bps,
            short_borrow_annualized=cfg.costs.short_borrow_annualized,
            sec_fee_bps=cfg.costs.sec_fee_bps,
        )

    @property
    def multiplier(self) -> float:
        return SCENARIO_MULTIPLIER[self.scenario]

    def one_way_cost_bps(self, *, is_sell: bool, participation: float = 0.0) -> float:
        """Cost of a single execution leg in bps of notional.

        ``participation`` is a 0..1 proxy for order size relative to typical
        volume; impact scales linearly with it.
        """
        friction = (
            self.half_spread_bps
            + self.base_slippage_bps
            + self.impact_coeff_bps * max(0.0, min(participation, 1.0))
        )
        fee = self.sec_fee_bps if is_sell else 0.0
        return self.commission_bps + fee + self.multiplier * friction

    def roundtrip_cost(
        self, side: Side, holding_sessions: int, participation: float = 0.0
    ) -> float:
        """Total round-trip cost as a return fraction (not bps).

        For shorts this includes the borrow fee accrued over the holding
        period, scenario-scaled: borrow availability is uncertain, so stressed
        scenarios assume more expensive borrow.
        """
        entry_sell = side is Side.SHORT
        legs_bps = self.one_way_cost_bps(
            is_sell=entry_sell, participation=participation
        ) + self.one_way_cost_bps(is_sell=not entry_sell, participation=participation)
        cost = legs_bps / 1e4
        if side is Side.SHORT:
            borrow = self.short_borrow_annualized * self.multiplier
            cost += borrow * holding_sessions / SESSIONS_PER_YEAR
        return cost

    def net_return(
        self, gross_return: float, side: Side, holding_sessions: int, participation: float = 0.0
    ) -> float:
        """Net return of a completed trade: directional gross minus costs."""
        directional = gross_return if side is Side.LONG else -gross_return
        return directional - self.roundtrip_cost(side, holding_sessions, participation)
