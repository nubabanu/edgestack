"""Broker abstraction.

Any future real-broker integration must implement :class:`BrokerAdapter` and
be explicitly enabled — and the bundled configuration refuses
``live_trading_enabled: true`` outright, because this repository ships no
live implementation. Only :class:`SimulatedBroker` exists.
"""

from __future__ import annotations

import abc

from edgestack.config import EdgeStackConfig
from edgestack.exceptions import LiveTradingDisabledError
from edgestack.execution.costs import CostModel
from edgestack.execution.fills import Bar, FillSimulator
from edgestack.execution.orders import Fill, Order


class BrokerAdapter(abc.ABC):
    """Order-submission surface. Implementations must be explicitly configured."""

    is_live: bool = False

    @abc.abstractmethod
    def execute_against_bar(self, order: Order, bar: Bar, *,
                            at_open_phase: bool) -> Fill | None: ...


class SimulatedBroker(BrokerAdapter):
    """Fills orders against historical/synthetic bars via the FillSimulator."""

    is_live = False

    def __init__(self, cost_model: CostModel) -> None:
        self._simulator = FillSimulator(cost_model)

    def execute_against_bar(self, order: Order, bar: Bar, *,
                            at_open_phase: bool) -> Fill | None:
        return self._simulator.try_fill(order, bar, at_open_phase=at_open_phase)


def get_broker(cfg: EdgeStackConfig) -> BrokerAdapter:
    """The only broker EdgeStack will hand out is the simulated one."""
    if cfg.paper.live_trading_enabled:  # unreachable: config validation refuses it
        raise LiveTradingDisabledError(
            "live trading is not implemented in EdgeStack"
        )
    return SimulatedBroker(CostModel.from_config(cfg))
