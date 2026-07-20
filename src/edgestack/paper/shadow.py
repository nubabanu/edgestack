"""Independent paper-shadow books with realistic simulated execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from edgestack.execution.fills import Bar
from edgestack.execution.orders import Fill, Order
from edgestack.paper.broker import SimulatedBroker


@dataclass(frozen=True)
class ShadowEquityPoint:
    session: date
    equity: float
    benchmark_equity: dict[str, float]


@dataclass
class ShadowBook:
    """A strategy-local book; it has no reference to the canonical portfolio."""

    strategy_id: str
    broker: SimulatedBroker
    initial_cash: float = 100_000.0
    cash: float = field(init=False)
    positions: dict[str, float] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    history: list[ShadowEquityPoint] = field(default_factory=list)
    _benchmark_equity: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.cash = self.initial_cash

    def execute(self, order: Order, bar: Bar, *, at_open_phase: bool) -> Fill | None:
        fill = self.broker.execute_against_bar(order, bar, at_open_phase=at_open_phase)
        if fill is None:
            return None
        self.cash -= fill.quantity * fill.price + fill.cost
        self.positions[fill.symbol] = self.positions.get(fill.symbol, 0.0) + fill.quantity
        self.fills.append(fill)
        return fill

    def mark_session(
        self,
        *,
        session: date,
        close_prices: dict[str, float],
        benchmark_returns: dict[str, float],
        cash_yield: float,
        financing_rate: float,
    ) -> ShadowEquityPoint:
        daily_rate = (cash_yield if self.cash >= 0 else financing_rate) / 252.0
        self.cash *= 1.0 + daily_rate
        missing = sorted(symbol for symbol in self.positions if symbol not in close_prices)
        if missing:
            raise ValueError(f"missing shadow close prices: {', '.join(missing)}")
        equity = self.cash + sum(
            quantity * close_prices[symbol] for symbol, quantity in self.positions.items()
        )
        for name, value in benchmark_returns.items():
            prior = self._benchmark_equity.get(name, self.initial_cash)
            self._benchmark_equity[name] = prior * (1.0 + value)
        point = ShadowEquityPoint(
            session=session,
            equity=equity,
            benchmark_equity=dict(sorted(self._benchmark_equity.items())),
        )
        self.history.append(point)
        return point


def shadow_books_are_isolated(*books: ShadowBook) -> bool:
    """Explicit invariant used at paper-publication boundaries."""
    return len({id(book.positions) for book in books}) == len(books)
