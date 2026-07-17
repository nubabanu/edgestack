"""Canonical target-weight paper ledger with actual-fill return accounting."""

from __future__ import annotations

import json
from datetime import date
from typing import Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgestack.exceptions import DataError
from edgestack.execution.fills import Bar
from edgestack.execution.orders import Order, OrderType
from edgestack.paper.broker import BrokerAdapter
from edgestack.recommendation.financing import accrue_cash_financing
from edgestack.recommendation.hashing import stable_hash
from edgestack.recommendation.schemas import (
    AssetKind,
    PortfolioRecommendationV2,
    RiskStateV2,
    WeightV2,
)
from edgestack.recommendation.service import CanonicalBundleRepository


class PaperModelV2(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PaperTargetV2(PaperModelV2):
    target_id: str
    recommendation_hash: str
    issued_session: date
    execution_session: date
    weights: tuple[WeightV2, ...]


class PaperOrderIntentV2(PaperModelV2):
    order_intent_id: str
    target_id: str
    symbol: str
    created_session: date
    requested_quantity: float
    remaining_quantity: float
    status: Literal["PENDING", "DELAYED", "PARTIAL", "FILLED", "SUPERSEDED"]


class PaperFillV2(PaperModelV2):
    order_intent_id: str
    target_id: str
    symbol: str
    session: date
    quantity: float
    price: float = Field(gt=0)
    transaction_cost: float = Field(ge=0)


class PaperPositionV2(PaperModelV2):
    symbol: str
    quantity: float = Field(ge=0)
    average_fill_price: float = Field(gt=0)
    last_price: float = Field(gt=0)


class AppliedCorporateActionV2(PaperModelV2):
    symbol: str
    session: date
    action_type: Literal["dividend", "split"]
    value: float = Field(gt=0)
    cash_effect: float


class PaperReturnV2(PaperModelV2):
    session: date
    starting_equity: float = Field(gt=0)
    ending_equity: float = Field(gt=0)
    actual_fill_return: float
    transaction_costs: float = Field(ge=0)
    dividend_cash: float = Field(ge=0)
    financing_cash_flow: float
    gross_exposure_at_close: float = Field(ge=0)


class CorporateActionInputV2(PaperModelV2):
    symbol: str
    session: date
    action_type: Literal["dividend", "split"]
    value: float = Field(gt=0)


class CanonicalPaperStateV2(PaperModelV2):
    schema_version: Literal[2] = 2
    state_version: int = Field(default=0, ge=0)
    last_session: date | None = None
    cash: float
    current_equity: float = Field(gt=0)
    positions: tuple[PaperPositionV2, ...] = ()
    pending_target: PaperTargetV2 | None = None
    targets: tuple[PaperTargetV2, ...] = ()
    open_orders: tuple[PaperOrderIntentV2, ...] = ()
    order_intents: tuple[PaperOrderIntentV2, ...] = ()
    fills: tuple[PaperFillV2, ...] = ()
    realized_returns: tuple[PaperReturnV2, ...] = ()
    corporate_actions: tuple[AppliedCorporateActionV2, ...] = ()
    cumulative_transaction_costs: float = Field(default=0.0, ge=0)
    cumulative_dividends: float = Field(default=0.0, ge=0)
    cumulative_financing: float = 0.0
    risk_state: RiskStateV2

    @model_validator(mode="after")
    def _unique_positions(self) -> CanonicalPaperStateV2:
        symbols = [position.symbol for position in self.positions]
        if len(symbols) != len(set(symbols)):
            raise ValueError("paper positions must be unique by symbol")
        values = (self.cash, self.current_equity, self.cumulative_financing)
        if any(not np.isfinite(value) for value in values):
            raise ValueError("paper cash and equity values must be finite")
        return self

    @classmethod
    def initial(cls, equity: float, risk_state: RiskStateV2) -> CanonicalPaperStateV2:
        return cls(cash=equity, current_equity=equity, risk_state=risk_state)


def queue_recommendation_target(
    state: CanonicalPaperStateV2,
    recommendation: PortfolioRecommendationV2,
) -> CanonicalPaperStateV2:
    recommendation_hash = stable_hash(recommendation.model_dump(mode="json"))
    target_id = stable_hash(
        {
            "recommendation_hash": recommendation_hash,
            "issued_session": recommendation.as_of.date(),
            "execution_session": recommendation.execution_at.date(),
            "weights": recommendation.personalized_target_weights,
        }
    )
    if state.pending_target is not None and state.pending_target.target_id == target_id:
        return state
    target = PaperTargetV2(
        target_id=target_id,
        recommendation_hash=recommendation_hash,
        issued_session=recommendation.as_of.date(),
        execution_session=recommendation.execution_at.date(),
        weights=recommendation.personalized_target_weights,
    )
    return CanonicalPaperStateV2.model_validate(
        {
            **state.model_dump(),
            "state_version": state.state_version + 1,
            "pending_target": target,
            "targets": (*state.targets, target),
            "risk_state": recommendation.output_risk_state,
        }
    )


def execute_paper_session(
    state: CanonicalPaperStateV2,
    *,
    session: date,
    bars: dict[str, Bar],
    corporate_actions: tuple[CorporateActionInputV2, ...],
    broker: BrokerAdapter,
    base_funding_rate: float,
    funding_spread_bps: float,
) -> CanonicalPaperStateV2:
    if state.last_session is not None and session <= state.last_session:
        raise DataError(
            f"paper session {session} already processed (last was {state.last_session})"
        )
    starting_equity = state.current_equity
    cash = state.cash
    positions = {position.symbol: position for position in state.positions}
    financing = 0.0
    if state.last_session is not None:
        financing = accrue_cash_financing(
            cash,
            base_rate=base_funding_rate,
            funding_spread_bps=funding_spread_bps,
            start=state.last_session,
            end=session,
        )
        cash += financing

    applied_actions = list(state.corporate_actions)
    dividend_cash = 0.0
    for action in sorted(corporate_actions, key=lambda item: (item.session, item.symbol)):
        if action.session > session or (
            state.last_session is not None and action.session <= state.last_session
        ):
            continue
        position = positions.get(action.symbol)
        if position is None:
            continue
        cash_effect = 0.0
        if action.action_type == "dividend":
            cash_effect = position.quantity * action.value
            cash += cash_effect
            dividend_cash += cash_effect
        else:
            position = position.model_copy(
                update={
                    "quantity": position.quantity * action.value,
                    "average_fill_price": position.average_fill_price / action.value,
                    "last_price": position.last_price / action.value,
                }
            )
            positions[action.symbol] = position
        applied_actions.append(
            AppliedCorporateActionV2(
                symbol=action.symbol,
                session=action.session,
                action_type=action.action_type,
                value=action.value,
                cash_effect=cash_effect,
            )
        )

    open_orders = list(state.open_orders)
    order_history = list(state.order_intents)
    pending_target = state.pending_target
    if pending_target is not None and pending_target.execution_session <= session:
        order_history.extend(
            order.model_copy(update={"status": "SUPERSEDED"}) for order in open_orders
        )
        open_orders = []
        required_symbols = {
            weight.symbol
            for weight in pending_target.weights
            if weight.asset_kind is not AssetKind.CASH and abs(weight.weight) > 1e-12
        } | set(positions)
        if required_symbols.issubset(bars):
            open_orders = _orders_for_target(pending_target, session, cash, positions, bars)
            pending_target = None

    remaining_orders: list[PaperOrderIntentV2] = []
    fill_records = list(state.fills)
    transaction_costs = 0.0
    final_intents: list[PaperOrderIntentV2] = []
    for intent in open_orders:
        bar = bars.get(intent.symbol)
        if bar is None:
            delayed = intent.model_copy(update={"status": "DELAYED"})
            remaining_orders.append(delayed)
            final_intents.append(delayed)
            continue
        order = Order(
            symbol=intent.symbol,
            quantity=intent.remaining_quantity,
            order_type=OrderType.MARKET_ON_OPEN,
            created_session=bar.session,
            tag="canonical_target_rebalance",
        )
        fill = broker.execute_against_bar(order, bar, at_open_phase=True)
        if fill is None:
            delayed = intent.model_copy(update={"status": "DELAYED"})
            remaining_orders.append(delayed)
            final_intents.append(delayed)
            continue
        cash -= fill.quantity * fill.price + fill.cost
        transaction_costs += fill.cost
        _apply_fill_to_positions(positions, fill.symbol, fill.quantity, fill.price)
        remaining = intent.remaining_quantity - fill.quantity
        status = "FILLED" if abs(remaining) <= 1e-9 else "PARTIAL"
        updated = intent.model_copy(update={"remaining_quantity": remaining, "status": status})
        final_intents.append(updated)
        if status == "PARTIAL":
            remaining_orders.append(updated)
        fill_records.append(
            PaperFillV2(
                order_intent_id=intent.order_intent_id,
                target_id=intent.target_id,
                symbol=intent.symbol,
                session=session,
                quantity=fill.quantity,
                price=fill.price,
                transaction_cost=fill.cost,
            )
        )
    order_history.extend(final_intents)

    marked_positions: list[PaperPositionV2] = []
    market_value = 0.0
    gross_value = 0.0
    for symbol, position in sorted(positions.items()):
        if position.quantity <= 1e-9:
            continue
        last_price = bars[symbol].close if symbol in bars else position.last_price
        marked = position.model_copy(update={"last_price": last_price})
        marked_positions.append(marked)
        value = marked.quantity * last_price
        market_value += value
        gross_value += abs(value)
    ending_equity = cash + market_value
    if ending_equity <= 0:
        raise DataError("paper equity is non-positive after canonical execution")
    realized = PaperReturnV2(
        session=session,
        starting_equity=starting_equity,
        ending_equity=ending_equity,
        actual_fill_return=ending_equity / starting_equity - 1.0,
        transaction_costs=transaction_costs,
        dividend_cash=dividend_cash,
        financing_cash_flow=financing,
        gross_exposure_at_close=gross_value / ending_equity,
    )
    return CanonicalPaperStateV2(
        state_version=state.state_version + 1,
        last_session=session,
        cash=cash,
        current_equity=ending_equity,
        positions=tuple(marked_positions),
        pending_target=pending_target,
        targets=state.targets,
        open_orders=tuple(remaining_orders),
        order_intents=tuple(order_history),
        fills=tuple(fill_records),
        realized_returns=(*state.realized_returns, realized),
        corporate_actions=tuple(applied_actions),
        cumulative_transaction_costs=state.cumulative_transaction_costs + transaction_costs,
        cumulative_dividends=state.cumulative_dividends + dividend_cash,
        cumulative_financing=state.cumulative_financing + financing,
        risk_state=state.risk_state,
    )


def load_paper_state(
    repository: CanonicalBundleRepository,
    *,
    initial_equity: float,
    risk_state: RiskStateV2,
) -> CanonicalPaperStateV2:
    if not repository.pointer_path.exists():
        return CanonicalPaperStateV2.initial(initial_equity, risk_state)
    pointer = repository.pointer()
    repository.latest()  # checksum and atomic-version verification
    path = repository.run_dir(pointer) / "paper_state.json"
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        payload = wrapper["payload"]
        return CanonicalPaperStateV2.model_validate(payload)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError(f"invalid canonical paper state: {exc}") from exc


def load_monitoring_payload(repository: CanonicalBundleRepository) -> dict[str, object]:
    pointer = repository.pointer()
    repository.latest()
    path = repository.run_dir(pointer) / "monitoring.json"
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        payload = wrapper["payload"]
        if not isinstance(payload, dict):
            raise TypeError("monitoring payload is not an object")
        return payload
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError(f"invalid canonical monitoring state: {exc}") from exc


def corporate_action_inputs(frame) -> tuple[CorporateActionInputV2, ...]:
    return tuple(
        CorporateActionInputV2(
            symbol=str(row.symbol),
            session=row.date.date(),
            action_type=cast(Literal["dividend", "split"], str(row.action_type)),
            value=float(row.value),
        )
        for row in frame.itertuples(index=False)
    )


def _orders_for_target(
    target: PaperTargetV2,
    session: date,
    cash: float,
    positions: dict[str, PaperPositionV2],
    bars: dict[str, Bar],
) -> list[PaperOrderIntentV2]:
    opening_equity = cash + sum(
        position.quantity * (bars[symbol].open if symbol in bars else position.last_price)
        for symbol, position in positions.items()
    )
    target_weights = {
        weight.symbol: weight.weight
        for weight in target.weights
        if weight.asset_kind is not AssetKind.CASH
    }
    output: list[PaperOrderIntentV2] = []
    for symbol in sorted(set(target_weights) | set(positions)):
        bar = bars.get(symbol)
        reference_price = bar.open if bar is not None else positions[symbol].last_price
        target_quantity = target_weights.get(symbol, 0.0) * opening_equity / reference_price
        current_quantity = positions[symbol].quantity if symbol in positions else 0.0
        quantity = target_quantity - current_quantity
        if abs(quantity) <= 1e-9:
            continue
        intent_id = stable_hash(
            {
                "target_id": target.target_id,
                "symbol": symbol,
                "session": session,
                "quantity": round(quantity, 12),
            }
        )
        output.append(
            PaperOrderIntentV2(
                order_intent_id=intent_id,
                target_id=target.target_id,
                symbol=symbol,
                created_session=session,
                requested_quantity=quantity,
                remaining_quantity=quantity,
                status="PENDING",
            )
        )
    return output


def _apply_fill_to_positions(
    positions: dict[str, PaperPositionV2], symbol: str, quantity: float, price: float
) -> None:
    current = positions.get(symbol)
    current_quantity = current.quantity if current is not None else 0.0
    updated_quantity = current_quantity + quantity
    if updated_quantity < -1e-7:
        raise DataError(f"canonical long-only paper order would short {symbol}")
    if updated_quantity <= 1e-9:
        positions.pop(symbol, None)
        return
    if quantity > 0:
        old_cost = current_quantity * current.average_fill_price if current is not None else 0.0
        average = (old_cost + quantity * price) / updated_quantity
    else:
        if current is None:
            raise DataError(f"paper sell has no position for {symbol}")
        average = current.average_fill_price
    positions[symbol] = PaperPositionV2(
        symbol=symbol,
        quantity=updated_quantity,
        average_fill_price=average,
        last_price=price,
    )
