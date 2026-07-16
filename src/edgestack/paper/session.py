"""Atomic canonical paper-session entry point (simulated fills only)."""

from __future__ import annotations

from datetime import date
from typing import Any, cast

import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog
from edgestack.exceptions import DataError
from edgestack.execution.fills import Bar
from edgestack.paper.broker import get_broker
from edgestack.paper.canonical import (
    CanonicalPaperStateV2,
    corporate_action_inputs,
    execute_paper_session,
    load_monitoring_payload,
    load_paper_state,
)
from edgestack.recommendation.publication import AtomicRecommendationPublisher
from edgestack.recommendation.registry import RecommendationRegistry
from edgestack.recommendation.schemas import RiskStateV2
from edgestack.recommendation.service import CanonicalBundleRepository


def load_state(catalog: DataCatalog, cfg: EdgeStackConfig) -> CanonicalPaperStateV2:
    repository = CanonicalBundleRepository(catalog.artifacts_dir)
    if repository.pointer_path.exists():
        risk_state = repository.latest().default_recommendation.output_risk_state
    else:
        risk_state = RiskStateV2.initial(cfg.paper.initial_cash)
    return load_paper_state(
        repository,
        initial_equity=cfg.paper.initial_cash,
        risk_state=risk_state,
    )


def run_session(cfg: EdgeStackConfig, as_of: date | None = None) -> str:
    """Execute pending canonical targets at the next available open and republish atomically."""
    catalog = DataCatalog(cfg)
    repository = CanonicalBundleRepository(catalog.artifacts_dir)
    bundle = repository.latest()
    risk_inputs = repository.risk_inputs()
    state = load_paper_state(
        repository,
        initial_equity=cfg.paper.initial_cash,
        risk_state=bundle.default_recommendation.output_risk_state,
    )
    selected_session = as_of or bundle.execution_at.date()
    panel = catalog.load_panel(end=selected_session)
    day = panel.loc[pd.to_datetime(panel["date"]).dt.date == selected_session]
    if day.empty:
        raise DataError(f"{selected_session} is not a session with paper price data")
    bars: dict[str, Bar] = {}
    for row in day.itertuples(index=False):
        record = cast(Any, row)
        bars[str(record.symbol)] = Bar(
            session=pd.Timestamp(selected_session),
            open=float(record.open),
            high=float(record.high),
            low=float(record.low),
            close=float(record.close),
            volume=float(record.volume),
        )
    actions = catalog.load_corporate_actions(
        tuple(sorted(set(bars) | {position.symbol for position in state.positions})),
        start=state.last_session,
        end=selected_session,
    )
    updated = execute_paper_session(
        state,
        session=selected_session,
        bars=bars,
        corporate_actions=corporate_action_inputs(actions),
        broker=get_broker(cfg),
        base_funding_rate=risk_inputs.funding_rate,
        funding_spread_bps=bundle.default_risk_profile.funding_spread_bps,
    )
    publication = AtomicRecommendationPublisher(catalog.artifacts_dir).publish(
        bundle=bundle,
        risk_inputs=risk_inputs,
        paper_state=updated.model_dump(mode="json"),
        monitoring=load_monitoring_payload(repository),
    )
    RecommendationRegistry(catalog).save_publication(publication)
    catalog.audit(
        "paper_session_v2",
        reason=selected_session.isoformat(),
        equity=updated.current_equity,
        fills=len(updated.fills) - len(state.fills),
        run_id=publication.run_id,
    )
    latest_return = updated.realized_returns[-1]
    report = (
        f"PAPER V2 session {selected_session} — canonical simulated fills only, not advice\n"
        f"equity {updated.current_equity:,.2f} (cash {updated.cash:,.2f}, "
        f"{len(updated.positions)} positions, {len(updated.open_orders)} delayed/partial orders)\n"
        f"actual-fill return {latest_return.actual_fill_return:+.4%}; "
        f"costs {latest_return.transaction_costs:,.2f}; "
        f"dividends {latest_return.dividend_cash:,.2f}; "
        f"financing {latest_return.financing_cash_flow:+,.2f}"
    )
    print(report)
    return report
