"""Canonical baseline assembly used by the fail-fast nightly orchestrator."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.calendar import TradingCalendar
from edgestack.data.catalog import DataCatalog
from edgestack.data.quality import assess_panel
from edgestack.exceptions import DataError
from edgestack.execution.fills import Bar
from edgestack.paper.broker import get_broker
from edgestack.paper.canonical import (
    CanonicalPaperStateV2,
    corporate_action_inputs,
    execute_paper_session,
    load_paper_state,
    queue_recommendation_target,
)
from edgestack.recommendation.financing import FundingRateObservation, fetch_dgs3mo
from edgestack.recommendation.growth import annualized_log_growth, financed_risk_matched_spy
from edgestack.recommendation.hashing import code_revision, stable_hash
from edgestack.recommendation.instrument_schemas import NewsEvidenceV2
from edgestack.recommendation.manifests import PublicationRecordV2
from edgestack.recommendation.policy import load_baseline_policy
from edgestack.recommendation.portfolio import (
    AssetMetadata,
    ExpectedReturnEstimate,
    build_base_recommendation,
)
from edgestack.recommendation.publication import AtomicRecommendationPublisher
from edgestack.recommendation.registry import RecommendationRegistry
from edgestack.recommendation.risk import estimate_risk_inputs, size_recommendation
from edgestack.recommendation.schemas import (
    AssetKind,
    CanonicalRecommendationBundleV2,
    EvidenceGrade,
    FreshnessV2,
    RecommendationStatus,
    RiskProfileV2,
    RiskStateV2,
    WatchlistEntryV2,
)
from edgestack.recommendation.service import CanonicalBundleRepository
from edgestack.research.store import ResearchStore


def build_and_publish_canonical_baseline(
    cfg: EdgeStackConfig,
    *,
    run_date: date,
    now: datetime | None = None,
    funding_observation: FundingRateObservation | None = None,
    bootstrap_replications: int = 2_000,
    news_evidence: tuple[NewsEvidenceV2, ...] = (),
) -> PublicationRecordV2:
    """Build baseline plus zero-weight ideas and atomically publish one bundle."""
    generated_at = now or datetime.now(UTC)
    catalog = DataCatalog(cfg)
    calendar = TradingCalendar(cfg.data.calendar)
    policy = load_baseline_policy()
    research_store = ResearchStore(catalog)
    promotion_history = research_store.promoted_sleeves()
    promoted_registry = research_store.capital_eligible_sleeves()
    symbols = tuple(
        dict.fromkeys(
            (
                *(weight.symbol for weight in policy.weights),
                *(
                    weight.symbol
                    for sleeve in promoted_registry
                    for weight in sleeve.symbol_weights
                ),
            )
        )
    )
    expected_session = (
        run_date if calendar.is_session(run_date) else calendar.prev_session(run_date).date()
    )
    panel = catalog.load_panel(symbols=symbols, end=expected_session)
    available = set(panel["symbol"].astype(str))
    if available != set(symbols):
        raise DataError(f"baseline data missing symbols: {sorted(set(symbols) - available)}")
    quality = assess_panel(panel, calendar)
    if quality.quarantined:
        failures = "; ".join(
            f"{item.symbol}: {', '.join(item.issues)}" for item in quality.quarantined
        )
        raise DataError(f"baseline data quality failed: {failures}")
    latest_by_symbol = panel.groupby("symbol")["date"].max()
    complete = all(pd.Timestamp(value).date() == expected_session for value in latest_by_symbol)
    if not complete:
        raise DataError(
            "baseline data is incomplete for expected session "
            f"{expected_session}: {latest_by_symbol.to_dict()}"
        )

    prepared = panel.copy()
    factor = prepared["adj_close"] / prepared["close"]
    if factor.isna().any() or not np.isfinite(factor).all() or (factor <= 0).any():
        raise DataError("baseline adjusted-open factors are missing or invalid")
    prepared["adjusted_open"] = prepared["open"] * factor
    adjusted_opens = (
        prepared.pivot(index="date", columns="symbol", values="adjusted_open")
        .reindex(columns=symbols)
        .dropna()
    )
    returns = adjusted_opens.pct_change(fill_method=None).dropna()
    if len(returns) < 252:
        raise DataError("baseline publication requires 252 aligned return sessions")
    from edgestack.research.evaluate import (
        daily_position_for_next_open,
        event_weights_for_next_open,
    )

    adjusted_closes = (
        prepared.pivot(index="date", columns="symbol", values="adj_close")
        .reindex(columns=symbols)
        .sort_index()
    )
    active_sleeves = []
    for sleeve in promoted_registry:
        parameters = sleeve.signal_parameters
        if not parameters:
            active_sleeves.append(sleeve)
            continue
        if parameters.get("evaluator") == "events":
            raw_stocks = parameters.get("stock_symbols", [])
            stock_symbols = (
                tuple(str(item) for item in raw_stocks) if isinstance(raw_stocks, list) else ()
            )
            desired = event_weights_for_next_open(
                catalog,
                parameters,
                adjusted_closes,
                stock_symbols,
            )
            metadata_by_symbol = {weight.symbol: weight for weight in sleeve.symbol_weights}
            projected = tuple(
                metadata_by_symbol[symbol].model_copy(update={"weight": weight})
                for symbol, weight in sorted(desired.items())
                if symbol in metadata_by_symbol and weight > 0
            )
            if projected:
                active_sleeves.append(sleeve.model_copy(update={"symbol_weights": projected}))
            continue
        if daily_position_for_next_open(parameters, adjusted_closes) > 0:
            active_sleeves.append(sleeve)
    promoted_sleeves = tuple(active_sleeves)

    metadata: dict[str, AssetMetadata] = {}
    for symbol in symbols:
        history = prepared.loc[prepared["symbol"] == symbol].sort_values("date").tail(60)
        median_adv = float((history["close"] * history["volume"]).median())
        policy_weight = next(
            (weight for weight in policy.weights if weight.symbol == symbol),
            None,
        )
        promoted_weight = next(
            (
                weight
                for sleeve in promoted_registry
                for weight in sleeve.symbol_weights
                if weight.symbol == symbol
            ),
            None,
        )
        source_weight = policy_weight or promoted_weight
        if source_weight is None:
            raise DataError(f"no metadata source for canonical symbol {symbol}")
        metadata[symbol] = AssetMetadata(
            symbol=symbol,
            asset_kind=source_weight.asset_kind,
            sector=source_weight.sector,
            median_adv=median_adv,
            broad_policy_asset=policy_weight is not None,
        )

    as_of = datetime.combine(expected_session, time(20), tzinfo=UTC)
    next_session = calendar.next_session(expected_session)
    execution_at = calendar.market_open_at(next_session).to_pydatetime()
    data_version = catalog.data_manifest_hash()
    artifact_version = stable_hash(
        {
            "kind": "canonical_promoted_plus_baseline",
            "code_revision": code_revision(),
            "policy": policy.model_dump(mode="json"),
            "promoted_registry": [sleeve.model_dump(mode="json") for sleeve in promoted_registry],
            "active_promoted_sleeves": [sleeve.sleeve_id for sleeve in promoted_sleeves],
        }
    )
    freshness = FreshnessV2(
        as_of=as_of,
        expected_session=expected_session,
        is_fresh=True,
        age_business_days=0,
        complete=True,
        compatible=True,
    )
    watchlist = load_legacy_watchlist(catalog.artifacts_dir)
    expected_estimates: dict[str, ExpectedReturnEstimate] = {}
    for symbol in symbols:
        contributions = [
            (
                sleeve.expected_net_return * weight.weight,
                sleeve.expected_return_lower_95 * weight.weight,
                sleeve.effective_sample_size,
            )
            for sleeve in promoted_sleeves
            for weight in sleeve.symbol_weights
            if weight.symbol == symbol
        ]
        if not contributions:
            continue
        mean = sum(item[0] for item in contributions)
        lower = sum(item[1] for item in contributions)
        expected_estimates[symbol] = ExpectedReturnEstimate(
            annualized_mean=mean,
            annualized_standard_error=max(0.0, mean - lower) / 1.96,
            effective_sample_size=min(item[2] for item in contributions),
            shrinkage_prior_n=0.0,
        )
    base = build_base_recommendation(
        policy=policy,
        promoted_sleeves=promoted_sleeves,
        watchlist=watchlist,
        returns=returns,
        metadata=metadata,
        expected_estimates=expected_estimates,
        freshness=freshness,
        as_of=pd.Timestamp(as_of),
        execution_at=pd.Timestamp(execution_at),
        data_version=data_version,
        artifact_version=artifact_version,
    )
    if base.status is RecommendationStatus.NO_ALLOCATION:
        raise DataError("baseline assembly produced NO_ALLOCATION; prior pointer was retained")

    repository = CanonicalBundleRepository(catalog.artifacts_dir)
    if repository.pointer_path.exists():
        previous_bundle = repository.latest()
        previous_risk_inputs = repository.risk_inputs()
        paper = load_paper_state(
            repository,
            initial_equity=cfg.paper.initial_cash,
            risk_state=previous_bundle.default_recommendation.output_risk_state,
        )
        if paper.last_session is None or paper.last_session < expected_session:
            paper_symbols = {position.symbol for position in paper.positions}
            if paper.pending_target is not None:
                paper_symbols.update(
                    weight.symbol
                    for weight in paper.pending_target.weights
                    if weight.asset_kind is not AssetKind.CASH
                )
            bars: dict[str, Bar] = {}
            if paper_symbols:
                paper_panel = catalog.load_panel(
                    symbols=tuple(sorted(paper_symbols)), end=expected_session
                )
                paper_day = paper_panel.loc[
                    pd.to_datetime(paper_panel["date"]).dt.date == expected_session
                ]
                for raw_row in paper_day.itertuples(index=False):
                    row = cast(Any, raw_row)
                    bars[str(row.symbol)] = Bar(
                        session=pd.Timestamp(expected_session),
                        open=float(row.open),
                        high=float(row.high),
                        low=float(row.low),
                        close=float(row.close),
                        volume=float(row.volume),
                    )
            actions = catalog.load_corporate_actions(
                tuple(sorted(paper_symbols)),
                start=paper.last_session,
                end=expected_session,
            )
            paper = execute_paper_session(
                paper,
                session=expected_session,
                bars=bars,
                corporate_actions=corporate_action_inputs(actions),
                broker=get_broker(cfg),
                base_funding_rate=previous_risk_inputs.funding_rate,
                funding_spread_bps=previous_bundle.default_risk_profile.funding_spread_bps,
            )
    else:
        previous_bundle = None
        previous_risk_inputs = None
        initial_risk_state = RiskStateV2.initial(cfg.paper.initial_cash)
        paper = CanonicalPaperStateV2.initial(cfg.paper.initial_cash, initial_risk_state)

    profile = RiskProfileV2(account_equity=paper.current_equity)
    base_weight_map = {weight.symbol: weight.weight for weight in base.unlevered_base_weights}
    portfolio_returns = returns.loc[:, list(symbols)].to_numpy() @ np.array(
        [base_weight_map.get(symbol, 0.0) for symbol in symbols]
    )
    observation = funding_observation or fetch_dgs3mo(
        catalog.artifacts_dir / "funding" / "dgs3mo.json", now=generated_at
    )
    risk_inputs = estimate_risk_inputs(
        pd.Series(portfolio_returns, index=returns.index),
        session=expected_session,
        liquidity_position_limits={
            symbol: 0.50 * item.median_adv / profile.account_equity
            for symbol, item in metadata.items()
        },
        funding_rate=observation.annualized_rate,
        funding_rate_as_of=observation.as_of,
        n_boot=bootstrap_replications,
        seed=cfg.project.random_seed,
    )
    recommendation = size_recommendation(
        base=base,
        profile=profile,
        state=paper.risk_state,
        inputs=risk_inputs,
    )
    bundle = CanonicalRecommendationBundleV2(
        generated_at=generated_at,
        session=expected_session,
        as_of=as_of,
        execution_at=execution_at,
        data_version=data_version,
        artifact_version=artifact_version,
        policy_version=policy.policy_version,
        baseline_policy=policy,
        default_risk_profile=profile,
        base_recommendation=base,
        default_recommendation=recommendation,
    )
    paper = queue_recommendation_target(paper, recommendation)
    monitoring: dict[str, Any] = {
        "schema_version": 2,
        "healthy": True,
        "promoted_sleeves": len(promoted_registry),
        "historically_promoted_sleeves": len(promotion_history),
        "active_promoted_sleeves": len(promoted_sleeves),
        "watchlist_ideas": len(watchlist),
        "news_items": len(news_evidence),
        "claim": (
            "immutable promoted sleeves plus passive baseline"
            if promoted_registry
            else "baseline policy only; no promoted alpha claim"
        ),
    }
    portfolio_series = pd.Series(portfolio_returns, index=returns.index)
    spy_series = returns["SPY"]
    policy_weights = {weight.symbol: weight.weight for weight in policy.weights}
    baseline_series = returns.loc[
        :, [weight.symbol for weight in policy.weights]
    ].to_numpy() @ np.array([policy_weights[weight.symbol] for weight in policy.weights])
    portfolio_volatility = float(portfolio_series.std(ddof=1) * np.sqrt(252))
    spy_volatility = float(spy_series.std(ddof=1) * np.sqrt(252))
    risk_scale = portfolio_volatility / spy_volatility if spy_volatility > 0 else 0.0
    financed_spy = financed_risk_matched_spy(
        spy_series,
        target_volatility=portfolio_volatility,
        financing_rate=observation.annualized_rate + profile.funding_spread_bps / 10_000.0,
        cash_yield=observation.annualized_rate,
    )
    monitoring["calendar_log_growth"] = annualized_log_growth(portfolio_series)
    monitoring["comparator_log_growth"] = {
        "buy_now_spy": annualized_log_growth(spy_series),
        "risk_matched_spy": annualized_log_growth(spy_series * risk_scale),
        "diversified_baseline": annualized_log_growth(
            pd.Series(baseline_series, index=returns.index)
        ),
        "financed_spy_same_risk": annualized_log_growth(financed_spy),
    }
    publication = AtomicRecommendationPublisher(catalog.artifacts_dir).publish(
        bundle=bundle,
        risk_inputs=risk_inputs,
        paper_state=paper.model_dump(mode="json"),
        monitoring=monitoring,
        news_evidence=news_evidence,
    )
    RecommendationRegistry(catalog).save_publication(publication)
    catalog.audit(
        "canonical_publication_v2",
        reason=expected_session.isoformat(),
        run_id=publication.run_id,
        bundle_hash=bundle.bundle_hash,
    )
    return publication


def load_legacy_watchlist(artifacts_dir: Path) -> tuple[WatchlistEntryV2, ...]:
    """Import old board ideas as zero-weight, previously accessed research only."""
    path = artifacts_dir / "live_board.json"
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
    except (OSError, ValueError) as exc:
        raise DataError(f"invalid legacy board for watchlist migration: {exc}") from exc
    output: list[WatchlistEntryV2] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        output.append(
            WatchlistEntryV2(
                symbol=symbol,
                asset_kind=AssetKind.STOCK,
                horizon_sessions=10,
                family="legacy_incomplete_manual_search",
                thesis=(
                    "Previously accessed 2024-2026 research idea imported for observation only; "
                    "legacy validation and search history are not promotion-compatible."
                ),
                evidence_grade=EvidenceGrade.INSUFFICIENT,
                zero_weight_reason=(
                    "legacy artifact is ineligible; freeze V2 prospectively and accumulate "
                    "252 sessions plus ESS >= 100"
                ),
            )
        )
    return tuple(output)
