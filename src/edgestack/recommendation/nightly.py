"""Canonical baseline assembly used by the fail-fast nightly orchestrator."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.calendar import TradingCalendar
from edgestack.data.catalog import DataCatalog
from edgestack.data.quality import assess_panel
from edgestack.exceptions import DataError
from edgestack.recommendation.financing import FundingRateObservation, fetch_dgs3mo
from edgestack.recommendation.hashing import stable_hash
from edgestack.recommendation.manifests import PublicationRecordV2
from edgestack.recommendation.policy import load_baseline_policy
from edgestack.recommendation.portfolio import AssetMetadata, build_base_recommendation
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


def build_and_publish_canonical_baseline(
    cfg: EdgeStackConfig,
    *,
    run_date: date,
    now: datetime | None = None,
    funding_observation: FundingRateObservation | None = None,
    bootstrap_replications: int = 2_000,
) -> PublicationRecordV2:
    """Build baseline plus zero-weight ideas and atomically publish one bundle."""
    generated_at = now or datetime.now(UTC)
    catalog = DataCatalog(cfg)
    calendar = TradingCalendar(cfg.data.calendar)
    policy = load_baseline_policy()
    symbols = tuple(weight.symbol for weight in policy.weights)
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

    metadata: dict[str, AssetMetadata] = {}
    for symbol in symbols:
        history = prepared.loc[prepared["symbol"] == symbol].sort_values("date").tail(60)
        median_adv = float((history["close"] * history["volume"]).median())
        policy_weight = next(weight for weight in policy.weights if weight.symbol == symbol)
        metadata[symbol] = AssetMetadata(
            symbol=symbol,
            asset_kind=AssetKind.ETF,
            sector=policy_weight.sector,
            median_adv=median_adv,
            broad_policy_asset=True,
        )

    as_of = datetime.combine(expected_session, time(20), tzinfo=UTC)
    next_session = calendar.next_session(expected_session)
    execution_at = calendar.market_open_at(next_session).to_pydatetime()
    data_version = catalog.data_manifest_hash()
    artifact_version = stable_hash(
        {
            "kind": "baseline_only",
            "policy": policy.model_dump(mode="json"),
            "promoted_sleeves": [],
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
    base = build_base_recommendation(
        policy=policy,
        promoted_sleeves=(),
        watchlist=watchlist,
        returns=returns,
        metadata=metadata,
        expected_estimates={},
        freshness=freshness,
        as_of=pd.Timestamp(as_of),
        execution_at=pd.Timestamp(execution_at),
        data_version=data_version,
        artifact_version=artifact_version,
    )
    if base.status is RecommendationStatus.NO_ALLOCATION:
        raise DataError("baseline assembly produced NO_ALLOCATION; prior pointer was retained")

    profile = RiskProfileV2(account_equity=cfg.paper.initial_cash)
    portfolio_returns = returns.loc[:, list(symbols)].to_numpy() @ np.array(
        [weight.weight for weight in policy.weights]
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
    repository = CanonicalBundleRepository(catalog.artifacts_dir)
    if repository.pointer_path.exists():
        state = repository.latest().default_recommendation.output_risk_state
    else:
        state = RiskStateV2.initial(profile.account_equity)
    recommendation = size_recommendation(
        base=base,
        profile=profile,
        state=state,
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
    paper_state = {
        "schema_version": 2,
        "status": "INITIALIZED",
        "cash": profile.account_equity,
        "positions": [],
        "target_weights": [
            weight.model_dump(mode="json") for weight in recommendation.personalized_target_weights
        ],
        "risk_state": recommendation.output_risk_state.model_dump(mode="json"),
    }
    monitoring = {
        "schema_version": 2,
        "healthy": True,
        "promoted_sleeves": 0,
        "watchlist_ideas": len(watchlist),
        "claim": "baseline policy only; no promoted alpha claim",
    }
    publication = AtomicRecommendationPublisher(catalog.artifacts_dir).publish(
        bundle=bundle,
        risk_inputs=risk_inputs,
        paper_state=paper_state,
        monitoring=monitoring,
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
