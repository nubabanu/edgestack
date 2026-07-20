"""Synchronize the forward opening-study archive into Edge Lab state."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog
from edgestack.exceptions import DataError
from edgestack.recommendation.hashing import stable_hash
from edgestack.research.coverage import scan_catalog_coverage
from edgestack.research.opening_fade import load_campaign_config
from edgestack.research.schemas import (
    CampaignLifecycle,
    CampaignSummaryV1,
    CoverageState,
    EvidenceGapV1,
)
from edgestack.research.store import ResearchStore


def _published_opening_result(root: Path) -> tuple[dict[str, Any], Path]:
    pointer_path = root / "current.json"
    if not pointer_path.exists():
        raise DataError("opening-fade current pointer is missing")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    relative = pointer.get("path")
    if not isinstance(relative, str):
        raise DataError("opening-fade pointer has no run path")
    run_dir = root / relative
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    occurrence = json.loads((run_dir / "occurrence.json").read_text(encoding="utf-8"))
    return {
        "pointer": pointer,
        "metrics": metrics,
        "occurrence": occurrence,
    }, run_dir


def _common_session_count(catalog: DataCatalog, symbols: tuple[str, ...]) -> int:
    session_sets: list[set[date]] = []
    for symbol in symbols:
        path = catalog.intraday_dir / "1m" / f"{symbol}.parquet"
        if not path.exists():
            return 0
        timestamps = pd.to_datetime(pd.read_parquet(path, columns=["timestamp"])["timestamp"])
        session_sets.append(set(timestamps.dt.date))
    return len(set.intersection(*session_sets)) if session_sets else 0


def sync_opening_fade_state(
    cfg: EdgeStackConfig,
    *,
    opening_config_path: Path = Path("configs/opening_fade.yaml"),
) -> CampaignSummaryV1:
    """Expose exact forward-archive progress without pretending it is evidence."""
    opening = load_campaign_config(opening_config_path)
    catalog = DataCatalog(cfg)
    store = ResearchStore(catalog)
    for coverage in scan_catalog_coverage(catalog):
        store.upsert_coverage(coverage)
    published, run_dir = _published_opening_result(opening.artifacts_dir)
    validation = dict(published["metrics"].get("validation", {}))
    occurrence = dict(published["occurrence"])
    pointer = dict(published["pointer"])
    symbols = opening.primary_symbols + opening.secondary_symbols
    one_minute = [
        item
        for item in store.coverage()
        if item.dataset == "intraday" and item.frequency == "1m" and item.symbol in symbols
    ]
    by_symbol = {item.symbol: item for item in one_minute}
    observed_bars = min(
        (by_symbol[symbol].observation_count for symbol in symbols if symbol in by_symbol),
        default=0,
    )
    starts = [
        item.start.date()
        for item in one_minute
        if item.start is not None and item.symbol in symbols
    ]
    common_start = max(starts) if len(by_symbol) == len(symbols) and starts else date.today()
    target_sessions = 252
    # A planning horizon, not a trading-calendar claim.  The acceptance gate
    # is the exact count of archived complete sessions below.
    target_end = common_start + timedelta(days=365)
    required_bars = target_sessions * 390
    common_sessions = _common_session_count(catalog, symbols)
    campaign_id = opening.campaign_version
    now = datetime.now(UTC)
    current = store.campaign(campaign_id)
    failure = (
        f"Only {common_sessions} common 1-minute sessions are archived; at least 60 are "
        "required for family inference and 252 for the frozen prospective gate."
    )
    campaign = CampaignSummaryV1(
        campaign_id=campaign_id,
        manifest_hash=opening.stable_hash(),
        name="Prospective opening sequence and VWAP family",
        family="intraday_opening_multi_family",
        lifecycle=CampaignLifecycle.NEEDS_DATA,
        created_at=current.created_at if current is not None else now,
        updated_at=now,
        trial_count=len(opening.candidates),
        completed_trials=0,
        data_requirements=(stable_hash({"campaign": campaign_id, "frequency": "1m"})[:24],),
        failure_reasons=(failure,),
        next_action=(
            "Continue the unchanged nightly 1m/5m forward collector; rerun after at least "
            "252 complete sessions."
        ),
        artifact_hash=str(pointer["run_id"]),
        promotion_eligible=False,
        previously_accessed=True,
        metrics={
            "attempted_trials": int(validation.get("trial_count", len(opening.candidates))),
            "eligible_symbol_sessions": int(occurrence.get("total_eligible_sessions", 0)),
            "common_archive_sessions": common_sessions,
            "minimum_inference_sessions": 60,
            "prospective_target_sessions": target_sessions,
            "paper_observation_candidates": len(validation.get("paper_observation_candidates", [])),
            "classification": str(pointer.get("classification", "INSUFFICIENT")),
            "artifact_path": str(run_dir),
        },
    )
    store.upsert_campaign(campaign)
    requirement_id = campaign.data_requirements[0]
    store.upsert_gap(
        EvidenceGapV1(
            requirement_id=requirement_id,
            campaign_id=campaign_id,
            dataset="intraday",
            symbols=symbols,
            frequency="1m",
            start=common_start,
            end=target_end,
            required_observations=required_bars,
            observed_observations=observed_bars,
            state=CoverageState.NEEDS_DATA,
            provider="yahoo_forward_archive",
            next_action=(
                f"Forward archive has {common_sessions}/{target_sessions} common sessions; "
                "free historical backfill is unavailable, so keep collecting nightly."
            ),
        )
    )
    return campaign
