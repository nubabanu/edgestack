"""Run the frozen public-data daily family without credential-only providers."""

from __future__ import annotations

import io
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog, atomic_write_bytes
from edgestack.exceptions import DataError
from edgestack.research.evaluate import evaluate_daily_campaign
from edgestack.research.schemas import CampaignLifecycle, CampaignSummaryV1
from edgestack.research.store import ResearchStore
from edgestack.research.strange_edges import (
    acquire_cash_yield,
    load_strange_edges_manifest,
)
from edgestack.research.templates import seed_registered_campaigns


def _materialize_current_cash_snapshot(
    cfg: EdgeStackConfig,
    *,
    through: date,
) -> Path:
    """Normalize the already-frozen public DGS3MO snapshot for cash carry.

    The graph endpoint is not an ALFRED vintage archive.  Its coverage record
    remains point-in-time false and campaigns using it cannot promote from the
    historical run.
    """
    manifest = load_strange_edges_manifest(Path("configs/strange_edges.yaml"))
    series, _metadata = acquire_cash_yield(
        cfg,
        manifest.government_contract_study,
        refresh=False,
    )
    series = series.loc[pd.DatetimeIndex(series.index).date <= through]
    if series.empty:
        raise DataError("the frozen DGS3MO snapshot does not cover the daily campaign")
    normalized = pd.DataFrame(
        {
            "series_id": "DGS3MO",
            "observation_date": series.index,
            "value": series.to_numpy(dtype=float) * 100.0,
            # A one-calendar-day delay is conservative for use as idle cash carry.
            "realtime_start": series.index + pd.Timedelta(days=1),
            "realtime_end": series.index + pd.Timedelta(days=1),
        }
    )
    buffer = io.BytesIO()
    normalized.to_parquet(buffer, index=False)
    path = Path(cfg.paths.data_dir) / "curated" / "macro" / "DGS3MO.parquet"
    atomic_write_bytes(path, buffer.getvalue())
    return path


def _artifact_payload(catalog: DataCatalog, campaign: CampaignSummaryV1) -> dict[str, Any]:
    if campaign.artifact_hash is None:
        return {}
    path = (
        catalog.artifacts_dir
        / "research"
        / "factory"
        / "runs"
        / campaign.artifact_hash
        / "result.json"
    )
    if not path.exists():
        raise DataError(f"daily campaign artifact is missing: {campaign.artifact_hash}")
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    return dict(wrapper.get("payload", {}))


def run_public_daily_campaign(
    cfg: EdgeStackConfig,
    *,
    through: date = date(2022, 12, 30),
) -> dict[str, Any]:
    """Register once, evaluate once, and return a compact deterministic summary."""
    if through >= cfg.validation.final_test_start:
        raise DataError(
            f"public daily study must end before final holdout {cfg.validation.final_test_start}"
        )
    _materialize_current_cash_snapshot(cfg, through=through)
    catalog = DataCatalog(cfg)
    store = ResearchStore(catalog)
    templates = seed_registered_campaigns(store, cfg, through=through)
    template = next(item for item in templates if item.evaluator == "daily")
    current = store.campaign(template.template_id)
    if current is None:
        raise DataError("daily public campaign was not registered")
    if current.lifecycle in {
        CampaignLifecycle.REJECTED,
        CampaignLifecycle.PAPER_SHADOW,
        CampaignLifecycle.PROMOTED,
    }:
        result = current
        if result.lifecycle is CampaignLifecycle.REJECTED and not result.failure_reasons:
            result = result.model_copy(
                update={
                    "updated_at": datetime.now(UTC),
                    "failure_reasons": (
                        "No candidate passed complete-family multiplicity, deflated Sharpe, "
                        "and lower-bound log-growth gates against every comparator.",
                    ),
                }
            )
            store.upsert_campaign(result)
    else:
        if current.lifecycle is CampaignLifecycle.NEEDS_DATA:
            current = current.model_copy(
                update={
                    "lifecycle": CampaignLifecycle.READY,
                    "updated_at": datetime.now(UTC),
                    "next_action": "Run the pre-registered public daily family exactly once.",
                }
            )
            store.upsert_campaign(current)
        if current.lifecycle is CampaignLifecycle.READY:
            current = current.model_copy(
                update={
                    "lifecycle": CampaignLifecycle.RUNNING,
                    "updated_at": datetime.now(UTC),
                }
            )
            store.upsert_campaign(current)
        result = evaluate_daily_campaign(store, template)
    payload = _artifact_payload(catalog, result)
    active_results = [
        item for item in payload.get("results", []) if int(item.get("invested_sessions", 0)) > 0
    ]
    ranked = sorted(
        active_results,
        key=lambda item: (
            float(item.get("q_value", 1.0)),
            -float(item.get("log_growth_improvement", -1.0)),
            str(item.get("candidate_id", "")),
        ),
    )
    return {
        "campaign": result.model_dump(mode="json"),
        "final_holdout_accessed": False,
        "cash_source_point_in_time": False,
        "registered_trials": len(template.candidate_family),
        "completed_trials": result.completed_trials,
        "winner_ids": list(payload.get("winner_ids", [])),
        "top_trials": ranked[:10],
        "limitations": [
            "Static surviving symbols; no point-in-time constituent history.",
            "Yahoo is an unofficial adjusted-price source.",
            "DGS3MO is a current FRED graph snapshot, not an ALFRED vintage archive.",
            "Historical winners can only begin independent paper shadow.",
        ],
    }
