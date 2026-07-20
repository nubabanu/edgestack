"""Resumable bounded worker for acquisition, evaluation, and shadow updates."""

from __future__ import annotations

import ctypes
import hashlib
import io
import os
import socket
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.calendar import TradingCalendar
from edgestack.data.catalog import DataCatalog, atomic_write_bytes, exclusive_path_lock
from edgestack.data.intraday import resample_intraday_bars
from edgestack.data.provider_credentials import ResearchProviderCredentials
from edgestack.data.providers.base import IntradayDataProvider
from edgestack.data.providers.edgar import SecEdgarBulkProvider
from edgestack.data.providers.finra import FinraShortSaleVolumeProvider
from edgestack.data.providers.fred import FredVintageProvider
from edgestack.data.providers.registry import get_price_provider
from edgestack.data.universe_pit import PitSP500Universe
from edgestack.exceptions import ProviderError
from edgestack.recommendation.hashing import stable_hash
from edgestack.research.coverage import (
    CoveragePlanner,
    DataRequirement,
    default_free_capabilities,
)
from edgestack.research.evaluate import (
    evaluate_daily_campaign,
    evaluate_event_campaign,
    evaluate_opening_campaign,
    monitor_promoted_shadow,
    review_daily_shadow_promotion,
    review_event_shadow_promotion,
    update_daily_shadow,
    update_event_shadow,
    update_opening_shadow,
)
from edgestack.research.locks import nightly_is_active
from edgestack.research.schemas import (
    AcquisitionJobV1,
    CampaignLifecycle,
    DataCoverageV1,
    JobState,
    WorkerHealthV1,
    WorkerState,
)
from edgestack.research.store import ResearchStore
from edgestack.research.templates import (
    CampaignTemplate,
    registered_templates,
    seed_registered_campaigns,
)
from edgestack.research.universe import (
    ALL_RESEARCH_ETFS,
    build_monthly_liquid_universe,
)


def _symbols(payload: dict[str, Any]) -> tuple[str, ...]:
    raw = payload.get("symbols")
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("research job symbols must be a list of strings")
    return tuple(raw)


def _storage_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return total / (1024**3)


def _factory_storage_gb(cfg: EdgeStackConfig) -> float:
    data = Path(cfg.paths.data_dir).resolve()
    artifacts = Path(cfg.paths.artifacts_dir).resolve()
    total = _storage_gb(data)
    if artifacts != data and not artifacts.is_relative_to(data):
        total += _storage_gb(artifacts)
    return total


def _write_grouped(frame: pd.DataFrame, directory: Path, *, group: str) -> None:
    if frame.empty:
        return
    normalized = frame.drop(columns=["retrieved_at"], errors="ignore")
    natural_keys = {
        "series_id": ["series_id", "observation_date", "realtime_start", "realtime_end"],
        "symbol": (
            ["symbol", "accepted_at", "form"] if "accepted_at" in normalized else ["symbol", "date"]
        ),
    }
    keys = [item for item in natural_keys.get(group, [group]) if item in normalized]
    for name, values in normalized.groupby(group, observed=True, sort=True):
        path = directory / f"{str(name).upper()}.parquet"
        with exclusive_path_lock(path):
            merged = values
            if path.exists():
                existing = pd.read_parquet(path).drop(columns=["retrieved_at"], errors="ignore")
                merged = pd.concat([existing, values], ignore_index=True)
            merged = merged.drop_duplicates(subset=keys, keep="last").sort_values(keys)

            buffer = io.BytesIO()
            merged.to_parquet(buffer, index=False)
            atomic_write_bytes(path, buffer.getvalue())


def _record_acquired_coverage(
    store: ResearchStore,
    frame: pd.DataFrame,
    *,
    dataset: str,
    provider: str,
    feed: str,
    frequency: str,
    group_column: str,
    timestamp_column: str,
    requested_names: tuple[str, ...],
    requested_start: date,
    requested_end: date,
    directory: Path,
    point_in_time: bool,
    limitations: tuple[str, ...],
    raw_content_hashes: tuple[str, ...],
    expect_market_sessions: bool,
) -> None:
    """Persist request-level provenance alongside the merged normalized files."""
    retrieved_at = datetime.now(UTC)
    for name in requested_names:
        if group_column in frame:
            selected = frame.loc[frame[group_column].astype(str).str.upper() == name.upper()]
        else:
            selected = frame.iloc[0:0]
        timestamps = (
            pd.to_datetime(selected[timestamp_column], utc=True, errors="coerce").dropna()
            if timestamp_column in selected
            else pd.Series(dtype="datetime64[ns, UTC]")
        )
        unique = timestamps.drop_duplicates()
        duplicate_count = len(timestamps) - len(unique)
        observed_sessions = {item.date() for item in unique}
        missing_sessions = 0
        if expect_market_sessions:
            expected = TradingCalendar().sessions(requested_start, requested_end)
            missing_sessions = len({item.date() for item in expected} - observed_sessions)
        normalized_path = directory / f"{name.upper()}.parquet"
        content_hash = (
            hashlib.sha256(normalized_path.read_bytes()).hexdigest()
            if normalized_path.exists()
            else stable_hash(
                {
                    "dataset": dataset,
                    "provider": provider,
                    "name": name,
                    "start": requested_start,
                    "end": requested_end,
                    "empty": True,
                }
            )
        )
        quality_status = (
            "EMPTY"
            if len(unique) == 0
            else "GAPS"
            if missing_sessions or duplicate_count
            else "PASS"
        )
        request_key = stable_hash(
            {
                "dataset": dataset,
                "provider": provider,
                "feed": feed,
                "frequency": frequency,
                "name": name,
                "start": requested_start,
                "end": requested_end,
            }
        )[:16]
        store.upsert_coverage(
            DataCoverageV1(
                dataset_id=f"{dataset}:{provider}:{frequency}:{name.upper()}:{request_key}",
                dataset=dataset,
                provider=provider,
                feed=feed,
                symbol=name.upper(),
                frequency=frequency,
                start=datetime.combine(requested_start, datetime.min.time(), tzinfo=UTC),
                end=datetime.combine(requested_end, datetime.max.time(), tzinfo=UTC),
                observation_count=len(unique),
                missing_sessions=missing_sessions,
                quality_status=quality_status,
                point_in_time=point_in_time,
                content_hash=content_hash,
                raw_content_hashes=raw_content_hashes,
                retrieved_at=retrieved_at,
                quality_results={
                    "returned_observations": len(unique),
                    "duplicate_observations": duplicate_count,
                    "missing_sessions": missing_sessions,
                },
                limitations=limitations,
            )
        )


def _record_requested_event_coverage(
    store: ResearchStore,
    frame: pd.DataFrame,
    *,
    provider: str,
    symbols: tuple[str, ...],
    start: date,
    end: date,
    limitations: tuple[str, ...],
    raw_content_hashes: tuple[str, ...],
) -> None:
    payload_hash = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    ).hexdigest()
    retrieved_at = datetime.now(UTC)
    for symbol in symbols:
        count = int((frame.get("symbol", pd.Series(dtype=str)) == symbol).sum())
        store.upsert_coverage(
            DataCoverageV1(
                dataset_id=f"earnings:{provider}:event:{symbol}",
                dataset="earnings",
                provider=provider,
                feed="SEC_EDGAR_8K_ITEM_2_02",
                symbol=symbol,
                frequency="event",
                start=datetime.combine(start, datetime.min.time(), tzinfo=UTC),
                end=datetime.combine(end, datetime.max.time(), tzinfo=UTC),
                observation_count=count,
                quality_status="PASS" if count else "EMPTY",
                point_in_time=True,
                content_hash=stable_hash(
                    {"payload_hash": payload_hash, "symbol": symbol, "start": start, "end": end}
                ),
                raw_content_hashes=raw_content_hashes,
                retrieved_at=retrieved_at,
                quality_results={"returned_events": count},
                limitations=limitations,
            )
        )


def set_below_normal_priority() -> None:
    """Best effort only; failure must not stop research correctness."""
    if os.name == "nt":
        try:
            handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
            ctypes.windll.kernel32.SetPriorityClass(handle, 0x00004000)  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            return
    else:
        try:
            nice = getattr(os, "nice", None)
            if nice is not None:
                nice(5)
        except OSError:
            return


def _execute_job_process(cfg_json: str, repo_root: str, job_json: str) -> str:
    """Spawn-safe entry point for one leased job in an isolated worker process."""
    set_below_normal_priority()
    worker = ResearchWorker(
        EdgeStackConfig.model_validate_json(cfg_json),
        repo_root=Path(repo_root),
    )
    job = AcquisitionJobV1.model_validate_json(job_json)
    return worker._execute_safely(job).model_dump_json()


class ResearchWorker:
    def __init__(self, cfg: EdgeStackConfig, *, repo_root: Path | None = None) -> None:
        self.cfg = cfg
        self.catalog = DataCatalog(cfg)
        self.store = ResearchStore(self.catalog)
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.owner = f"{socket.gethostname()}:{os.getpid()}"
        self.templates: dict[str, CampaignTemplate] = {}

    def _health(
        self,
        state: WorkerState,
        *,
        current_job_id: str | None = None,
        paused: bool = False,
        last_error: str | None = None,
    ) -> WorkerHealthV1:
        return WorkerHealthV1(
            state=state,
            paused=paused,
            current_job_id=current_job_id,
            process_priority=self.cfg.research.process_priority,
            max_workers=self.cfg.research.max_workers,
            storage_used_gb=_factory_storage_gb(self.cfg),
            storage_cap_gb=self.cfg.research.storage_cap_gb,
            heartbeat_at=datetime.now(UTC),
            last_error=last_error,
        )

    def pause(self) -> WorkerHealthV1:
        health = self._health(WorkerState.PAUSED, paused=True)
        self.store.set_worker(health)
        return health

    def resume(self) -> WorkerHealthV1:
        health = self._health(WorkerState.IDLE)
        self.store.set_worker(health)
        return health

    def status(self) -> WorkerHealthV1:
        return self.store.worker() or self._health(WorkerState.IDLE)

    def prepare(self, *, through: date | None = None) -> None:
        provisional_templates = registered_templates(self.cfg, through=through)
        credentials = ResearchProviderCredentials()
        planner = CoveragePlanner(
            self.store,
            default_free_capabilities(
                alpaca_configured=bool(credentials.alpaca_key_id and credentials.alpaca_secret_key),
                fred_configured=bool(credentials.fred_key),
                sec_configured=bool(credentials.sec_agent),
            ),
        )
        planner.refresh_catalog()
        cutoff = max(
            requirement.end for item in provisional_templates for requirement in item.requirements
        )
        if not self._prepare_monthly_universe(planner, cutoff=cutoff):
            self.templates = {}
            return
        templates = seed_registered_campaigns(self.store, self.cfg, through=through)
        self.templates = {item.template_id: item for item in templates}
        existing_jobs = {item.job_id: item for item in self.store.jobs()}
        for template in templates:
            gaps = [planner.plan(requirement) for requirement in template.requirements]
            current = self.store.campaign(template.template_id)
            if current is None or current.lifecycle not in {
                CampaignLifecycle.NEEDS_DATA,
                CampaignLifecycle.READY,
                CampaignLifecycle.BLOCKED_FREE_TIER,
            }:
                continue
            if all(gap.state.value == "READY" for gap in gaps):
                lifecycle = CampaignLifecycle.READY
                next_action = "Run the pre-registered campaign exactly once."
                evaluation_id = f"evaluate:{template.template_id}:{template.manifest_hash[:12]}"
                if evaluation_id not in existing_jobs:
                    now = datetime.now(UTC)
                    self.store.upsert_job(
                        AcquisitionJobV1(
                            job_id=evaluation_id,
                            requirement_id="evaluation",
                            campaign_id=template.template_id,
                            kind="EVALUATE_CAMPAIGN",
                            priority=60,
                            created_at=now,
                            updated_at=now,
                            payload={"evaluator": template.evaluator},
                        )
                    )
            elif any(gap.state.value == "NEEDS_DATA" for gap in gaps):
                lifecycle = CampaignLifecycle.NEEDS_DATA
                next_action = "Complete queued evidence acquisition jobs."
            else:
                lifecycle = CampaignLifecycle.BLOCKED_FREE_TIER
                next_action = "Park this campaign and continue other registered families."
            self.store.upsert_campaign(
                current.model_copy(
                    update={
                        "lifecycle": lifecycle,
                        "updated_at": datetime.now(UTC),
                        "next_action": next_action,
                        "failure_reasons": tuple(
                            gap.next_action
                            for gap in gaps
                            if gap.state.value == "BLOCKED_FREE_TIER"
                        ),
                    }
                )
            )
        shadow_through = max(item.requirements[0].end for item in templates)
        for shadow in self.store.shadows():
            campaign = self.store.campaign(shadow.campaign_id)
            if (
                campaign is None
                or campaign.family
                not in {
                    "daily_multi_family",
                    "event_context_multi_family",
                    "intraday_opening_multi_family",
                }
                or shadow.status in {CampaignLifecycle.REJECTED, CampaignLifecycle.RETIRED}
            ):
                continue
            if shadow.resolved_through is not None and shadow.resolved_through >= shadow_through:
                continue
            shadow_key = stable_hash({"strategy": shadow.strategy_id, "through": shadow_through})[
                :24
            ]
            job_id = f"shadow:{shadow_key}"
            if job_id not in existing_jobs:
                now = datetime.now(UTC)
                self.store.upsert_job(
                    AcquisitionJobV1(
                        job_id=job_id,
                        requirement_id="prospective-shadow-update",
                        campaign_id=shadow.campaign_id,
                        kind="UPDATE_SHADOW",
                        priority=10,
                        created_at=now,
                        updated_at=now,
                        payload={"strategy_id": shadow.strategy_id},
                    )
                )

    def _prepare_monthly_universe(
        self,
        planner: CoveragePlanner,
        *,
        cutoff: date,
    ) -> bool:
        """Acquire, rank, and persist the PIT monthly stock/ETF storage tiers."""
        try:
            pit = PitSP500Universe(
                Path(self.cfg.paths.data_dir) / "cache" / "universe",
                earliest=date(2000, 1, 1),
            )
        except Exception as exc:
            self.catalog.audit("research_universe_unavailable", reason=str(exc)[:300])
            return False
        members = pit.members(cutoff)
        daily_start = cutoff - timedelta(days=140)
        daily_gap = planner.plan(
            DataRequirement(
                campaign_id=f"system-monthly-universe-{cutoff:%Y%m%d}",
                dataset="prices",
                symbols=members,
                frequency="1d",
                start=daily_start,
                end=cutoff,
                required_observations=40,
                preferred_providers=("alpaca", "yahoo"),
                priority=0,
            )
        )
        if daily_gap.state.value != "READY":
            return False
        frames = []
        for symbol in members:
            path = self.catalog.prices_dir / f"{symbol}.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path, columns=["symbol", "date", "close", "volume"])
            frame["date"] = pd.to_datetime(frame["date"])
            frames.append(frame.loc[frame["date"].dt.date >= daily_start])
        if not frames:
            return False
        universe = build_monthly_liquid_universe(
            pd.concat(frames, ignore_index=True),
            pit,
            stock_count=self.cfg.research.liquid_stock_count,
            one_minute_stock_count=self.cfg.research.one_minute_stock_count,
            min_price=self.cfg.universe.min_price,
            min_median_dollar_volume=self.cfg.universe.min_median_dollar_volume,
        )
        latest_as_of = max(item.as_of for item in universe)
        latest = [item for item in universe if item.as_of == latest_as_of]
        output = pd.DataFrame([asdict(item) for item in latest]).sort_values(
            ["asset_kind", "rank", "symbol"]
        )
        buffer = io.BytesIO()
        output.to_parquet(buffer, index=False)
        content = buffer.getvalue()
        content_hash = hashlib.sha256(content).hexdigest()
        directory = Path(self.cfg.paths.data_dir) / "curated" / "research_universe"
        path = directory / f"{cutoff:%Y%m%d}-{content_hash}.parquet"
        atomic_write_bytes(path, content)
        atomic_write_bytes(
            directory / "current.json",
            (
                '{"as_of":"'
                + cutoff.isoformat()
                + '","content_hash":"'
                + content_hash
                + '","file":"'
                + path.name
                + '"}'
            ).encode(),
        )
        self.store.upsert_coverage(
            DataCoverageV1(
                dataset_id=f"research_universe:pit:{cutoff:%Y%m%d}",
                dataset="research_universe",
                provider="wikipedia_sp500_change_log+catalog_prices",
                feed="monthly_ranked",
                symbol="S&P500+ETF_TIERS",
                frequency="monthly",
                start=datetime.combine(cutoff, datetime.min.time(), tzinfo=UTC),
                end=datetime.combine(cutoff, datetime.max.time(), tzinfo=UTC),
                observation_count=len(latest),
                quality_status="PASS",
                point_in_time=True,
                content_hash=content_hash,
                retrieved_at=datetime.now(UTC),
                limitations=pit.snapshot(cutoff).limitations,
            )
        )
        one_minute_stocks = tuple(
            item.symbol
            for item in latest
            if item.asset_kind == "STOCK" and item.intraday_interval_minutes == 1
        )
        five_minute_stocks = tuple(
            item.symbol
            for item in latest
            if item.asset_kind == "STOCK" and item.intraday_interval_minutes == 5
        )
        intraday_start = date(2016, 1, 4)
        for frequency, symbols, observations in (
            ("1m", (*ALL_RESEARCH_ETFS, *one_minute_stocks), 252 * 390),
            ("5m", five_minute_stocks, 252 * 78),
        ):
            if not symbols:
                continue
            planner.plan(
                DataRequirement(
                    campaign_id=f"system-monthly-universe-{cutoff:%Y%m%d}",
                    dataset="intraday",
                    symbols=tuple(symbols),
                    frequency=frequency,
                    start=intraday_start,
                    end=cutoff,
                    required_observations=observations,
                    preferred_providers=("alpaca", "yahoo"),
                    priority=0,
                )
            )
        return True

    def run_cycle(self) -> list[AcquisitionJobV1]:
        prior = self.status()
        if prior.paused:
            self.store.set_worker(self._health(WorkerState.PAUSED, paused=True))
            return []
        if self.cfg.research.pause_during_nightly and nightly_is_active(
            Path(self.cfg.paths.artifacts_dir)
        ):
            self.store.set_worker(self._health(WorkerState.NIGHTLY_PAUSE))
            return []
        if _factory_storage_gb(self.cfg) >= self.cfg.research.storage_cap_gb:
            self.store.set_worker(self._health(WorkerState.STORAGE_BLOCKED))
            return []
        self.prepare()
        leased = []
        for _ in range(self.cfg.research.max_workers):
            job = self.store.lease_next_job(self.owner, lease_minutes=60)
            if job is None:
                break
            leased.append(job)
        if not leased:
            self.store.set_worker(self._health(WorkerState.IDLE))
            return []
        self.store.set_worker(self._health(WorkerState.RUNNING, current_job_id=leased[0].job_id))
        cfg_json = self.cfg.model_dump_json()
        with ProcessPoolExecutor(max_workers=self.cfg.research.max_workers) as pool:
            future_jobs = {
                pool.submit(
                    _execute_job_process,
                    cfg_json,
                    str(self.repo_root),
                    job.model_dump_json(),
                ): job
                for job in leased
            }
            pending = set(future_jobs)
            by_job_id: dict[str, AcquisitionJobV1] = {}
            while pending:
                done, pending = wait(pending, timeout=30.0, return_when=FIRST_COMPLETED)
                for future in done:
                    completed_job = AcquisitionJobV1.model_validate_json(future.result())
                    by_job_id[completed_job.job_id] = completed_job
                for future in pending:
                    job = future_jobs[future]
                    self.store.renew_lease(job.job_id, self.owner, lease_minutes=60)
            completed = [by_job_id[job.job_id] for job in leased]
        self.store.set_worker(self._health(WorkerState.IDLE))
        return completed

    def run_forever(self, *, poll_seconds: float = 5.0) -> None:
        set_below_normal_priority()
        while True:
            self.run_cycle()
            time.sleep(max(1.0, poll_seconds))

    def _execute_safely(self, job: AcquisitionJobV1) -> AcquisitionJobV1:
        try:
            complete, payload = self._execute(job)
        except Exception as exc:
            attempts = job.attempts + 1
            payload = dict(job.payload)
            fallback_raw = payload.get("fallback_providers")
            fallbacks = list(fallback_raw) if isinstance(fallback_raw, list) else []
            if isinstance(exc, ProviderError) and fallbacks:
                payload["provider"] = fallbacks.pop(0)
                payload["fallback_providers"] = fallbacks
                updated = job.model_copy(
                    update={
                        "state": JobState.RETRY,
                        "attempts": attempts,
                        "payload": payload,
                        "not_before": datetime.now(UTC) + timedelta(minutes=1),
                        "lease_owner": None,
                        "lease_expires_at": None,
                        "updated_at": datetime.now(UTC),
                        "last_error": "primary provider failed; switched to free fallback",
                    }
                )
                self.store.upsert_job(updated)
                return updated
            blocked = isinstance(exc, ProviderError) and "unconfigured" in str(exc).lower()
            state = (
                JobState.BLOCKED_FREE_TIER
                if blocked
                else JobState.FAILED
                if attempts >= 5
                else JobState.RETRY
            )
            updated = job.model_copy(
                update={
                    "state": state,
                    "attempts": attempts,
                    "not_before": (
                        datetime.now(UTC) + timedelta(minutes=min(60, 2**attempts))
                        if state is JobState.RETRY
                        else None
                    ),
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "updated_at": datetime.now(UTC),
                    "last_error": str(exc)[:500],
                }
            )
        else:
            updated = job.model_copy(
                update={
                    "state": JobState.SUCCEEDED if complete else JobState.RETRY,
                    "payload": payload,
                    "not_before": None if complete else datetime.now(UTC),
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "updated_at": datetime.now(UTC),
                    "last_error": None,
                }
            )
        if job.kind == "EVALUATE_CAMPAIGN" and updated.state in {
            JobState.FAILED,
            JobState.BLOCKED_FREE_TIER,
        }:
            campaign = self.store.campaign(job.campaign_id)
            if campaign is not None and campaign.lifecycle is CampaignLifecycle.RUNNING:
                self.store.upsert_campaign(
                    campaign.model_copy(
                        update={
                            "lifecycle": CampaignLifecycle.REJECTED,
                            "updated_at": datetime.now(UTC),
                            "failure_reasons": (
                                updated.last_error or "campaign evaluation failed",
                            ),
                            "next_action": (
                                "Repair the evaluator or register a new immutable campaign; "
                                "this failed run cannot promote."
                            ),
                            "promotion_eligible": False,
                        }
                    )
                )
        self.store.upsert_job(updated)
        return updated

    def _execute(self, job: AcquisitionJobV1) -> tuple[bool, dict[str, Any]]:
        if job.kind == "ACQUIRE_DATA":
            return self._acquire(job)
        if job.kind == "EVALUATE_CAMPAIGN":
            template = self.templates.get(job.campaign_id)
            if template is None:
                cohort = next(
                    (
                        part
                        for part in job.campaign_id.split("-")
                        if len(part) == 8 and part.isdigit()
                    ),
                    None,
                )
                if cohort is not None:
                    cutoff = datetime.strptime(cohort, "%Y%m%d").date()
                    template = next(
                        (
                            item
                            for item in registered_templates(self.cfg, through=cutoff)
                            if item.template_id == job.campaign_id
                        ),
                        None,
                    )
            if template is None:
                raise ValueError(f"cannot reconstruct frozen campaign {job.campaign_id}")
            current = self.store.campaign(job.campaign_id)
            if current is None:
                raise ValueError(f"unknown campaign {job.campaign_id}")
            self.store.upsert_campaign(
                current.model_copy(
                    update={"lifecycle": CampaignLifecycle.RUNNING, "updated_at": datetime.now(UTC)}
                )
            )
            if template.evaluator == "daily":
                evaluate_daily_campaign(self.store, template)
            elif template.evaluator == "opening_fade":
                evaluate_opening_campaign(self.store, template, repo_root=self.repo_root)
            elif template.evaluator == "events":
                evaluate_event_campaign(self.store, template)
            else:
                raise ProviderError(f"campaign evaluator {template.evaluator!r} is unavailable")
            return True, dict(job.payload)
        if job.kind == "UPDATE_SHADOW":
            template = self.templates.get(job.campaign_id)
            if template is None:
                cohort = next(
                    (
                        part
                        for part in job.campaign_id.split("-")
                        if len(part) == 8 and part.isdigit()
                    ),
                    None,
                )
                if cohort is not None:
                    cutoff = datetime.strptime(cohort, "%Y%m%d").date()
                    template = next(
                        (
                            item
                            for item in registered_templates(self.cfg, through=cutoff)
                            if item.template_id == job.campaign_id
                        ),
                        None,
                    )
            if template is None:
                raise ValueError(f"cannot reconstruct frozen campaign {job.campaign_id}")
            strategy_id = str(job.payload["strategy_id"])
            if template.evaluator == "daily":
                updated = update_daily_shadow(self.store, strategy_id)
                updated = review_daily_shadow_promotion(self.store, template, updated)
            elif template.evaluator == "events":
                updated = update_event_shadow(self.store, template, strategy_id)
                updated = review_event_shadow_promotion(self.store, template, updated)
            elif template.evaluator == "opening_fade" and template.previously_accessed:
                updated = update_opening_shadow(
                    self.store,
                    template,
                    strategy_id,
                    repo_root=self.repo_root,
                )
            else:
                raise ProviderError(
                    f"prospective updater {template.evaluator!r} is not capital eligible"
                )
            updated = monitor_promoted_shadow(self.store, updated)
            self.store.upsert_shadow(updated)
            return True, dict(job.payload)
        raise ValueError(f"unsupported research job kind {job.kind}")

    def _acquire(self, job: AcquisitionJobV1) -> tuple[bool, dict[str, Any]]:
        payload = dict(job.payload)
        dataset = str(payload["dataset"])
        if dataset in {"prices", "intraday"}:
            return self._acquire_market(payload)
        if dataset == "macro":
            credentials = ResearchProviderCredentials()
            fred_provider = FredVintageProvider(
                credentials.fred_key,
                raw_dir=Path(self.cfg.paths.data_dir) / "raw" / "fred",
                timeout=self.cfg.data.request_timeout_seconds,
            )
            frame = fred_provider.fetch_series(
                _symbols(payload),
                date.fromisoformat(str(payload["start"])),
                date.fromisoformat(str(payload["end"])),
            )
            _write_grouped(
                frame,
                Path(self.cfg.paths.data_dir) / "curated" / "macro",
                group="series_id",
            )
            _record_acquired_coverage(
                self.store,
                frame,
                dataset="macro",
                provider=fred_provider.metadata.name,
                feed="ALFRED_ALL_VINTAGES",
                frequency=str(payload["frequency"]),
                group_column="series_id",
                timestamp_column="observation_date",
                requested_names=_symbols(payload),
                requested_start=date.fromisoformat(str(payload["start"])),
                requested_end=date.fromisoformat(str(payload["end"])),
                directory=Path(self.cfg.paths.data_dir) / "curated" / "macro",
                point_in_time=fred_provider.metadata.is_point_in_time,
                limitations=fred_provider.metadata.limitations,
                raw_content_hashes=fred_provider.last_raw_hashes,
                expect_market_sessions=False,
            )
            return True, payload
        if dataset == "short_sale_volume":
            finra_provider = FinraShortSaleVolumeProvider(
                raw_dir=Path(self.cfg.paths.data_dir) / "raw" / "finra",
                timeout=self.cfg.data.request_timeout_seconds,
            )
            frame = finra_provider.fetch_daily_short_sale_volume(
                _symbols(payload),
                date.fromisoformat(str(payload["start"])),
                date.fromisoformat(str(payload["end"])),
            )
            _write_grouped(
                frame,
                Path(self.cfg.paths.data_dir) / "curated" / "short_sale_volume",
                group="symbol",
            )
            _record_acquired_coverage(
                self.store,
                frame,
                dataset="short_sale_volume",
                provider=finra_provider.metadata.name,
                feed="FINRA_OTC_REG_SHO_DAILY",
                frequency="1d",
                group_column="symbol",
                timestamp_column="date",
                requested_names=_symbols(payload),
                requested_start=date.fromisoformat(str(payload["start"])),
                requested_end=date.fromisoformat(str(payload["end"])),
                directory=Path(self.cfg.paths.data_dir) / "curated" / "short_sale_volume",
                point_in_time=finra_provider.metadata.is_point_in_time,
                limitations=finra_provider.metadata.limitations,
                raw_content_hashes=finra_provider.last_raw_hashes,
                expect_market_sessions=True,
            )
            return True, payload
        if dataset == "earnings":
            sec_user_agent = ResearchProviderCredentials().sec_agent
            if sec_user_agent is None:
                raise ProviderError(
                    "SEC EDGAR is unconfigured; set EDGESTACK_SEC_USER_AGENT to a real contact"
                )
            edgar_provider = SecEdgarBulkProvider(
                raw_dir=Path(self.cfg.paths.data_dir) / "raw" / "sec_edgar",
                user_agent=sec_user_agent,
                timeout=max(120.0, self.cfg.data.request_timeout_seconds),
            )
            edgar_provider.fetch_bulk_snapshot("submissions")
            edgar_provider.fetch_bulk_snapshot("companyfacts")
            start_at = datetime.combine(
                date.fromisoformat(str(payload["start"])), datetime.min.time(), tzinfo=UTC
            )
            end_at = datetime.combine(
                date.fromisoformat(str(payload["end"])), datetime.max.time(), tzinfo=UTC
            )
            frame = edgar_provider.fetch_earnings_events(_symbols(payload), start_at, end_at)
            _write_grouped(
                frame,
                Path(self.cfg.paths.data_dir) / "curated" / "events",
                group="symbol",
            )
            _record_requested_event_coverage(
                self.store,
                frame,
                provider=edgar_provider.metadata.name,
                symbols=_symbols(payload),
                start=start_at.date(),
                end=end_at.date(),
                limitations=edgar_provider.metadata.limitations,
                raw_content_hashes=edgar_provider.last_raw_hashes,
            )
            return True, payload
        raise ProviderError(f"no acquisition handler for dataset {dataset!r}")

    def _acquire_market(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        provider_name = str(payload["provider"])
        provider = get_price_provider(provider_name, self.cfg)
        symbols = tuple(item.upper() for item in _symbols(payload))
        frequency = str(payload["frequency"])
        requested_start = date.fromisoformat(str(payload["start"]))
        requested_end = date.fromisoformat(str(payload["end"]))
        cursor = date.fromisoformat(str(payload.get("cursor_start") or requested_start.isoformat()))
        symbol_index = int(payload.get("cursor_symbol_index") or 0)
        batch = symbols[symbol_index : symbol_index + 8]
        if not batch:
            batch = symbols[:8]
            symbol_index = 0
        if frequency == "1d":
            chunk_days = 365
        elif provider_name == "yahoo" and frequency == "1m":
            chunk_days = 6
        elif provider_name == "yahoo":
            chunk_days = 30
        else:
            chunk_days = 30
        chunk_end = min(requested_end, cursor + timedelta(days=chunk_days - 1))
        if frequency == "1d":
            frame = provider.fetch_daily_bars(batch, cursor, chunk_end)
            self.catalog.write_bars(frame, provider=provider_name)
            _record_acquired_coverage(
                self.store,
                frame,
                dataset="prices",
                provider=provider_name,
                feed="SIP_DELAYED" if provider_name == "alpaca" else "YAHOO_CHART",
                frequency=frequency,
                group_column="symbol",
                timestamp_column="date",
                requested_names=batch,
                requested_start=cursor,
                requested_end=chunk_end,
                directory=self.catalog.prices_dir,
                point_in_time=provider.metadata.is_point_in_time,
                limitations=provider.metadata.limitations,
                raw_content_hashes=tuple(getattr(provider, "last_raw_hashes", ())),
                expect_market_sessions=True,
            )
        else:
            if not isinstance(provider, IntradayDataProvider):
                raise ProviderError(f"provider {provider_name!r} has no intraday capability")
            frame = provider.fetch_intraday_bars(
                batch,
                cursor,
                chunk_end,
                interval=frequency,
                include_prepost=True,
            )
            self.catalog.write_intraday_bars(frame, provider=provider_name)
            _record_acquired_coverage(
                self.store,
                frame,
                dataset="intraday",
                provider=provider_name,
                feed="SIP_DELAYED" if provider_name == "alpaca" else "YAHOO_CHART",
                frequency=frequency,
                group_column="symbol",
                timestamp_column="timestamp",
                requested_names=batch,
                requested_start=cursor,
                requested_end=chunk_end,
                directory=self.catalog.intraday_dir / frequency,
                point_in_time=provider.metadata.is_point_in_time,
                limitations=provider.metadata.limitations,
                raw_content_hashes=tuple(getattr(provider, "last_raw_hashes", ())),
                expect_market_sessions=True,
            )
            if frequency in {"1m", "5m"}:
                source_minutes = int(frequency[:-1])
                for target in (15, 60):
                    if target > source_minutes and target % source_minutes == 0:
                        derived = resample_intraday_bars(frame, target)
                        self.catalog.write_intraday_bars(
                            derived, provider=f"{provider_name}:{frequency}-derived"
                        )
                        _record_acquired_coverage(
                            self.store,
                            derived,
                            dataset="intraday",
                            provider=f"{provider_name}:{frequency}-derived",
                            feed=f"DETERMINISTIC_{frequency}_RESAMPLE",
                            frequency=f"{target}m",
                            group_column="symbol",
                            timestamp_column="timestamp",
                            requested_names=batch,
                            requested_start=cursor,
                            requested_end=chunk_end,
                            directory=self.catalog.intraday_dir / f"{target}m",
                            point_in_time=provider.metadata.is_point_in_time,
                            limitations=(
                                *provider.metadata.limitations,
                                f"Derived deterministically from normalized {frequency} bars.",
                            ),
                            raw_content_hashes=tuple(getattr(provider, "last_raw_hashes", ())),
                            expect_market_sessions=True,
                        )
        next_symbol_index = symbol_index + len(batch)
        next_cursor = cursor
        if next_symbol_index >= len(symbols):
            next_symbol_index = 0
            next_cursor = chunk_end + timedelta(days=1)
        payload["cursor_start"] = next_cursor.isoformat()
        payload["cursor_symbol_index"] = next_symbol_index
        complete = next_cursor > requested_end
        return complete, payload
