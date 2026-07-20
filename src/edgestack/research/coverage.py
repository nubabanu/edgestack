"""Coverage inventory and evidence-gap to acquisition-job planning."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from edgestack.data.calendar import TradingCalendar
from edgestack.data.catalog import DataCatalog
from edgestack.recommendation.hashing import stable_hash
from edgestack.research.schemas import (
    AcquisitionJobV1,
    CoverageState,
    DataCoverageV1,
    EvidenceGapV1,
    JobState,
)
from edgestack.research.store import ResearchStore


@dataclass(frozen=True)
class DataRequirement:
    campaign_id: str
    dataset: str
    symbols: tuple[str, ...]
    frequency: str
    start: date
    end: date
    required_observations: int
    preferred_providers: tuple[str, ...]
    priority: int = 20

    @property
    def requirement_id(self) -> str:
        return stable_hash(
            {
                "campaign_id": self.campaign_id,
                "dataset": self.dataset,
                "symbols": self.symbols,
                "frequency": self.frequency,
                "start": self.start,
                "end": self.end,
                "required_observations": self.required_observations,
                "preferred_providers": self.preferred_providers,
                "priority": self.priority,
            }
        )[:24]


@dataclass(frozen=True)
class ProviderCapability:
    provider: str
    datasets: tuple[str, ...]
    frequencies: tuple[str, ...]
    free_tier: bool = True
    configured: bool = True
    max_range_days: dict[str, int] | None = None

    def supports(self, requirement: DataRequirement) -> bool:
        range_supported = True
        if self.max_range_days and requirement.frequency in self.max_range_days:
            requested_days = (requirement.end - requirement.start).days + 1
            range_supported = requested_days <= self.max_range_days[requirement.frequency]
        return (
            self.free_tier
            and self.configured
            and requirement.dataset in self.datasets
            and (requirement.frequency in self.frequencies or "*" in self.frequencies)
            and range_supported
        )


def default_free_capabilities(
    *, alpaca_configured: bool, fred_configured: bool, sec_configured: bool
) -> tuple[ProviderCapability, ...]:
    return (
        ProviderCapability(
            "alpaca",
            ("prices", "intraday"),
            ("1d", "1m", "5m", "15m", "60m"),
            configured=alpaca_configured,
        ),
        ProviderCapability(
            "yahoo",
            ("prices", "intraday"),
            ("1d", "1m", "5m", "15m", "60m"),
            max_range_days={"1m": 7, "5m": 60, "15m": 60, "60m": 730},
        ),
        ProviderCapability("fred", ("macro",), ("*",), configured=fred_configured),
        ProviderCapability(
            "sec_edgar",
            ("earnings", "fundamentals"),
            ("event", "quarterly"),
            configured=sec_configured,
        ),
        ProviderCapability(
            "finra",
            ("short_sale_volume",),
            ("1d",),
            # The query API is a rolling window, not a full historical archive.
            max_range_days={"1d": 730},
        ),
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coverage_from_frame(
    path: Path,
    *,
    dataset: str,
    provider: str,
    feed: str,
    symbol: str,
    frequency: str,
    timestamp_column: str,
) -> DataCoverageV1:
    frame = pd.read_parquet(path, columns=[timestamp_column])
    values = pd.to_datetime(frame[timestamp_column], utc=timestamp_column == "timestamp")
    unique = values.dropna().drop_duplicates()
    missing_sessions = 0
    if dataset in {"prices", "intraday"} and len(unique):
        observed_sessions = {item.date() for item in unique}
        expected_sessions = TradingCalendar().sessions(
            min(observed_sessions), max(observed_sessions)
        )
        missing_sessions = len({item.date() for item in expected_sessions} - observed_sessions)
    limitations = ["Curated catalog inventory; provider request metadata is stored separately."]
    if dataset == "short_sale_volume":
        limitations.append(
            "FINRA observations are off-exchange short-sale volume, not short interest."
        )
    return DataCoverageV1(
        dataset_id=f"{dataset}:{provider}:{feed}:{frequency}:{symbol}",
        dataset=dataset,
        provider=provider,
        feed=feed,
        symbol=symbol,
        frequency=frequency,
        start=unique.min().to_pydatetime() if len(unique) else None,
        end=unique.max().to_pydatetime() if len(unique) else None,
        observation_count=len(unique),
        missing_sessions=missing_sessions,
        quality_status=(
            "PASS" if len(unique) and missing_sessions == 0 else "GAPS" if len(unique) else "EMPTY"
        ),
        point_in_time=dataset in {"macro", "earnings", "short_sale_volume", "intraday"},
        content_hash=_file_hash(path),
        retrieved_at=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
        quality_results={
            "returned_observations": len(unique),
            "duplicate_observations": len(values) - len(unique),
            "missing_sessions": missing_sessions,
        },
        limitations=tuple(limitations),
    )


def scan_catalog_coverage(catalog: DataCatalog) -> list[DataCoverageV1]:
    """Inventory local daily and intraday Parquet without changing price data."""
    output: list[DataCoverageV1] = []
    if catalog.prices_dir.exists():
        for path in sorted(catalog.prices_dir.glob("*.parquet")):
            output.append(
                _coverage_from_frame(
                    path,
                    dataset="prices",
                    provider="catalog",
                    feed="curated",
                    symbol=path.stem.upper(),
                    frequency="1d",
                    timestamp_column="date",
                )
            )
    if catalog.intraday_dir.exists():
        for directory in sorted(catalog.intraday_dir.glob("*m")):
            if not directory.is_dir() or not directory.name[:-1].isdigit():
                continue
            for path in sorted(directory.glob("*.parquet")):
                output.append(
                    _coverage_from_frame(
                        path,
                        dataset="intraday",
                        provider="catalog",
                        feed="curated",
                        symbol=path.stem.upper(),
                        frequency=directory.name,
                        timestamp_column="timestamp",
                    )
                )
    extra_sets = (
        (catalog.data_dir / "curated" / "events", "earnings", "event", "accepted_at"),
        (catalog.data_dir / "curated" / "macro", "macro", "daily", "observation_date"),
        (
            catalog.data_dir / "curated" / "short_sale_volume",
            "short_sale_volume",
            "1d",
            "date",
        ),
    )
    for directory, dataset, frequency, timestamp_column in extra_sets:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.parquet")):
            try:
                output.append(
                    _coverage_from_frame(
                        path,
                        dataset=dataset,
                        provider="catalog",
                        feed="curated",
                        symbol=path.stem.upper(),
                        frequency=frequency,
                        timestamp_column=timestamp_column,
                    )
                )
            except (KeyError, ValueError):
                continue
    return output


class CoveragePlanner:
    def __init__(
        self,
        store: ResearchStore,
        capabilities: tuple[ProviderCapability, ...],
    ) -> None:
        self.store = store
        self.capabilities = capabilities

    def refresh_catalog(self) -> list[DataCoverageV1]:
        entries = scan_catalog_coverage(self.store.catalog)
        for entry in entries:
            self.store.upsert_coverage(entry)
        return entries

    def plan(self, requirement: DataRequirement) -> EvidenceGapV1:
        matching = [
            item
            for item in self.store.coverage()
            if item.dataset == requirement.dataset
            and item.symbol in requirement.symbols
            and item.frequency == requirement.frequency
        ]
        by_symbol: dict[str, DataCoverageV1] = {}
        for item in matching:
            incumbent = by_symbol.get(item.symbol)
            item_span = (
                (item.end - item.start).total_seconds()
                if item.start is not None and item.end is not None
                else -1.0
            )
            incumbent_span = (
                (incumbent.end - incumbent.start).total_seconds()
                if incumbent is not None
                and incumbent.start is not None
                and incumbent.end is not None
                else -1.0
            )
            item_rank = (item_span, item.observation_count, item.quality_status == "PASS")
            incumbent_rank = (
                incumbent_span,
                incumbent.observation_count if incumbent is not None else -1,
                incumbent.quality_status == "PASS" if incumbent is not None else False,
            )
            if incumbent is None or item_rank > incumbent_rank:
                by_symbol[item.symbol] = item
        observed = min(
            (
                by_symbol[symbol].observation_count
                for symbol in requirement.symbols
                if symbol in by_symbol
            ),
            default=0,
        )
        all_present = all(symbol in by_symbol for symbol in requirement.symbols)
        required_start = requirement.start
        required_end = requirement.end
        if requirement.dataset in {"prices", "intraday", "short_sale_volume"}:
            required_sessions = TradingCalendar().sessions(requirement.start, requirement.end)
            if len(required_sessions):
                required_start = required_sessions[0].date()
                required_end = required_sessions[-1].date()

        def covers(item: DataCoverageV1) -> bool:
            start = item.start
            end = item.end
            return bool(
                start is not None
                and end is not None
                and start.date() <= required_start
                and end.date() >= required_end
                and item.quality_status == "PASS"
            )

        covers_dates = all(
            covers(by_symbol[symbol]) for symbol in requirement.symbols if symbol in by_symbol
        )
        if all_present and covers_dates and observed >= requirement.required_observations:
            for job in self.store.jobs():
                if job.requirement_id == requirement.requirement_id and job.state in {
                    JobState.PENDING,
                    JobState.RETRY,
                }:
                    self.store.upsert_job(
                        job.model_copy(
                            update={
                                "state": JobState.SUCCEEDED,
                                "updated_at": datetime.now(UTC),
                                "not_before": None,
                                "lease_owner": None,
                                "lease_expires_at": None,
                                "last_error": None,
                            }
                        )
                    )
            gap = EvidenceGapV1(
                requirement_id=requirement.requirement_id,
                campaign_id=requirement.campaign_id,
                dataset=requirement.dataset,
                symbols=requirement.symbols,
                frequency=requirement.frequency,
                start=requirement.start,
                end=requirement.end,
                required_observations=requirement.required_observations,
                observed_observations=observed,
                state=CoverageState.READY,
                next_action="Run the frozen campaign without changing its manifest.",
            )
            self.store.upsert_gap(gap)
            return gap

        providers = {item.provider: item for item in self.capabilities}
        selected = next(
            (
                providers[name]
                for name in requirement.preferred_providers
                if name in providers and providers[name].supports(requirement)
            ),
            None,
        )
        fallbacks = [
            name
            for name in requirement.preferred_providers
            if name != (selected.provider if selected else None)
            and name in providers
            and providers[name].supports(requirement)
        ]
        if selected is None:
            gap = EvidenceGapV1(
                requirement_id=requirement.requirement_id,
                campaign_id=requirement.campaign_id,
                dataset=requirement.dataset,
                symbols=requirement.symbols,
                frequency=requirement.frequency,
                start=requirement.start,
                end=requirement.end,
                required_observations=requirement.required_observations,
                observed_observations=observed,
                state=CoverageState.BLOCKED_FREE_TIER,
                next_action=(
                    "No configured free provider can satisfy this requirement; "
                    "park it and continue other campaigns."
                ),
            )
            self.store.upsert_gap(gap)
            return gap

        now = datetime.now(UTC)
        job_id = stable_hash(
            {"requirement": requirement.requirement_id, "provider": selected.provider}
        )[:24]
        jobs_by_id = {item.job_id: item for item in self.store.jobs()}
        existing = jobs_by_id.get(job_id)
        while existing is not None and existing.state.value == "SUCCEEDED":
            raw_remaining = existing.payload.get("fallback_providers")
            remaining = list(raw_remaining) if isinstance(raw_remaining, list) else []
            next_provider = next(
                (
                    name
                    for name in remaining
                    if name in providers and providers[name].supports(requirement)
                ),
                None,
            )
            if next_provider is None:
                break
            selected = providers[next_provider]
            fallbacks = [name for name in remaining if name != next_provider]
            job_id = stable_hash(
                {"requirement": requirement.requirement_id, "provider": selected.provider}
            )[:24]
            existing = jobs_by_id.get(job_id)
        if existing is not None and existing.state.value in {
            "SUCCEEDED",
            "FAILED",
            "BLOCKED_FREE_TIER",
        }:
            gap = EvidenceGapV1(
                requirement_id=requirement.requirement_id,
                campaign_id=requirement.campaign_id,
                dataset=requirement.dataset,
                symbols=requirement.symbols,
                frequency=requirement.frequency,
                start=requirement.start,
                end=requirement.end,
                required_observations=requirement.required_observations,
                observed_observations=observed,
                state=CoverageState.BLOCKED_FREE_TIER,
                provider=selected.provider,
                next_action=(
                    "The configured free source completed or exhausted its retries without "
                    "meeting this frozen requirement; park the campaign."
                ),
                job_id=job_id,
            )
            self.store.upsert_gap(gap)
            return gap
        if existing is None:
            self.store.upsert_job(
                AcquisitionJobV1(
                    job_id=job_id,
                    requirement_id=requirement.requirement_id,
                    campaign_id=requirement.campaign_id,
                    kind="ACQUIRE_DATA",
                    priority=requirement.priority,
                    created_at=now,
                    updated_at=now,
                    payload={
                        "dataset": requirement.dataset,
                        "symbols": list(requirement.symbols),
                        "frequency": requirement.frequency,
                        "start": requirement.start.isoformat(),
                        "end": requirement.end.isoformat(),
                        "provider": selected.provider,
                        "fallback_providers": fallbacks,
                    },
                )
            )
        gap = EvidenceGapV1(
            requirement_id=requirement.requirement_id,
            campaign_id=requirement.campaign_id,
            dataset=requirement.dataset,
            symbols=requirement.symbols,
            frequency=requirement.frequency,
            start=requirement.start,
            end=requirement.end,
            required_observations=requirement.required_observations,
            observed_observations=observed,
            state=CoverageState.NEEDS_DATA,
            provider=selected.provider,
            next_action=(
                f"Acquisition job {job_id} will fetch the missing {requirement.frequency} coverage."
            ),
            job_id=job_id,
        )
        self.store.upsert_gap(gap)
        return gap
