"""DuckDB persistence for the continuous edge factory."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from edgestack.data.catalog import DataCatalog
from edgestack.exceptions import DataError
from edgestack.recommendation.schemas import SleeveContributionV2
from edgestack.research.schemas import (
    AcquisitionJobV1,
    CampaignLifecycle,
    CampaignSummaryV1,
    DataCoverageV1,
    EvidenceGapV1,
    JobState,
    ShadowStrategyV1,
    WorkerHealthV1,
)

if TYPE_CHECKING:
    import pandas as pd

    from edgestack.recommendation.ensemble import PromotedEnsembleV1

_BLOCKED_STATES = {
    "BLOCKED_FREE_TIER",
    "BLOCKED_LICENSE",
    "BLOCKED_POINT_IN_TIME",
    "BLOCKED_COMPLIANCE",
    "BLOCKED_COMPONENTS",
}

_CAMPAIGN_TRANSITIONS = {
    "NEEDS_DATA": {"NEEDS_DATA", "READY", *_BLOCKED_STATES},
    "BLOCKED_FREE_TIER": {"BLOCKED_FREE_TIER", "NEEDS_DATA", "READY"},
    "BLOCKED_LICENSE": {"BLOCKED_LICENSE", "NEEDS_DATA", "READY"},
    "BLOCKED_POINT_IN_TIME": {"BLOCKED_POINT_IN_TIME", "NEEDS_DATA", "READY"},
    "BLOCKED_COMPLIANCE": {"BLOCKED_COMPLIANCE", "NEEDS_DATA", "READY"},
    "BLOCKED_COMPONENTS": {"BLOCKED_COMPONENTS", "NEEDS_DATA", "READY"},
    "READY": {"READY", "NEEDS_DATA", "RUNNING", *_BLOCKED_STATES},
    "RUNNING": {"RUNNING", "REJECTED", "SHADOW_ELIGIBLE", "PAPER_SHADOW"},
    "SHADOW_ELIGIBLE": {"SHADOW_ELIGIBLE", "PAPER_SHADOW", "REJECTED"},
    "PAPER_SHADOW": {
        "PAPER_SHADOW",
        "PROMOTED",
        "REJECTED",
        "DEGRADED",
        "SUSPENDED",
        "RETIRED",
    },
    "PROMOTED": {"PROMOTED", "DEGRADED", "SUSPENDED", "RETIRED"},
    "DEGRADED": {"DEGRADED", "PROMOTED", "SUSPENDED", "RETIRED"},
    "SUSPENDED": {"SUSPENDED", "PROMOTED", "RETIRED"},
    "REJECTED": {"REJECTED"},
    "RETIRED": {"RETIRED"},
}


def _json_payload(model: Any) -> str:
    return model.model_dump_json()


def _raw_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


class ResearchStore:
    """Small typed repository layered on the canonical :class:`DataCatalog`."""

    def __init__(self, catalog: DataCatalog) -> None:
        self.catalog = catalog

    def upsert_coverage(self, coverage: DataCoverageV1) -> None:
        with self.catalog.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO data_coverage_v1 VALUES (?, ?, ?)",
                [coverage.dataset_id, coverage.retrieved_at, _json_payload(coverage)],
            )

    def coverage(self) -> list[DataCoverageV1]:
        with self.catalog.connect() as con:
            rows = con.execute(
                "SELECT payload FROM data_coverage_v1 ORDER BY dataset_id"
            ).fetchall()
        return [DataCoverageV1.model_validate_json(_raw_json(row[0])) for row in rows]

    def upsert_gap(self, gap: EvidenceGapV1) -> None:
        now = datetime.now(UTC)
        with self.catalog.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO evidence_gaps_v1 VALUES (?, ?, ?, ?, ?)",
                [gap.requirement_id, gap.campaign_id, now, gap.state.value, _json_payload(gap)],
            )

    def gaps(self, *, campaign_id: str | None = None) -> list[EvidenceGapV1]:
        sql = "SELECT payload FROM evidence_gaps_v1"
        params: list[Any] = []
        if campaign_id is not None:
            sql += " WHERE campaign_id = ?"
            params.append(campaign_id)
        sql += " ORDER BY requirement_id"
        with self.catalog.connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [EvidenceGapV1.model_validate_json(_raw_json(row[0])) for row in rows]

    def upsert_campaign(self, campaign: CampaignSummaryV1) -> None:
        with self.catalog.connect() as con:
            existing = con.execute(
                "SELECT payload FROM research_campaigns_v1 WHERE campaign_id = ?",
                [campaign.campaign_id],
            ).fetchone()
            if existing is not None:
                prior = CampaignSummaryV1.model_validate_json(_raw_json(existing[0]))
                if prior.manifest_hash != campaign.manifest_hash:
                    raise DataError("research campaign manifest is immutable")
                allowed = _CAMPAIGN_TRANSITIONS[prior.lifecycle.value]
                if campaign.lifecycle.value not in allowed:
                    raise DataError(
                        f"invalid campaign transition {prior.lifecycle.value} -> "
                        f"{campaign.lifecycle.value}"
                    )
            con.execute(
                "INSERT OR REPLACE INTO research_campaigns_v1 VALUES (?, ?, ?, ?, ?, ?)",
                [
                    campaign.campaign_id,
                    campaign.manifest_hash,
                    campaign.lifecycle.value,
                    campaign.created_at,
                    campaign.updated_at,
                    _json_payload(campaign),
                ],
            )

    def campaigns(self) -> list[CampaignSummaryV1]:
        with self.catalog.connect() as con:
            rows = con.execute(
                "SELECT payload FROM research_campaigns_v1 ORDER BY created_at, campaign_id"
            ).fetchall()
        return [CampaignSummaryV1.model_validate_json(_raw_json(row[0])) for row in rows]

    def campaign(self, campaign_id: str) -> CampaignSummaryV1 | None:
        with self.catalog.connect() as con:
            row = con.execute(
                "SELECT payload FROM research_campaigns_v1 WHERE campaign_id = ?",
                [campaign_id],
            ).fetchone()
        return CampaignSummaryV1.model_validate_json(_raw_json(row[0])) if row else None

    def family_trial_count(self, family: str) -> int:
        """Global multiplicity ledger across every cohort of an economic family."""
        with self.catalog.connect() as con:
            row = con.execute(
                """SELECT COUNT(*) FROM trial_ledger_v2
                   WHERE json_extract_string(payload, '$.family') = ?""",
                [family],
            ).fetchone()
        return int(row[0]) if row else 0

    def family_trial_count_as_of(self, campaign_id: str, family: str) -> int:
        """Freeze cumulative family multiplicity at campaign registration time.

        Later campaigns pay for all earlier searches, while their registration
        cannot retroactively change an already content-addressed result.
        """
        with self.catalog.connect() as con:
            campaign = con.execute(
                "SELECT created_at FROM research_campaigns_v1 WHERE campaign_id = ?",
                [campaign_id],
            ).fetchone()
            if campaign is None:
                return 0
            row = con.execute(
                """SELECT COUNT(*)
                   FROM trial_ledger_v2 AS trial
                   JOIN research_campaigns_v1 AS campaign
                     ON trial.experiment_id = campaign.campaign_id
                   WHERE json_extract_string(trial.payload, '$.family') = ?
                     AND campaign.created_at <= ?""",
                [family, campaign[0]],
            ).fetchone()
        return int(row[0]) if row else 0

    def upsert_job(self, job: AcquisitionJobV1) -> None:
        with self.catalog.connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO acquisition_jobs_v1
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    job.job_id,
                    job.priority,
                    job.state.value,
                    job.attempts,
                    job.not_before,
                    job.lease_owner,
                    job.lease_expires_at,
                    job.created_at,
                    job.updated_at,
                    _json_payload(job),
                ],
            )

    def jobs(self, *, states: tuple[JobState, ...] | None = None) -> list[AcquisitionJobV1]:
        sql = "SELECT payload FROM acquisition_jobs_v1"
        params: list[Any] = []
        if states:
            placeholders = ",".join("?" for _ in states)
            sql += f" WHERE state IN ({placeholders})"
            params.extend(item.value for item in states)
        sql += " ORDER BY priority, created_at, job_id"
        with self.catalog.connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [AcquisitionJobV1.model_validate_json(_raw_json(row[0])) for row in rows]

    def lease_next_job(
        self,
        owner: str,
        *,
        lease_minutes: int = 30,
        now: datetime | None = None,
    ) -> AcquisitionJobV1 | None:
        """Atomically lease the highest-priority runnable job."""
        instant = now or datetime.now(UTC)
        expiry = instant + timedelta(minutes=lease_minutes)
        with self.catalog.connect() as con:
            con.execute("BEGIN TRANSACTION")
            row = con.execute(
                """
                SELECT payload FROM acquisition_jobs_v1
                WHERE (state IN ('PENDING', 'RETRY') AND (not_before IS NULL OR not_before <= ?))
                   OR (state = 'LEASED' AND lease_expires_at < ?)
                ORDER BY priority, created_at, job_id
                LIMIT 1
                """,
                [instant, instant],
            ).fetchone()
            if row is None:
                con.execute("COMMIT")
                return None
            job = AcquisitionJobV1.model_validate_json(_raw_json(row[0]))
            leased = job.model_copy(
                update={
                    "state": JobState.LEASED,
                    "lease_owner": owner,
                    "lease_expires_at": expiry,
                    "updated_at": instant,
                }
            )
            con.execute(
                """
                UPDATE acquisition_jobs_v1
                SET state = ?, lease_owner = ?, lease_expires_at = ?, updated_at = ?, payload = ?
                WHERE job_id = ?
                """,
                [
                    leased.state.value,
                    owner,
                    expiry,
                    instant,
                    _json_payload(leased),
                    leased.job_id,
                ],
            )
            con.execute("COMMIT")
        return leased

    def set_worker(self, worker: WorkerHealthV1, *, worker_id: str = "default") -> None:
        with self.catalog.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO research_worker_state_v1 VALUES (?, ?, ?)",
                [worker_id, datetime.now(UTC), _json_payload(worker)],
            )

    def renew_lease(
        self,
        job_id: str,
        owner: str,
        *,
        lease_minutes: int = 60,
    ) -> AcquisitionJobV1 | None:
        """Extend only the live lease held by ``owner``; leave terminal jobs unchanged."""
        with self.catalog.connect() as con:
            row = con.execute(
                "SELECT payload FROM acquisition_jobs_v1 WHERE job_id = ?",
                [job_id],
            ).fetchone()
            if row is None:
                return None
            job = AcquisitionJobV1.model_validate_json(_raw_json(row[0]))
            if job.state is not JobState.LEASED or job.lease_owner != owner:
                return None
            now = datetime.now(UTC)
            renewed = job.model_copy(
                update={
                    "lease_expires_at": now + timedelta(minutes=lease_minutes),
                    "updated_at": now,
                }
            )
            con.execute(
                """UPDATE acquisition_jobs_v1
                   SET lease_expires_at = ?, updated_at = ?, payload = ? WHERE job_id = ?""",
                [
                    renewed.lease_expires_at,
                    renewed.updated_at,
                    _json_payload(renewed),
                    renewed.job_id,
                ],
            )
        return renewed

    def worker(self, *, worker_id: str = "default") -> WorkerHealthV1 | None:
        with self.catalog.connect() as con:
            row = con.execute(
                "SELECT payload FROM research_worker_state_v1 WHERE worker_id = ?",
                [worker_id],
            ).fetchone()
        return WorkerHealthV1.model_validate_json(_raw_json(row[0])) if row else None

    def upsert_shadow(self, strategy: ShadowStrategyV1) -> None:
        with self.catalog.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO shadow_strategies_v1 VALUES (?, ?, ?, ?)",
                [
                    strategy.strategy_id,
                    strategy.campaign_id,
                    datetime.now(UTC),
                    _json_payload(strategy),
                ],
            )

    def shadows(self) -> list[ShadowStrategyV1]:
        with self.catalog.connect() as con:
            rows = con.execute(
                "SELECT payload FROM shadow_strategies_v1 ORDER BY updated_at, strategy_id"
            ).fetchall()
        return [ShadowStrategyV1.model_validate_json(_raw_json(row[0])) for row in rows]

    def save_promoted_sleeve(self, sleeve: SleeveContributionV2) -> None:
        with self.catalog.connect() as con:
            existing = con.execute(
                "SELECT payload FROM promoted_sleeves_v2 WHERE sleeve_id = ?",
                [sleeve.sleeve_id],
            ).fetchone()
            if existing is not None:
                if SleeveContributionV2.model_validate_json(_raw_json(existing[0])) != sleeve:
                    raise DataError(f"promoted sleeve {sleeve.sleeve_id} is immutable")
                return
            con.execute(
                "INSERT INTO promoted_sleeves_v2 VALUES (?, ?, ?, ?)",
                [
                    sleeve.sleeve_id,
                    sleeve.artifact_hash,
                    datetime.now(UTC),
                    sleeve.model_dump_json(),
                ],
            )

    def promoted_sleeves(self) -> tuple[SleeveContributionV2, ...]:
        with self.catalog.connect() as con:
            rows = con.execute(
                "SELECT payload FROM promoted_sleeves_v2 ORDER BY sleeve_id"
            ).fetchall()
        return tuple(SleeveContributionV2.model_validate_json(_raw_json(row[0])) for row in rows)

    def capital_eligible_sleeves(self) -> tuple[SleeveContributionV2, ...]:
        """Only immutable sleeves whose independent shadow remains PROMOTED."""
        promoted_shadows = {
            shadow.strategy_id
            for shadow in self.shadows()
            if shadow.status is CampaignLifecycle.PROMOTED
        }
        return tuple(
            sleeve for sleeve in self.promoted_sleeves() if sleeve.sleeve_id in promoted_shadows
        )

    def build_promoted_ensemble(
        self,
        returns: pd.DataFrame,
        *,
        max_weight: float = 0.60,
        max_family_weight: float = 0.70,
    ) -> PromotedEnsembleV1:
        """Allocate only across sleeves passing both registry and shadow gates."""
        from edgestack.recommendation.ensemble import (
            build_promoted_ensemble,
            ensemble_optimizer_trials,
        )
        from edgestack.recommendation.manifests import TrialStatus
        from edgestack.recommendation.registry import RecommendationRegistry

        sleeves = self.capital_eligible_sleeves()
        trials = ensemble_optimizer_trials(
            returns,
            capital_eligible_sleeves=sleeves,
            max_weight=max_weight,
            max_family_weight=max_family_weight,
        )
        registry = RecommendationRegistry(self.catalog)
        existing = {trial.trial_id: trial for trial in registry.trials(trials[0].experiment_id)}
        for trial in trials:
            prior = existing.get(trial.trial_id)
            if prior is None:
                registry.register_trial(trial)
                continue
            normalized = prior.model_copy(
                update={"status": TrialStatus.REGISTERED, "failure_reason": None}
            )
            if normalized != trial:
                raise DataError(f"optimizer trial identity collision: {trial.trial_id}")

        result = build_promoted_ensemble(
            returns,
            capital_eligible_sleeves=sleeves,
            registered_optimizer_trials=trials,
            max_weight=max_weight,
            max_family_weight=max_family_weight,
        )
        for trial in trials:
            registry.update_trial(trial.model_copy(update={"status": TrialStatus.SUCCEEDED}))
        return result
