"""Auditable intake for human- and agent-generated research hypotheses.

This module borrows the useful automation boundary from agentic research
systems: agents may propose finite experiments, but they do not receive a
privileged path around trial accounting, test-period locks, or promotion.
"""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import product
from typing import Any

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog
from edgestack.data.provider_credentials import ResearchProviderCredentials
from edgestack.exceptions import DataError
from edgestack.recommendation.hashing import stable_hash
from edgestack.recommendation.manifests import TrialKind, TrialRecordV2, TrialStatus
from edgestack.research.coverage import (
    CoveragePlanner,
    DataRequirement,
    default_free_capabilities,
)
from edgestack.research.schemas import (
    CampaignLifecycle,
    CampaignSummaryV1,
    CandidateProposalV1,
    ProposalAttemptV1,
    ProposalAuditV1,
    ProposalStage,
)
from edgestack.research.store import ResearchStore


def _expanded_parameters(proposal: CandidateProposalV1) -> tuple[dict[str, Any], ...]:
    names = tuple(sorted(proposal.parameter_grid))
    if not names:
        return ({},)
    return tuple(
        dict(zip(names, values, strict=True))
        for values in product(*(proposal.parameter_grid[name] for name in names))
    )


def _trial(proposal: CandidateProposalV1, parameters: dict[str, Any]) -> TrialRecordV2:
    identity = {
        "proposal_manifest_hash": proposal.manifest_hash,
        "parameters": parameters,
    }
    raw_horizon = parameters.get(
        "horizon_sessions",
        parameters.get("horizon", parameters.get("hold", 1)),
    )
    if isinstance(raw_horizon, bool) or not isinstance(raw_horizon, (int, float)):
        horizon = 1
    else:
        horizon = max(1, int(raw_horizon))
    return TrialRecordV2(
        trial_id=stable_hash(identity)[:24],
        experiment_id=proposal.proposal_id,
        kind=TrialKind.STANDALONE,
        family=proposal.family,
        horizon_sessions=horizon,
        parameters={
            **parameters,
            "feature_expressions": list(proposal.feature_expressions),
            "proposal_manifest_hash": proposal.manifest_hash,
        },
        status=TrialStatus.REGISTERED,
    )


class ProposalRegistry:
    """Immutable proposal, attempt, and pre-evaluation trial ledger."""

    def __init__(self, catalog: DataCatalog) -> None:
        self.catalog = catalog

    def register(self, proposal: CandidateProposalV1) -> ProposalAuditV1:
        """Register the complete grid atomically before any evaluation."""
        trials = tuple(
            _trial(proposal, parameters) for parameters in _expanded_parameters(proposal)
        )
        if len(trials) != proposal.trial_count:
            raise DataError("expanded proposal trial count is inconsistent")
        campaign = CampaignSummaryV1(
            campaign_id=proposal.proposal_id,
            manifest_hash=proposal.manifest_hash,
            name=proposal.name,
            family=proposal.family,
            lifecycle=CampaignLifecycle.NEEDS_DATA,
            created_at=proposal.created_at,
            updated_at=proposal.created_at,
            trial_count=len(trials),
            data_requirements=tuple(
                query.requirement_id(proposal.proposal_id) for query in proposal.data_queries
            ),
            next_action=(
                "Acquire the frozen data requirements, then bind this proposal to a "
                "registered evaluator without changing its manifest."
            ),
            promotion_eligible=False,
            previously_accessed=proposal.previously_accessed,
        )
        changed = False
        with self.catalog.connect() as con:
            con.execute("BEGIN TRANSACTION")
            try:
                prior = con.execute(
                    "SELECT payload FROM research_proposals_v1 WHERE proposal_id = ?",
                    [proposal.proposal_id],
                ).fetchone()
                if prior is not None:
                    existing = CandidateProposalV1.model_validate_json(str(prior[0]))
                    if existing != proposal:
                        raise DataError(f"proposal {proposal.proposal_id} is immutable")
                else:
                    changed = True
                    con.execute(
                        "INSERT INTO research_proposals_v1 VALUES (?, ?, ?, ?)",
                        [
                            proposal.proposal_id,
                            proposal.manifest_hash,
                            proposal.created_at,
                            proposal.model_dump_json(),
                        ],
                    )

                prior_campaign = con.execute(
                    "SELECT payload FROM research_campaigns_v1 WHERE campaign_id = ?",
                    [proposal.proposal_id],
                ).fetchone()
                if prior_campaign is not None:
                    existing_campaign = CampaignSummaryV1.model_validate_json(
                        str(prior_campaign[0])
                    )
                    if existing_campaign.manifest_hash != proposal.manifest_hash:
                        raise DataError("proposal id collides with another frozen campaign")
                else:
                    changed = True
                    con.execute(
                        "INSERT INTO research_campaigns_v1 VALUES (?, ?, ?, ?, ?, ?)",
                        [
                            campaign.campaign_id,
                            campaign.manifest_hash,
                            campaign.lifecycle.value,
                            campaign.created_at,
                            campaign.updated_at,
                            campaign.model_dump_json(),
                        ],
                    )

                for trial in trials:
                    existing_trial = con.execute(
                        "SELECT payload FROM trial_ledger_v2 WHERE trial_id = ?",
                        [trial.trial_id],
                    ).fetchone()
                    if existing_trial is not None:
                        if TrialRecordV2.model_validate_json(str(existing_trial[0])) != trial:
                            raise DataError(f"trial hash collision: {trial.trial_id}")
                        continue
                    changed = True
                    con.execute(
                        "INSERT INTO trial_ledger_v2 VALUES (?, ?, ?, ?, ?)",
                        [
                            trial.trial_id,
                            trial.experiment_id,
                            proposal.created_at,
                            trial.status.value,
                            trial.model_dump_json(),
                        ],
                    )
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
        if changed:
            self.catalog.audit(
                "research_proposal_registered",
                reason="finite external hypothesis registered before evaluation",
                experiment_id=proposal.proposal_id,
                manifest_hash=proposal.manifest_hash,
                trial_count=len(trials),
                source=proposal.source,
            )
        return self.audit(proposal.proposal_id)

    def proposals(self) -> tuple[CandidateProposalV1, ...]:
        with self.catalog.connect() as con:
            rows = con.execute(
                "SELECT payload FROM research_proposals_v1 ORDER BY created_at, proposal_id"
            ).fetchall()
        return tuple(CandidateProposalV1.model_validate_json(str(row[0])) for row in rows)

    def proposal(self, proposal_id: str) -> CandidateProposalV1 | None:
        with self.catalog.connect() as con:
            row = con.execute(
                "SELECT payload FROM research_proposals_v1 WHERE proposal_id = ?",
                [proposal_id],
            ).fetchone()
        return CandidateProposalV1.model_validate_json(str(row[0])) if row else None

    def append_attempt(self, attempt: ProposalAttemptV1) -> ProposalAuditV1:
        """Append exactly one contiguous audit event; history is never updated."""
        proposal = self.proposal(attempt.proposal_id)
        if proposal is None:
            raise DataError(f"unknown proposal: {attempt.proposal_id}")
        if attempt.stage is ProposalStage.FINAL_HOLDOUT and not attempt.viewed_guarded_data:
            raise DataError("FINAL_HOLDOUT attempts must declare guarded-data access")
        query_ids = {query.requirement_id(proposal.proposal_id) for query in proposal.data_queries}
        if attempt.stage is ProposalStage.DATA_QUERY and attempt.data_query_id is None:
            raise DataError("DATA_QUERY attempts must identify a frozen proposal query")
        if attempt.data_query_id is not None and attempt.data_query_id not in query_ids:
            raise DataError("attempt references a data query outside the frozen proposal")
        if (attempt.stage is ProposalStage.REVISION) != (
            attempt.revision_of_attempt_id is not None
        ):
            raise DataError("REVISION attempts must identify exactly one earlier attempt")
        with self.catalog.connect() as con:
            rows = con.execute(
                """SELECT attempt_id, sequence FROM research_proposal_attempts_v1
                   WHERE proposal_id = ? ORDER BY sequence""",
                [attempt.proposal_id],
            ).fetchall()
            expected = len(rows) + 1
            if attempt.sequence != expected:
                raise DataError(
                    f"proposal attempt sequence must be contiguous; expected {expected}"
                )
            if any(str(row[0]) == attempt.attempt_id for row in rows):
                raise DataError(f"attempt already registered: {attempt.attempt_id}")
            registered_attempt_ids = {str(row[0]) for row in rows}
            if (
                attempt.revision_of_attempt_id is not None
                and attempt.revision_of_attempt_id not in registered_attempt_ids
            ):
                raise DataError("revision target must be an earlier attempt in the same proposal")
            con.execute(
                "INSERT INTO research_proposal_attempts_v1 VALUES (?, ?, ?, ?, ?, ?)",
                [
                    attempt.attempt_id,
                    attempt.proposal_id,
                    attempt.sequence,
                    attempt.occurred_at,
                    attempt.stage.value,
                    attempt.model_dump_json(),
                ],
            )
        self.catalog.audit(
            "research_proposal_attempt",
            reason=attempt.stage.value,
            experiment_id=attempt.proposal_id,
            attempt_id=attempt.attempt_id,
            sequence=attempt.sequence,
            viewed_guarded_data=attempt.viewed_guarded_data,
        )
        return self.audit(attempt.proposal_id)

    def attempts(self, proposal_id: str) -> tuple[ProposalAttemptV1, ...]:
        with self.catalog.connect() as con:
            rows = con.execute(
                """SELECT payload FROM research_proposal_attempts_v1
                   WHERE proposal_id = ? ORDER BY sequence""",
                [proposal_id],
            ).fetchall()
        return tuple(ProposalAttemptV1.model_validate_json(str(row[0])) for row in rows)

    def audit(self, proposal_id: str) -> ProposalAuditV1:
        proposal = self.proposal(proposal_id)
        if proposal is None:
            raise DataError(f"unknown proposal: {proposal_id}")
        attempts = self.attempts(proposal_id)
        with self.catalog.connect() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM trial_ledger_v2 WHERE experiment_id = ?",
                [proposal_id],
            ).fetchone()
        registered = int(row[0]) if row else 0
        complete_trials = registered == proposal.trial_count
        sequences = [item.sequence for item in attempts]
        complete_sequence = sequences == list(range(1, len(sequences) + 1))
        holdout_accesses = sum(item.stage is ProposalStage.FINAL_HOLDOUT for item in attempts)
        guarded = any(item.viewed_guarded_data for item in attempts)
        reasons: list[str] = []
        if not complete_trials:
            reasons.append("The complete parameter grid is not present in the global trial ledger.")
        if not complete_sequence:
            reasons.append("The attempt history is not contiguous.")
        if proposal.previously_accessed:
            reasons.append("The source proposal declares previously accessed results.")
        if not proposal.result_blind:
            reasons.append("The source proposal was not generated blind to evaluation results.")
        if guarded:
            reasons.append("An external attempt accessed guarded data.")
        if holdout_accesses > 1:
            reasons.append("The external workflow accessed the final holdout repeatedly.")
        eligible = bool(
            complete_trials
            and complete_sequence
            and proposal.result_blind
            and not proposal.previously_accessed
            and not guarded
            and holdout_accesses == 0
        )
        return ProposalAuditV1(
            proposal_id=proposal.proposal_id,
            manifest_hash=proposal.manifest_hash,
            expected_trials=proposal.trial_count,
            registered_trials=registered,
            attempts=len(attempts),
            final_holdout_accesses=holdout_accesses,
            complete_trial_registry=complete_trials,
            complete_attempt_sequence=complete_sequence,
            guarded_data_accessed=guarded,
            intake_eligible_for_fresh_campaign=eligible,
            reasons=tuple(reasons),
        )


def register_and_plan_proposal(
    cfg: EdgeStackConfig,
    proposal: CandidateProposalV1,
) -> ProposalAuditV1:
    """Register a proposal, inventory local data, and enqueue viable gaps."""
    catalog = DataCatalog(cfg)
    registry = ProposalRegistry(catalog)
    audit = registry.register(proposal)
    credentials = ResearchProviderCredentials()
    planner = CoveragePlanner(
        store=ResearchStore(catalog),
        capabilities=default_free_capabilities(
            alpaca_configured=bool(credentials.alpaca_key_id and credentials.alpaca_secret_key),
            fred_configured=bool(credentials.fred_key),
            sec_configured=bool(credentials.sec_agent),
        ),
    )
    planner.refresh_catalog()
    for query in proposal.data_queries:
        planner.plan(
            DataRequirement(
                campaign_id=proposal.proposal_id,
                dataset=query.dataset,
                symbols=query.symbols,
                frequency=query.frequency,
                start=query.start,
                end=query.end,
                required_observations=query.required_observations,
                preferred_providers=query.preferred_providers,
                priority=30,
            )
        )
    return audit


def make_attempt_id(
    proposal_id: str,
    sequence: int,
    occurred_at: datetime | None = None,
) -> str:
    """Build a deterministic-enough id while keeping the timestamp explicit."""
    instant = occurred_at or datetime.now(UTC)
    return stable_hash({"proposal_id": proposal_id, "sequence": sequence, "occurred_at": instant})[
        :24
    ]
