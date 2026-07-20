"""Read-only query facade for research, coverage, shadow, and growth state."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from edgestack.data.catalog import DataCatalog
from edgestack.data.provider_credentials import ResearchProviderCredentials
from edgestack.exceptions import DataError
from edgestack.recommendation.schemas import RecommendationStatus
from edgestack.recommendation.service import CanonicalBundleRepository
from edgestack.research.proposals import ProposalRegistry
from edgestack.research.schemas import (
    CampaignSummaryV1,
    CandidateProposalV1,
    DataCoverageV1,
    GrowthDiagnosticsV1,
    ProposalAttemptV1,
    ProposalAuditV1,
    ResearchOverviewV1,
    ShadowStrategyV1,
    WorkerHealthV1,
)
from edgestack.research.store import ResearchStore


class ResearchQueryService:
    """Serve persisted server-owned state without triggering research work."""

    def __init__(self, catalog: DataCatalog):
        self.catalog = catalog
        self.store = ResearchStore(catalog)
        self.canonical = CanonicalBundleRepository(catalog.artifacts_dir)
        self.proposal_registry = ProposalRegistry(catalog)

    def overview(self) -> ResearchOverviewV1:
        campaigns = self.store.campaigns()
        proposals = self.proposal_registry.proposals()
        coverage = self.store.coverage()
        jobs = self.store.jobs()
        gaps = self.store.gaps()
        worker = self.store.worker() or WorkerHealthV1(
            max_workers=self.catalog.cfg.research.max_workers,
            storage_cap_gb=self.catalog.cfg.research.storage_cap_gb,
            process_priority=self.catalog.cfg.research.process_priority,
        )
        campaign_counts: dict[str, int] = {}
        job_counts: dict[str, int] = {}
        coverage_counts: dict[str, int] = {}
        for campaign in campaigns:
            campaign_counts[campaign.lifecycle.value] = (
                campaign_counts.get(campaign.lifecycle.value, 0) + 1
            )
        for job in jobs:
            job_counts[job.state.value] = job_counts.get(job.state.value, 0) + 1
        for item in coverage:
            coverage_counts[item.quality_status] = coverage_counts.get(item.quality_status, 0) + 1
        active = next(
            (
                item
                for item in campaigns
                if item.lifecycle.value in {"RUNNING", "READY", "NEEDS_DATA"}
            ),
            None,
        )
        blockers = tuple(
            sorted(
                {
                    reason
                    for campaign in campaigns
                    if campaign.lifecycle.value.startswith("BLOCKED")
                    or campaign.lifecycle.value == "REJECTED"
                    for reason in campaign.failure_reasons
                }
            )
        )
        next_jobs = tuple(item.job_id for item in jobs if item.state.value in {"PENDING", "RETRY"})[
            :10
        ]
        credentials = ResearchProviderCredentials()
        coverage_providers = {item.provider for item in coverage}
        return ResearchOverviewV1(
            generated_at=datetime.now(UTC),
            worker=worker,
            proposal_count=len(proposals),
            fresh_proposal_count=sum(
                self.proposal_registry.audit(item.proposal_id).intake_eligible_for_fresh_campaign
                for item in proposals
            ),
            campaign_counts=campaign_counts,
            acquisition_job_counts=job_counts,
            coverage_counts=coverage_counts,
            provider_health={
                "alpaca_delayed_sip": (
                    "CONFIGURED"
                    if credentials.alpaca_key_id and credentials.alpaca_secret_key
                    else "UNCONFIGURED"
                ),
                "fred_alfred": ("CONFIGURED" if credentials.fred_key else "UNCONFIGURED"),
                "sec_edgar": ("CONFIGURED" if credentials.sec_agent else "USER_AGENT_REQUIRED"),
                "finra_short_sale_volume": "AVAILABLE",
                "usaspending": (
                    "AVAILABLE_CURRENT_SNAPSHOT"
                    if "usaspending" in coverage_providers
                    else "AVAILABLE_NOT_ACQUIRED"
                ),
                "fred_graph_dgs3mo": (
                    "AVAILABLE_CURRENT_SNAPSHOT"
                    if "fred" in coverage_providers
                    else "AVAILABLE_NOT_ACQUIRED"
                ),
                "yahoo_fallback": "AVAILABLE_LIMITED_INTRADAY",
            },
            current_campaign=active,
            evidence_gaps=tuple(gaps),
            next_jobs=next_jobs,
            blockers=blockers,
        )

    def campaigns(self) -> tuple[CampaignSummaryV1, ...]:
        return tuple(self.store.campaigns())

    def campaign(self, campaign_id: str) -> CampaignSummaryV1:
        campaign = self.store.campaign(campaign_id)
        if campaign is None:
            raise DataError(f"unknown research campaign: {campaign_id}")
        return campaign

    def coverage(self) -> tuple[DataCoverageV1, ...]:
        return tuple(self.store.coverage())

    def proposals(self) -> tuple[CandidateProposalV1, ...]:
        return self.proposal_registry.proposals()

    def proposal(self, proposal_id: str) -> CandidateProposalV1:
        proposal = self.proposal_registry.proposal(proposal_id)
        if proposal is None:
            raise DataError(f"unknown research proposal: {proposal_id}")
        return proposal

    def proposal_attempts(self, proposal_id: str) -> tuple[ProposalAttemptV1, ...]:
        self.proposal(proposal_id)
        return self.proposal_registry.attempts(proposal_id)

    def proposal_audit(self, proposal_id: str) -> ProposalAuditV1:
        return self.proposal_registry.audit(proposal_id)

    def shadows(self) -> tuple[ShadowStrategyV1, ...]:
        return tuple(self.store.shadows())

    def growth_diagnostics(self) -> GrowthDiagnosticsV1:
        """Project only the atomically published canonical recommendation."""
        try:
            pointer = self.canonical.pointer()
            bundle = self.canonical.latest()
        except DataError:
            return GrowthDiagnosticsV1(
                as_of=datetime.now(UTC),
                action="NO_ALLOCATION",
                expected_log_growth=0.0,
                expected_log_growth_lower_95=0.0,
                quarter_kelly_limit=0.0,
                effective_leverage=0.0,
                evidence_state="INSUFFICIENT",
                warnings=("No canonical recommendation has been published.",),
            )
        recommendation = bundle.default_recommendation
        monitoring = self.canonical.monitoring()
        limits = {item.name: item.leverage_limit for item in recommendation.constraints}
        action: Literal["ENTER_NEXT_OPEN", "WAIT", "REDUCE", "NO_ALLOCATION"] = "NO_ALLOCATION"
        reduction_constraints = {
            "drawdown_state",
            "market_freshness",
            "funding_freshness",
            "allocation_health",
        }
        promoted_count = int(monitoring.get("promoted_sleeves", 0))
        active_promoted_count = int(monitoring.get("active_promoted_sleeves", 0))
        if promoted_count == 0:
            limits["evidence_gate"] = 0.0
            return GrowthDiagnosticsV1(
                canonical_run_id=pointer.run_id,
                as_of=bundle.as_of,
                action="NO_ALLOCATION",
                expected_log_growth=0.0,
                expected_log_growth_lower_95=0.0,
                comparator_log_growth={},
                quarter_kelly_limit=0.0,
                effective_leverage=0.0,
                constraint_limits=limits,
                binding_constraints=("evidence_gate",),
                evidence_state="WATCHLIST_ONLY",
                warnings=(
                    *recommendation.warnings,
                    "No independently promoted sleeve exists; research leads remain "
                    "capital-ineligible.",
                ),
            )
        if promoted_count > 0:
            if reduction_constraints.intersection(recommendation.binding_constraints):
                action = "REDUCE"
            elif active_promoted_count == 0:
                action = "WAIT"
            elif recommendation.status is RecommendationStatus.ACTIVE:
                action = "ENTER_NEXT_OPEN"
            else:
                action = "WAIT"
        variance_drag = 0.5 * recommendation.expected_volatility**2
        expected_log_growth = recommendation.expected_net_return - variance_drag
        promoted_lower = sum(
            sleeve.expected_return_lower_95
            for sleeve in (
                *bundle.base_recommendation.promoted_sleeves,
                *bundle.base_recommendation.promoted_compound_sleeves,
            )
        )
        expected_lower = promoted_lower * recommendation.effective_leverage - variance_drag
        if promoted_count == 0:
            expected_lower = 0.0
        return GrowthDiagnosticsV1(
            canonical_run_id=pointer.run_id,
            as_of=bundle.as_of,
            action=action,
            expected_log_growth=expected_log_growth,
            expected_log_growth_lower_95=expected_lower,
            comparator_log_growth={
                str(name): float(value)
                for name, value in monitoring.get("comparator_log_growth", {}).items()
            },
            quarter_kelly_limit=limits.get("quarter_kelly", 0.0),
            effective_leverage=recommendation.effective_leverage,
            constraint_limits=limits,
            binding_constraints=recommendation.binding_constraints,
            evidence_state=(
                "PROMOTED" if promoted_count > 0 else recommendation.evidence_grade.value
            ),
            warnings=recommendation.warnings,
        )
