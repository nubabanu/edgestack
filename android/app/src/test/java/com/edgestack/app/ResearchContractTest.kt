package com.edgestack.app

import com.edgestack.app.core.AppJson
import com.edgestack.app.domain.model.ResearchSnapshotV1
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ResearchContractTest {
    @Test
    fun edgeFactoryContractsRoundTripForOfflineDisplay() {
        val payload = """
            {
              "overview": {
                "generated_at": "2026-07-20T12:00:00Z",
                "worker": {
                  "state": "RUNNING",
                  "current_job_id": "job-1",
                  "storage_used_gb": 12.5,
                  "storage_cap_gb": 75.0
                },
                "proposal_count": 3,
                "fresh_proposal_count": 1,
                "campaign_counts": {"NEEDS_DATA": 1, "PAPER_SHADOW": 1},
                "acquisition_job_counts": {"LEASED": 1},
                "coverage_counts": {"PASS": 10},
                "provider_health": {"alpaca_delayed_sip": "CONFIGURED"},
                "evidence_gaps": [{
                  "requirement_id": "req",
                  "campaign_id": "campaign",
                  "dataset": "intraday",
                  "symbols": ["SPY"],
                  "frequency": "1m",
                  "start": "2016-01-04",
                  "end": "2026-06-30",
                  "required_observations": 98280,
                  "observed_observations": 50000,
                  "state": "NEEDS_DATA",
                  "provider": "alpaca",
                  "next_action": "resume acquisition"
                }]
              },
              "campaigns": [{
                "campaign_id": "earnings",
                "manifest_hash": "abc",
                "name": "Filing drift",
                "family": "earnings_filing_drift",
                "lifecycle": "REJECTED",
                "created_at": "2026-07-20T12:00:00Z",
                "updated_at": "2026-07-20T13:00:00Z",
                "trial_count": 18,
                "completed_trials": 18,
                "metrics": {"best_q_value": 0.011, "best_candidate_id": "candidate"}
              }],
              "coverage": [],
              "strategies": [],
              "growth": {
                "as_of": "2026-07-20T20:00:00Z",
                "action": "WAIT",
                "expected_log_growth": 0.08,
                "expected_log_growth_lower_95": 0.01,
                "quarter_kelly_limit": 1.4,
                "effective_leverage": 1.0,
                "constraint_limits": {"quarter_kelly": 1.4, "user_cap": 2.0},
                "binding_constraints": ["funding_freshness"],
                "evidence_state": "PROMOTED"
              }
            }
        """.trimIndent()
        val snapshot = AppJson.decodeFromString(ResearchSnapshotV1.serializer(), payload)
        assertEquals("RUNNING", snapshot.overview?.worker?.state)
        assertEquals(3, snapshot.overview?.proposalCount)
        assertEquals(1, snapshot.overview?.freshProposalCount)
        assertEquals(50_000, snapshot.overview?.evidenceGaps?.single()?.observedObservations)
        assertEquals("WAIT", snapshot.growth?.action)
        assertEquals(18, snapshot.campaigns.single().completedTrials)
        assertTrue(snapshot.campaigns.single().metrics.containsKey("best_q_value"))

        val encoded = AppJson.encodeToString(ResearchSnapshotV1.serializer(), snapshot)
        val decoded = AppJson.decodeFromString(ResearchSnapshotV1.serializer(), encoded)
        assertEquals(snapshot, decoded)
        assertTrue(decoded.growth?.bindingConstraints?.contains("funding_freshness") == true)
    }
}
