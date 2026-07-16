package com.edgestack.app

import com.edgestack.app.core.AppJson
import com.edgestack.app.domain.model.CanonicalRecommendationBundleV2
import com.edgestack.app.domain.model.InstrumentAnalysisV2
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CanonicalContractTest {
    private fun seed(): CanonicalRecommendationBundleV2 {
        val text = File("src/main/assets/seed/recommendation.json").readText()
        return AppJson.decodeFromString(CanonicalRecommendationBundleV2.serializer(), text)
    }

    @Test
    fun pythonSeedParsesAndPreservesSizingInvariants() {
        val bundle = seed()
        val base = bundle.baseRecommendation.unleveredBaseWeights.sumOf { it.weight }
        val recommendation = bundle.defaultRecommendation
        val risky = recommendation.personalizedTargetWeights
            .filterNot { it.assetKind == "CASH" }
            .sumOf { kotlin.math.abs(it.weight) }
        val cash = recommendation.personalizedTargetWeights.single { it.assetKind == "CASH" }

        assertEquals(2, bundle.schemaVersion)
        assertEquals(1.0, base, 1e-12)
        assertEquals(recommendation.effectiveLeverage, risky, 1e-12)
        assertEquals(1.0 - recommendation.effectiveLeverage, cash.weight, 1e-12)
        assertTrue(bundle.baseRecommendation.watchlist.all { it.prospectiveSessions < 252 })
    }

    @Test
    fun contractRoundTripsWithoutOnDeviceRecalculation() {
        val original = seed()
        val encoded = AppJson.encodeToString(CanonicalRecommendationBundleV2.serializer(), original)
        val decoded = AppJson.decodeFromString(CanonicalRecommendationBundleV2.serializer(), encoded)
        assertEquals(original, decoded)
    }

    @Test
    fun instrumentAnalysisPreservesServerAbstentionAndProxyDisclosure() {
        val payload = """
            {
              "schema_version": 2,
              "analysis_id": "fixture",
              "resolution": {
                "requested_symbol": "GOLD",
                "resolved_symbol": "GLD",
                "instrument_kind": "COMMODITY_PROXY",
                "proxy_for": "physical gold",
                "notes": ["tracking error applies"]
              },
              "as_of": "2026-07-16T20:00:00Z",
              "data_version": "data-v2",
              "artifact_version": "artifact-v2",
              "policy_version": "baseline-diversified-v1",
              "status": "INSUFFICIENT_EVIDENCE",
              "overall_rating": "NOT_RATED",
              "alignment": {
                "aligned_trade": false,
                "promoted_tailwinds": 0,
                "promoted_headwinds": 0,
                "observational_tailwinds": 2,
                "observational_headwinds": 1,
                "missing_inputs": ["day", "week", "month", "year"],
                "explanation": "No promoted timing rule."
              },
              "horizon_analyses": [],
              "chosen_time_ratings": [{
                "resolution": "MINUTE_15",
                "horizon": "DAY",
                "requested_time": "2026-07-20T09:30:00-04:00",
                "matched_slot": "09:30",
                "rating": "ABOVE_AVERAGE",
                "score": {
                  "net_win_rate": 0.56,
                  "shrunk_win_rate": 0.55,
                  "win_score": 53.2,
                  "expected_net_return": 0.001,
                  "lower_95": -0.002,
                  "observations": 42,
                  "effective_sample_size": 31.0,
                  "rank": 2,
                  "candidates_ranked": 26,
                  "multiple_testing_adjusted_pvalue": 0.8,
                  "evidence_grade": "INSUFFICIENT"
                },
                "recommendation": "Research only."
              }],
              "exit_plans": [{
                "horizon": "DAY",
                "entry_slot": "09:30",
                "preferred_exit": "15:45 New York",
                "holding_sessions": 0,
                "data_resolution": "MINUTE_15",
                "rationale": "Historical conditional exit."
              }],
              "tailwind_calendars": [{
                "resolution": "MINUTE_15",
                "timezone": "America/New_York",
                "horizon": "DAY",
                "cells": [],
                "warning": "Research only."
              }],
              "recheck_plan": {
                "enabled": true,
                "intended_entry_at": "2026-07-20T09:30:00-04:00",
                "next_check_at": "2026-07-16T21:00:00Z",
                "cadence_minutes": 60,
                "required_resolution": "HOUR",
                "reason": "Recheck hourly."
              },
              "disclaimer": "Research and paper-trading output only."
            }
        """.trimIndent()

        val analysis = AppJson.decodeFromString(InstrumentAnalysisV2.serializer(), payload)

        assertEquals("GLD", analysis.resolution.resolvedSymbol)
        assertEquals("physical gold", analysis.resolution.proxyFor)
        assertEquals("NOT_RATED", analysis.overallRating)
        assertTrue(!analysis.alignment.alignedTrade)
        assertEquals(53.2, analysis.chosenTimeRatings.single().score!!.winScore, 1e-12)
        assertEquals("15:45 New York", analysis.exitPlans.single().preferredExit)
        assertEquals(60, analysis.recheckPlan.cadenceMinutes)
    }
}
