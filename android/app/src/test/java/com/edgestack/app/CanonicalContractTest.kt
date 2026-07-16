package com.edgestack.app

import com.edgestack.app.core.AppJson
import com.edgestack.app.domain.model.CanonicalRecommendationBundleV2
import com.edgestack.app.domain.model.InstrumentAnalysisV2
import com.edgestack.app.domain.model.SniperPlanV2
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

    @Test
    fun sniperContractPreservesShadowRolesSizingAndHardExclusions() {
        val payload = """
            {
              "schema_version": 2,
              "generated_at": "2026-07-16T20:00:00Z",
              "session": "2026-07-16",
              "data_version": "data-v2",
              "artifact_version": "artifact-v2",
              "policy_version": "baseline-diversified-v1",
              "account_equity": 100000,
              "max_tolerable_loss": 250,
              "requested_vehicle": "SPY",
              "policy_ranking": [{
                "rank": 1,
                "strategy_id": "C1_C2_PRIMARY",
                "stage": 1,
                "role": "PRIMARY_ENGINE",
                "activation": "SHADOW_READY",
                "conviction": "HIGHEST",
                "rule": "RSI(2) or three down above 200-DMA.",
                "reason": "Requires frozen promotion."
              }],
              "stage_1_candidates": [{
                "strategy_id": "C1_C2_PRIMARY",
                "component_triggers": ["C2_THREE_DOWN"],
                "symbol": "SPY",
                "status": "TRIGGERED_SHADOW",
                "signal_session": "2026-07-16",
                "entry_window": "Next regular-session open",
                "exit_rule": "Close above 5-DMA or fourth close.",
                "maximum_holding_sessions": 4,
                "sizing": {
                  "account_equity": 100000,
                  "max_tolerable_loss": 250,
                  "adverse_move_p05": -0.04,
                  "risk_notional": 6250,
                  "capped_notional": 6250,
                  "portfolio_weight": 0.0625,
                  "estimated_shares": 10,
                  "cap_applied": false,
                  "resolved_signal_outcomes": 40,
                  "estimate_source": "prior signals",
                  "warning": "Tail loss can be worse."
                },
                "evidence": {
                  "observations": 40,
                  "effective_sample_size": 30,
                  "evidence_grade": "INSUFFICIENT",
                  "promoted": false,
                  "warning": "Previously accessed history."
                },
                "actionable": false,
                "paper_only": true
              }],
              "stage_2_candidates": [],
              "overlays": [{
                "strategy_id": "C3_VIX_CONTANGO",
                "state": "UNAVAILABLE",
                "threshold": "positive contango",
                "can_initiate": false,
                "effect": "Veto only."
              }],
              "excluded_strategy_ids": [
                "A3_MINOR_HOLIDAY", "A4_SMALL_CAP_JANUARY", "A5_WEEKEND_MONDAY",
                "B_NAIVE_OVERNIGHT", "D_SHORT_VOL_INCOME"
              ],
              "stage_1_promotion_satisfied": false,
              "warnings": ["No canonical weight."],
              "disclaimer": "Research and paper-trading shadow plan only."
            }
        """.trimIndent()

        val plan = AppJson.decodeFromString(SniperPlanV2.serializer(), payload)

        assertEquals("TRIGGERED_SHADOW", plan.stage1Candidates.single().status)
        assertEquals(6250.0, plan.stage1Candidates.single().sizing!!.cappedNotional, 1e-12)
        assertTrue(!plan.stage1Candidates.single().actionable)
        assertTrue(!plan.overlays.single().canInitiate)
        assertEquals(5, plan.excludedStrategyIds.size)
    }
}
