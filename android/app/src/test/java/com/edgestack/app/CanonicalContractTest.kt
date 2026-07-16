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
              "disclaimer": "Research and paper-trading output only."
            }
        """.trimIndent()

        val analysis = AppJson.decodeFromString(InstrumentAnalysisV2.serializer(), payload)

        assertEquals("GLD", analysis.resolution.resolvedSymbol)
        assertEquals("physical gold", analysis.resolution.proxyFor)
        assertEquals("NOT_RATED", analysis.overallRating)
        assertTrue(!analysis.alignment.alignedTrade)
    }
}
