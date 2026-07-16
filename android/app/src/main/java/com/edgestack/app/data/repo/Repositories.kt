package com.edgestack.app.data.repo

import com.edgestack.app.data.local.JsonFileStore
import com.edgestack.app.data.local.SeedAssets
import com.edgestack.app.data.local.SettingsStore
import com.edgestack.app.data.remote.EdgeStackApi
import com.edgestack.app.domain.TradingCalendar
import com.edgestack.app.domain.model.CanonicalRecommendationBundleV2
import com.edgestack.app.domain.model.InstrumentAnalysisRequestV2
import com.edgestack.app.domain.model.InstrumentAnalysisV2
import com.edgestack.app.domain.model.PaperResponse
import com.edgestack.app.domain.model.PortfolioRecommendationV2
import com.edgestack.app.domain.model.RecommendationPreviewRequestV2
import com.edgestack.app.domain.model.TrackedPosition
import com.edgestack.app.domain.model.TrackedPositions
import okhttp3.OkHttpClient
import java.time.LocalDate

/** Offline reads show the last server result (or canonical seed) without recomputation. */
class RecommendationRepository(
    private val seed: SeedAssets,
    private val store: JsonFileStore,
) {
    fun loadBundle(): CanonicalRecommendationBundleV2 =
        store.read(
            "recommendation.json",
            CanonicalRecommendationBundleV2.serializer(),
        ) ?: seed.recommendation()

    fun saveBundle(bundle: CanonicalRecommendationBundleV2) =
        store.write(
            "recommendation.json",
            CanonicalRecommendationBundleV2.serializer(),
            bundle,
        )

    fun loadPreview(): PortfolioRecommendationV2? =
        store.read("recommendation_preview.json", PortfolioRecommendationV2.serializer())

    fun savePreview(preview: PortfolioRecommendationV2) =
        store.write(
            "recommendation_preview.json",
            PortfolioRecommendationV2.serializer(),
            preview,
        )

    fun clearPreview() = store.delete("recommendation_preview.json")

    fun displayedRecommendation(): PortfolioRecommendationV2 =
        loadPreview() ?: loadBundle().defaultRecommendation
}

class PositionsRepository(private val store: JsonFileStore) {
    fun load(): List<TrackedPosition> =
        store.read("positions.json", TrackedPositions.serializer())?.positions.orEmpty()

    fun save(positions: List<TrackedPosition>) =
        store.write("positions.json", TrackedPositions.serializer(), TrackedPositions(positions))

    fun add(position: TrackedPosition) = save(load() + position)

    fun remove(symbol: String, entryDate: String) =
        save(load().filterNot { it.symbol == symbol && it.entryDate == entryDate })
}

class InstrumentAnalysisRepository(private val store: JsonFileStore) {
    fun loadLast(): InstrumentAnalysisV2? =
        store.read("instrument_analysis.json", InstrumentAnalysisV2.serializer())

    fun save(analysis: InstrumentAnalysisV2) =
        store.write(
            "instrument_analysis.json",
            InstrumentAnalysisV2.serializer(),
            analysis,
        )
}

class SyncRepository(
    private val settingsStore: SettingsStore,
    private val recommendationRepo: RecommendationRepository,
    private val instrumentRepo: InstrumentAnalysisRepository,
    private val http: OkHttpClient,
) {
    private suspend fun api(): EdgeStackApi? {
        val url = settingsStore.current().baseUrl
        if (url.isBlank()) return null
        return EdgeStackApi.create(url, http)
    }

    suspend fun paper(): Result<PaperResponse> = runCatching {
        (api() ?: error("no server URL configured")).paper()
    }

    suspend fun testConnection(): Result<String> = runCatching {
        val api = api() ?: error("no server URL configured")
        api.health()
        val version = api.version()
        "EdgeStack API ${version.version} (${version.configHash.take(8)})"
    }

    suspend fun preview(): Result<PortfolioRecommendationV2> = runCatching {
        val api = api() ?: error("no server URL configured")
        val settings = settingsStore.current()
        val bundle = recommendationRepo.loadBundle()
        val previousState = settingsStore.decodedRiskState(settings)
            ?: bundle.defaultRecommendation.outputRiskState
        val profile = settings.profile(previousState.currentEquity)
        val result = api.previewRecommendation(
            RecommendationPreviewRequestV2(profile = profile, riskState = previousState),
        )
        recommendationRepo.savePreview(result)
        settingsStore.saveCanonicalState(
            result.status,
            fingerprint(result),
            result.freshness.isFresh,
            result.outputRiskState,
        )
        result
    }

    suspend fun analyzeInstrument(
        symbol: String,
        intendedEntryAt: String? = null,
        roundTripCostBps: Double = 10.0,
    ): Result<InstrumentAnalysisV2> = runCatching {
        val api = api() ?: error("no server URL configured")
        val result = api.analyzeInstrument(
            InstrumentAnalysisRequestV2(
                symbol = symbol.trim().uppercase(),
                intendedEntryAt = intendedEntryAt,
                roundTripCostBps = roundTripCostBps,
            ),
        )
        instrumentRepo.save(result)
        result
    }

    suspend fun syncAll(): Result<String> = runCatching {
        val api = api() ?: error("no server URL configured")
        val bundle = api.latestRecommendation()
        recommendationRepo.saveBundle(bundle)
        val previewResult = preview()
        val preview = previewResult.getOrElse {
            recommendationRepo.clearPreview()
            val fallback = bundle.defaultRecommendation
            settingsStore.saveCanonicalState(
                fallback.status,
                fingerprint(fallback),
                fallback.freshness.isFresh,
                fallback.outputRiskState,
            )
            fallback
        }
        settingsStore.stampSync(System.currentTimeMillis())
        "canonical ${bundle.session}: ${preview.status}, ${"%.2f".format(preview.effectiveLeverage)}x"
    }

    private fun fingerprint(recommendation: PortfolioRecommendationV2): String = listOf(
        recommendation.status,
        recommendation.effectiveLeverage.toString(),
        recommendation.personalizedTargetWeights.joinToString { "${it.symbol}:${it.weight}" },
        recommendation.freshness.isFresh.toString(),
        recommendation.outputRiskState.drawdownState,
    ).joinToString("|")
}

class CalendarRepository(private val seed: SeedAssets) {
    val calendar: TradingCalendar by lazy {
        TradingCalendar(seed.calendar().sessions.map { LocalDate.parse(it) })
    }
}
