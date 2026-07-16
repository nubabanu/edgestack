package com.edgestack.app.data.repo

import com.edgestack.app.data.local.JsonFileStore
import com.edgestack.app.data.local.SeedAssets
import com.edgestack.app.data.local.SettingsStore
import com.edgestack.app.data.remote.EdgeStackApi
import com.edgestack.app.domain.TradingCalendar
import com.edgestack.app.domain.model.BacktestRunV2
import com.edgestack.app.domain.model.CanonicalRecommendationBundleV2
import com.edgestack.app.domain.model.EdgeCatalog
import com.edgestack.app.domain.model.EdgeSummaryV2
import com.edgestack.app.domain.model.InstrumentAnalysisRequestV2
import com.edgestack.app.domain.model.InstrumentAnalysisV2
import com.edgestack.app.domain.model.InstrumentRecheckRequestV2
import com.edgestack.app.domain.model.InstrumentRecheckV2
import com.edgestack.app.domain.model.PatternLeaderBoardV2
import com.edgestack.app.domain.model.PatternLeaderRequestV2
import com.edgestack.app.domain.model.PaperResponse
import com.edgestack.app.domain.model.PortfolioRecommendationV2
import com.edgestack.app.domain.model.RecommendationPreviewRequestV2
import com.edgestack.app.domain.model.SniperPlanV2
import com.edgestack.app.domain.model.SniperPreviewRequestV2
import com.edgestack.app.domain.model.TrackedPosition
import com.edgestack.app.domain.model.TrackedPositions
import kotlinx.serialization.json.JsonObject
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

    fun loadRequest(): InstrumentAnalysisRequestV2? =
        store.read("instrument_request.json", InstrumentAnalysisRequestV2.serializer())

    fun save(analysis: InstrumentAnalysisV2, request: InstrumentAnalysisRequestV2? = null) {
        store.write(
            "instrument_analysis.json",
            InstrumentAnalysisV2.serializer(),
            analysis,
        )
        if (request != null) {
            store.write(
                "instrument_request.json",
                InstrumentAnalysisRequestV2.serializer(),
                request,
            )
        }
    }
}

/** Last server-fetched edge catalog and monitoring snapshot for offline display. */
class EdgesRepository(private val store: JsonFileStore) {
    fun loadCatalog(): List<EdgeSummaryV2> =
        store.read("edges.json", EdgeCatalog.serializer())?.edges.orEmpty()

    fun saveCatalog(edges: List<EdgeSummaryV2>) =
        store.write("edges.json", EdgeCatalog.serializer(), EdgeCatalog(edges))

    fun loadMonitoring(): JsonObject? =
        store.read("edges_monitoring.json", JsonObject.serializer())

    fun saveMonitoring(payload: JsonObject) =
        store.write("edges_monitoring.json", JsonObject.serializer(), payload)
}

/** Last server-evaluated sniper plan. Offline mode displays it without recalculating signals. */
class SniperRepository(private val store: JsonFileStore) {
    fun load(): SniperPlanV2? =
        store.read("sniper_plan.json", SniperPlanV2.serializer())

    fun save(plan: SniperPlanV2) =
        store.write("sniper_plan.json", SniperPlanV2.serializer(), plan)
}

class SyncRepository(
    private val settingsStore: SettingsStore,
    private val recommendationRepo: RecommendationRepository,
    private val instrumentRepo: InstrumentAnalysisRepository,
    private val sniperRepo: SniperRepository,
    private val edgesRepo: EdgesRepository,
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
        val request = InstrumentAnalysisRequestV2(
            symbol = symbol.trim().uppercase(),
            intendedEntryAt = intendedEntryAt?.takeUnless { it.matches(Regex("\\d{4}-\\d{2}-\\d{2}")) },
            intendedEntryDate = intendedEntryAt?.takeIf { it.matches(Regex("\\d{4}-\\d{2}-\\d{2}")) },
            roundTripCostBps = roundTripCostBps,
        )
        val result = api.analyzeInstrument(request)
        instrumentRepo.save(result, request)
        result
    }

    suspend fun recheckInstrument(): Result<InstrumentRecheckV2> = runCatching {
        val api = api() ?: error("no server URL configured")
        val previous = instrumentRepo.loadLast() ?: error("no cached instrument analysis")
        val request = instrumentRepo.loadRequest() ?: error("no cached instrument request")
        val result = api.recheckInstrument(
            InstrumentRecheckRequestV2(previousAnalysis = previous, request = request),
        )
        instrumentRepo.save(result.analysis, request)
        result
    }

    suspend fun patternLeaders(
        symbols: List<String>,
        resolution: String = "DAY",
        horizon: String = "WEEK",
    ): Result<PatternLeaderBoardV2> = runCatching {
        val api = api() ?: error("no server URL configured")
        api.patternLeaders(
            PatternLeaderRequestV2(
                symbols = symbols,
                resolution = resolution,
                horizon = horizon,
            ),
        )
    }

    suspend fun edges(): Result<List<EdgeSummaryV2>> = runCatching {
        val api = api() ?: error("no server URL configured")
        api.edges().also(edgesRepo::saveCatalog)
    }

    suspend fun monitoringEdges(): Result<JsonObject> = runCatching {
        val api = api() ?: error("no server URL configured")
        api.monitoringEdges().also(edgesRepo::saveMonitoring)
    }

    suspend fun backtests(): Result<List<BacktestRunV2>> = runCatching {
        val api = api() ?: error("no server URL configured")
        api.backtests()
    }

    suspend fun latestSniper(): Result<SniperPlanV2> = runCatching {
        val api = api() ?: error("no server URL configured")
        val plan = api.latestSniper()
        sniperRepo.save(plan)
        plan
    }

    suspend fun previewSniper(
        accountEquity: Double,
        maxTolerableLoss: Double,
        vehicle: String,
    ): Result<SniperPlanV2> = runCatching {
        require(accountEquity > 0) { "account equity must be positive" }
        require(maxTolerableLoss > 0 && maxTolerableLoss <= accountEquity) {
            "max tolerable loss must be positive and no greater than equity"
        }
        val api = api() ?: error("no server URL configured")
        val plan = api.previewSniper(
            SniperPreviewRequestV2(
                accountEquity = accountEquity,
                maxTolerableLoss = maxTolerableLoss,
                vehicle = vehicle.trim().uppercase(),
            ),
        )
        sniperRepo.save(plan)
        plan
    }

    suspend fun syncAll(): Result<String> = runCatching {
        val api = api() ?: error("no server URL configured")
        val bundle = api.latestRecommendation()
        recommendationRepo.saveBundle(bundle)
        runCatching { api.latestSniper() }.onSuccess(sniperRepo::save)
        runCatching { api.edges() }.onSuccess(edgesRepo::saveCatalog)
        runCatching { api.monitoringEdges() }.onSuccess(edgesRepo::saveMonitoring)
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
