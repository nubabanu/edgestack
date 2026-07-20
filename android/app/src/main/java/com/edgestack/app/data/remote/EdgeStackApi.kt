package com.edgestack.app.data.remote

import com.edgestack.app.core.AppJson
import com.edgestack.app.domain.model.BacktestRunV2
import com.edgestack.app.domain.model.CanonicalRecommendationBundleV2
import com.edgestack.app.domain.model.CampaignSummaryV1
import com.edgestack.app.domain.model.DataCoverageV1
import com.edgestack.app.domain.model.EdgeSummaryV2
import com.edgestack.app.domain.model.InstrumentAnalysisRequestV2
import com.edgestack.app.domain.model.InstrumentAnalysisV2
import com.edgestack.app.domain.model.InstrumentRecheckRequestV2
import com.edgestack.app.domain.model.InstrumentRecheckV2
import com.edgestack.app.domain.model.GrowthDiagnosticsV1
import com.edgestack.app.domain.model.OilDecisionRequestV2
import com.edgestack.app.domain.model.OilDecisionSnapshotV2
import com.edgestack.app.domain.model.PatternLeaderBoardV2
import com.edgestack.app.domain.model.PatternLeaderRequestV2
import com.edgestack.app.domain.model.PaperResponse
import com.edgestack.app.domain.model.PortfolioRecommendationV2
import com.edgestack.app.domain.model.QuoteBatchV1
import com.edgestack.app.domain.model.RecommendationPreviewRequestV2
import com.edgestack.app.domain.model.ResearchOverviewV1
import com.edgestack.app.domain.model.ShadowStrategyV1
import com.edgestack.app.domain.model.SniperPlanV2
import com.edgestack.app.domain.model.SniperPreviewRequestV2
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.kotlinx.serialization.asConverterFactory
import retrofit2.http.GET
import retrofit2.http.Body
import retrofit2.http.POST
import retrofit2.http.Path
import retrofit2.http.Query

@Serializable
data class VersionInfo(
    val version: String = "?",
    @SerialName("config_hash") val configHash: String = "",
    val disclaimer: String = "",
)

/** Read-only client for the PC-hosted FastAPI (src/edgestack/api/app.py). */
interface EdgeStackApi {
    @GET("health") suspend fun health(): Map<String, String>
    @GET("version") suspend fun version(): VersionInfo
    @GET("paper") suspend fun paper(): PaperResponse
    @GET("market/quotes")
    suspend fun marketQuotes(@Query("symbols") symbols: String): QuoteBatchV1
    @GET("recommendations/latest")
    suspend fun latestRecommendation(): CanonicalRecommendationBundleV2
    @GET("research/overview") suspend fun researchOverview(): ResearchOverviewV1
    @GET("research/campaigns") suspend fun researchCampaigns(): List<CampaignSummaryV1>
    @GET("research/campaigns/{id}")
    suspend fun researchCampaign(@Path("id") id: String): CampaignSummaryV1
    @GET("data/coverage") suspend fun dataCoverage(): List<DataCoverageV1>
    @GET("paper/strategies") suspend fun paperStrategies(): List<ShadowStrategyV1>
    @GET("recommendations/growth-diagnostics")
    suspend fun growthDiagnostics(): GrowthDiagnosticsV1
    @POST("recommendations/preview")
    suspend fun previewRecommendation(
        @Body request: RecommendationPreviewRequestV2,
    ): PortfolioRecommendationV2
    @POST("instruments/analyze")
    suspend fun analyzeInstrument(
        @Body request: InstrumentAnalysisRequestV2,
    ): InstrumentAnalysisV2
    @POST("instruments/recheck")
    suspend fun recheckInstrument(
        @Body request: InstrumentRecheckRequestV2,
    ): InstrumentRecheckV2
    @POST("instruments/pattern-leaders")
    suspend fun patternLeaders(
        @Body request: PatternLeaderRequestV2,
    ): PatternLeaderBoardV2
    @POST("oil/decision")
    suspend fun oilDecision(
        @Body request: OilDecisionRequestV2,
    ): OilDecisionSnapshotV2
    @GET("edges") suspend fun edges(): List<EdgeSummaryV2>
    @GET("edges/{id}") suspend fun edgeDetail(@Path("id") id: String): JsonObject
    @GET("monitoring/edges") suspend fun monitoringEdges(): JsonObject
    @GET("backtests") suspend fun backtests(): List<BacktestRunV2>
    @GET("backtests/{runId}") suspend fun backtestDetail(@Path("runId") runId: String): JsonObject
    @GET("sniper/latest")
    suspend fun latestSniper(): SniperPlanV2
    @POST("sniper/preview")
    suspend fun previewSniper(
        @Body request: SniperPreviewRequestV2,
    ): SniperPlanV2

    companion object {
        fun create(baseUrl: String, http: OkHttpClient): EdgeStackApi =
            Retrofit.Builder()
                .baseUrl(if (baseUrl.endsWith("/")) baseUrl else "$baseUrl/")
                .client(http)
                .addConverterFactory(
                    AppJson.asConverterFactory("application/json".toMediaType()))
                .build()
                .create(EdgeStackApi::class.java)
    }
}
