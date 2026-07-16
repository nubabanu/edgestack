package com.edgestack.app.data.remote

import com.edgestack.app.core.AppJson
import com.edgestack.app.domain.model.CanonicalRecommendationBundleV2
import com.edgestack.app.domain.model.PaperResponse
import com.edgestack.app.domain.model.PortfolioRecommendationV2
import com.edgestack.app.domain.model.RecommendationPreviewRequestV2
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.kotlinx.serialization.asConverterFactory
import retrofit2.http.GET
import retrofit2.http.Body
import retrofit2.http.POST

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
    @GET("recommendations/latest")
    suspend fun latestRecommendation(): CanonicalRecommendationBundleV2
    @POST("recommendations/preview")
    suspend fun previewRecommendation(
        @Body request: RecommendationPreviewRequestV2,
    ): PortfolioRecommendationV2

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
