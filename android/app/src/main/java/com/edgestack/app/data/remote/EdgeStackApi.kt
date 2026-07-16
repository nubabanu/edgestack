package com.edgestack.app.data.remote

import com.edgestack.app.core.AppJson
import com.edgestack.app.domain.model.Board
import com.edgestack.app.domain.model.Edge
import com.edgestack.app.domain.model.PaperResponse
import com.edgestack.app.domain.model.PicksBundle
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.kotlinx.serialization.asConverterFactory
import retrofit2.http.GET
import retrofit2.http.Path

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
    @GET("board") suspend fun board(): Board
    @GET("picks") suspend fun picks(): PicksBundle
    @GET("paper") suspend fun paper(): PaperResponse
    @GET("edges") suspend fun edges(): List<Edge>
    @GET("edges/{id}") suspend fun edgeDetail(@Path("id") id: String): Edge

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
