package com.edgestack.app.di

import android.content.Context
import com.edgestack.app.core.AppJson
import com.edgestack.app.data.local.JsonFileStore
import com.edgestack.app.data.local.SeedAssets
import com.edgestack.app.data.local.SettingsStore
import com.edgestack.app.data.remote.YahooChartClient
import com.edgestack.app.data.repo.CalendarRepository
import com.edgestack.app.data.repo.EdgesRepository
import com.edgestack.app.data.repo.InstrumentAnalysisRepository
import com.edgestack.app.data.repo.OilDecisionRepository
import com.edgestack.app.data.repo.PositionsRepository
import com.edgestack.app.data.repo.QuoteRepository
import com.edgestack.app.data.repo.RecommendationRepository
import com.edgestack.app.data.repo.ResearchRepository
import com.edgestack.app.data.repo.SniperRepository
import com.edgestack.app.data.repo.SyncRepository
import okhttp3.OkHttpClient
import java.util.concurrent.TimeUnit

/** Manual DI — one instance per process, created in EdgeStackApp. */
class AppContainer(context: Context) {

    val http: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .build()

    val settings = SettingsStore(context)
    private val seed = SeedAssets(context)
    private val fileStore = JsonFileStore(context, AppJson)

    val recommendationRepo = RecommendationRepository(seed, fileStore)
    val instrumentRepo = InstrumentAnalysisRepository(fileStore)
    val oilRepo = OilDecisionRepository(fileStore)
    val sniperRepo = SniperRepository(fileStore)
    val edgesRepo = EdgesRepository(fileStore)
    val researchRepo = ResearchRepository(fileStore)
    val positionsRepo = PositionsRepository(fileStore)
    val calendarRepo = CalendarRepository(seed)
    val syncRepo = SyncRepository(
        settings,
        recommendationRepo,
        instrumentRepo,
        oilRepo,
        sniperRepo,
        edgesRepo,
        researchRepo,
        http,
    )
    val yahoo = YahooChartClient(http)
    val quotes = QuoteRepository(syncRepo, yahoo)
}
