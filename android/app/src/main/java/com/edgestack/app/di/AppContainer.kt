package com.edgestack.app.di

import android.content.Context
import com.edgestack.app.core.AppJson
import com.edgestack.app.data.local.JsonFileStore
import com.edgestack.app.data.local.SeedAssets
import com.edgestack.app.data.local.SettingsStore
import com.edgestack.app.data.remote.YahooChartClient
import com.edgestack.app.data.repo.BoardRepository
import com.edgestack.app.data.repo.CalendarRepository
import com.edgestack.app.data.repo.EdgesRepository
import com.edgestack.app.data.repo.OverlayRepository
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
    val yahoo = YahooChartClient(http)

    val boardRepo = BoardRepository(seed, fileStore, yahoo)
    val edgesRepo = EdgesRepository(seed, fileStore)
    val overlayRepo = OverlayRepository(yahoo)
    val calendarRepo = CalendarRepository(seed)
    val syncRepo = SyncRepository(settings, boardRepo, edgesRepo, http)
}
