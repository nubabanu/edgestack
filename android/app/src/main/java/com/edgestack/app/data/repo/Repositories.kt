package com.edgestack.app.data.repo

import com.edgestack.app.data.local.JsonFileStore
import com.edgestack.app.data.local.SeedAssets
import com.edgestack.app.data.local.SettingsStore
import com.edgestack.app.data.remote.EdgeStackApi
import com.edgestack.app.data.remote.YahooChartClient
import com.edgestack.app.domain.EnsembleCalculator
import com.edgestack.app.domain.OverlayCalculator
import com.edgestack.app.domain.TradingCalendar
import com.edgestack.app.domain.model.Board
import com.edgestack.app.domain.model.Edge
import com.edgestack.app.domain.model.EdgesBundle
import com.edgestack.app.domain.model.OverlayState
import com.edgestack.app.domain.model.PaperResponse
import com.edgestack.app.domain.model.PicksBundle
import com.edgestack.app.domain.model.TrackedPosition
import com.edgestack.app.domain.model.TrackedPositions
import com.edgestack.app.domain.model.SpyBar
import okhttp3.OkHttpClient
import java.time.LocalDate

/** Read order everywhere: file cache -> bundled seed. Sync overwrites the cache. */
class BoardRepository(
    private val seed: SeedAssets,
    private val store: JsonFileStore,
    private val yahoo: YahooChartClient,
) {
    fun load(): Board =
        store.read("board.json", Board.serializer()) ?: seed.board()

    fun save(board: Board) = store.write("board.json", Board.serializer(), board)

    fun loadPicks(): PicksBundle? =
        store.read("picks.json", PicksBundle.serializer()) ?: seed.picks()

    fun savePicks(picks: PicksBundle) =
        store.write("picks.json", PicksBundle.serializer(), picks)

    suspend fun liveQuotes(board: Board): Map<String, Double> =
        yahoo.latestQuotes(board.rows.map { it.symbol })
}

class EdgesRepository(
    private val seed: SeedAssets,
    private val store: JsonFileStore,
) {
    fun load(): EdgesBundle =
        store.read("edges.json", EdgesBundle.serializer()) ?: seed.edges()

    fun save(bundle: EdgesBundle) =
        store.write("edges.json", EdgesBundle.serializer(), bundle)
}

class OverlayRepository(
    private val yahoo: YahooChartClient,
    private val calendar: TradingCalendar,
) {

    private val cache = java.util.concurrent.ConcurrentHashMap<String, List<SpyBar>>()

    suspend fun bars(symbol: String = "SPY",
                     forceRefresh: Boolean = false): List<SpyBar> {
        val cached = cache[symbol].orEmpty()
        if (cached.isEmpty() || forceRefresh ||
            cached.last().date < LocalDate.now().minusDays(1)
        ) {
            runCatching { yahoo.dailyHistory(symbol, years = 3) }
                .onSuccess {
                    if (it.size > OverlayCalculator.WARMUP) cache[symbol] = it
                }
        }
        return cache[symbol].orEmpty()
    }

    suspend fun state(base: Double, forceRefresh: Boolean = false): OverlayState? =
        OverlayCalculator.state(bars("SPY", forceRefresh), base = base,
                                calendar = calendar)

    suspend fun ensembleState(symbol: String,
                              forceRefresh: Boolean = false): EnsembleCalculator.State? =
        EnsembleCalculator.state(bars(symbol, forceRefresh))
}

class PositionsRepository(private val store: JsonFileStore) {

    fun load(): List<TrackedPosition> =
        store.read("positions.json", TrackedPositions.serializer())?.positions
            .orEmpty()

    fun save(positions: List<TrackedPosition>) =
        store.write("positions.json", TrackedPositions.serializer(),
                    TrackedPositions(positions))

    fun add(p: TrackedPosition) = save(load() + p)

    fun remove(symbol: String, entryDate: String) =
        save(load().filterNot { it.symbol == symbol && it.entryDate == entryDate })
}

class SyncRepository(
    private val settingsStore: SettingsStore,
    private val boardRepo: BoardRepository,
    private val edgesRepo: EdgesRepository,
    private val http: OkHttpClient,
) {
    private suspend fun api(): EdgeStackApi? {
        val url = settingsStore.current().baseUrl
        if (url.isBlank()) return null
        return EdgeStackApi.create(url, http)
    }

    suspend fun paper(): Result<PaperResponse> = runCatching {
        val api = api() ?: error("no server URL configured")
        api.paper()
    }

    suspend fun testConnection(): Result<String> = runCatching {
        val api = api() ?: error("no server URL configured")
        api.health()
        val v = api.version()
        "EdgeStack API ${v.version} (${v.configHash.take(8)})"
    }

    /** Best-effort sync; returns a short human status line. */
    suspend fun syncAll(): Result<String> = runCatching {
        val api = api() ?: error("no server URL configured")
        var boardMsg = "board: skipped"
        runCatching { api.board() }.onSuccess {
            boardRepo.save(it)
            boardMsg = "board: ${it.asOf}"
        }
        runCatching { api.picks() }.onSuccess { boardRepo.savePicks(it) }
        val edges: List<Edge> = api.edges()
        edgesRepo.save(EdgesBundle(edges = edges))
        settingsStore.stampSync(System.currentTimeMillis())
        "$boardMsg, ${edges.size} edges"
    }
}

class CalendarRepository(private val seed: SeedAssets) {
    val calendar: TradingCalendar by lazy {
        TradingCalendar(seed.calendar().sessions.map { LocalDate.parse(it) })
    }
}
