package com.edgestack.app.domain.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import java.time.LocalDate

/** One daily OHLC bar; [adj] is the dividend/split-adjusted close. */
data class SpyBar(
    val date: LocalDate,
    val open: Double,
    val high: Double,
    val low: Double,
    val close: Double,
    val adj: Double,
)

@Serializable
data class BoardRegime(
    @SerialName("bench_trend_200") val benchTrend200: Double? = null,
    @SerialName("bench_vol_20") val benchVol20: Double? = null,
    val trend: String = "?",
    val vol: String = "?",
)

@Serializable
data class BoardRow(
    val symbol: String,
    val close: Double,
    val conviction: Double,
    @SerialName("e_net_10d") val eNet10d: Double,
    val hit: Double,
    @SerialName("n_edges") val nEdges: Int,
    val families: Int,
    @SerialName("top_edges") val topEdges: String = "",
    val stop: Double,
    val target: Double,
)

@Serializable
data class Board(
    @SerialName("schema_version") val schemaVersion: Int = 1,
    @SerialName("as_of") val asOf: String,
    @SerialName("generated_at") val generatedAt: String = "",
    val disclaimer: String = "",
    val regime: BoardRegime = BoardRegime(),
    val rows: List<BoardRow> = emptyList(),
)

/** Tolerant edge model: known fields typed, everything else stays in raw JSON. */
@Serializable
data class EdgeIdentity(
    @SerialName("edge_id") val edgeId: String,
    val name: String = "",
    val family: String = "",
    val direction: String = "",
    @SerialName("holding_horizon") val holdingHorizon: Int = 0,
)

@Serializable
data class EdgeStats(
    @SerialName("net_mean_return") val netMeanReturn: Double? = null,
    @SerialName("q_value") val qValue: Double? = null,
    @SerialName("deflated_sharpe_ratio") val deflatedSharpe: Double? = null,
    @SerialName("sample_size") val sampleSize: Int? = null,
    @SerialName("probability_of_positive_net_return") val probPositive: Double? = null,
)

@Serializable
data class Edge(
    val identity: EdgeIdentity,
    val stats: EdgeStats = EdgeStats(),
    @SerialName("current_status") val currentStatus: String? = null,
)

@Serializable
data class EdgesBundle(
    @SerialName("schema_version") val schemaVersion: Int = 1,
    @SerialName("generated_at") val generatedAt: String = "",
    val edges: List<Edge> = emptyList(),
)

@Serializable
data class CalendarBundle(
    @SerialName("schema_version") val schemaVersion: Int = 1,
    val exchange: String = "XNYS",
    val sessions: List<String> = emptyList(),
)

@Serializable
data class Pick(
    val horizon: String,
    val symbol: String,
    val name: String = "",
    val close: Double? = null,
    val buy: String = "",
    val sell: String = "",
    val stop: Double? = null,
    val target: Double? = null,
    val rationale: String = "",
    val validated: Boolean = false,
)

@Serializable
data class PicksBundle(
    @SerialName("schema_version") val schemaVersion: Int = 1,
    @SerialName("as_of") val asOf: String = "",
    val disclaimer: String = "",
    val picks: List<Pick?> = emptyList(),
)

/** A user-entered real position tracked by the app. */
@Serializable
data class TrackedPosition(
    val symbol: String,
    val entryPrice: Double,
    val quantity: Double,
    val entryDate: String,           // ISO yyyy-MM-dd
    val stop: Double? = null,
    val target: Double? = null,
    val horizonSessions: Int = 10,
)

@Serializable
data class TrackedPositions(val positions: List<TrackedPosition> = emptyList())

// --- paper account (GET /paper on the PC) ---
@Serializable
data class PaperPositionDto(
    val symbol: String,
    val quantity: Double = 0.0,
    @SerialName("entry_price") val entryPrice: Double = 0.0,
    @SerialName("entry_session") val entrySession: String = "",
    @SerialName("stop_price") val stopPrice: Double = 0.0,
    @SerialName("target_price") val targetPrice: Double = 0.0,
)

@Serializable
data class PaperTradeDto(
    val symbol: String,
    @SerialName("net_pnl") val netPnl: Double = 0.0,
    @SerialName("exit_reason") val exitReason: String = "",
    @SerialName("exit_session") val exitSession: String = "",
    @SerialName("realized_net_return") val realizedNetReturn: Double = 0.0,
)

@Serializable
data class PaperStateDto(
    val cash: Double = 0.0,
    @SerialName("last_session") val lastSession: String? = null,
    val positions: List<PaperPositionDto> = emptyList(),
    val trades: List<PaperTradeDto> = emptyList(),
)

@Serializable
data class EquityPoint(val date: String, val equity: Double)

@Serializable
data class PaperResponse(
    val state: PaperStateDto = PaperStateDto(),
    @SerialName("equity_history") val equityHistory: List<EquityPoint> = emptyList(),
)

/** Result of the on-device overlay computation for one session. */
data class OverlayState(
    val date: LocalDate,
    val appliedL: Double,
    val firedRules: List<String>,
    val spyClose: Double,
    val sma200: Double?,
    val vol20: Double?,
    val lastReturn: Double?,
    val history: List<Pair<LocalDate, Double>>,
)

enum class QuoteStatus { STOPPED, TARGET_HIT, OPEN }
