package com.edgestack.app.domain.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Lenient DTOs for the script-owned watcher artifacts served by
 * GET /watchers/tranche and GET /watchers/oil-surge. The artifacts evolve with
 * the research scripts, so every field has a default and unknown keys are
 * ignored (AppJson). Display-only: the device never evaluates a trigger.
 */

@Serializable
data class TriggerStateV1(
    val fired: Boolean = false,
    val detail: String = "",
)

@Serializable
data class TrancheSymbolV1(
    val symbol: String = "?",
    @SerialName("as_of") val asOf: String = "",
    val close: Double? = null,
    @SerialName("T1") val t1: TriggerStateV1? = null,
    @SerialName("T2") val t2: TriggerStateV1? = null,
    @SerialName("T3") val t3: TriggerStateV1? = null,
    @SerialName("REL") val rel: TriggerStateV1? = null,
    @SerialName("CAL") val cal: String? = null,
    @SerialName("window_open") val windowOpen: Boolean = false,
    @SerialName("GO") val go: Int? = null,
    val earnings: String? = null,
) {
    /** (label, state) pairs for display, in watcher order. */
    fun triggers(): List<Pair<String, TriggerStateV1>> = listOfNotNull(
        t1?.let { "T1 dip" to it },
        t2?.let { "T2 repair" to it },
        t3?.let { "T3 trend" to it },
        rel?.let { "REL" to it },
    )
}

@Serializable
data class BreadthV1(
    val count: Int = 0,
    val total: Int = 0,
    val names: String = "",
)

@Serializable
data class PaperTradeV1(
    val symbol: String = "?",
    val trigger: String = "?",
    val eur: Double = 0.0,
    @SerialName("signal_date") val signalDate: String = "",
    @SerialName("fill_price") val fillPrice: Double? = null,
    @SerialName("fill_date") val fillDate: String? = null,
    val shares: Double? = null,
)

@Serializable
data class PaperBookV1(val trades: List<PaperTradeV1> = emptyList())

@Serializable
data class TrancheWatchV1(
    @SerialName("run_date") val runDate: String? = null,
    @SerialName("file_modified_at") val fileModifiedAt: String = "",
    val stale: Boolean = true,
    @SerialName("run_status") val runStatus: String? = null,
    @SerialName("go_alerts_enabled") val goAlertsEnabled: Boolean = false,
    val breadth: BreadthV1? = null,
    val symbols: List<TrancheSymbolV1> = emptyList(),
    @SerialName("paper_book") val paperBook: PaperBookV1? = null,
    val disclaimer: String = "",
)

@Serializable
data class OilEpisodeV1(
    @SerialName("shock_date") val shockDate: String = "",
    @SerialName("shock_kind") val shockKind: String = "",
    @SerialName("shock_ret") val shockRet: Double = 0.0,
    @SerialName("pre_shock_close") val preShockClose: Double? = null,
    @SerialName("post_shock_high") val postShockHigh: Double? = null,
    val sessions: Int = 0,
    val dips: Int = 0,
)

@Serializable
data class OilStateV1(
    val phase: String = "IDLE",
    @SerialName("last_session") val lastSession: String? = null,
    val episode: OilEpisodeV1? = null,
    @SerialName("tickets_enabled") val ticketsEnabled: Boolean = false,
)

@Serializable
data class OilVerdictV1(
    val verdict: String = "",
    @SerialName("rules_passed") val rulesPassed: List<String> = emptyList(),
)

@Serializable
data class LatestCloseV1(val date: String = "", val close: Double = 0.0)

@Serializable
data class OilSurgeV1(
    val state: OilStateV1 = OilStateV1(),
    @SerialName("file_modified_at") val fileModifiedAt: String = "",
    val stale: Boolean = true,
    @SerialName("study_verdict") val studyVerdict: OilVerdictV1? = null,
    @SerialName("dip_tickets_enabled") val dipTicketsEnabled: Boolean = false,
    @SerialName("latest_closes") val latestCloses: Map<String, LatestCloseV1> = emptyMap(),
    @SerialName("paper_book") val paperBook: PaperBookV1? = null,
    val disclaimer: String = "",
)
