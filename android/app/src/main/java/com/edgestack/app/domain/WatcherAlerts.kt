package com.edgestack.app.domain

import com.edgestack.app.domain.model.OilSurgeV1

/**
 * Pure diff between two synced oil-surge snapshots -> (title, text) alerts.
 * First sync (before == null) stays silent; states come from the server only.
 */
object WatcherAlerts {

    private const val WATCHING = "WATCHING"

    fun oilAlerts(before: OilSurgeV1?, after: OilSurgeV1): List<Pair<String, String>> {
        if (before == null) return emptyList()
        val alerts = mutableListOf<Pair<String, String>>()
        val episode = after.state.episode
        if (after.state.phase == WATCHING && before.state.phase != WATCHING) {
            alerts += "Oil shock: dip watch armed" to (
                "Shock ${episode?.shockDate ?: "?"} (${pct(episode?.shockRet)}); watching for " +
                    "pullbacks from the post-shock high. Display-only research."
                )
        }
        val beforeDips = before.state.episode?.dips ?: 0
        val afterDips = episode?.dips ?: 0
        if (after.state.phase == WATCHING && afterDips > beforeDips) {
            alerts += "Oil dip in surge regime" to (
                "Dip ${afterDips} since shock ${episode?.shockDate ?: "?"} " +
                    "(session ${episode?.sessions ?: "?"}/10). " +
                    if (after.dipTicketsEnabled) "Paper ticket created." else
                        "DISPLAY-ONLY: entry rule not validated (study verdict " +
                            "${after.studyVerdict?.verdict ?: "not run"})."
                )
        }
        return alerts
    }

    private fun pct(value: Double?): String =
        if (value == null) "?" else "%+.1f%%".format(value * 100)
}
