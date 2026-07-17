package com.edgestack.app.domain

import com.edgestack.app.domain.model.MacroEvent
import java.time.LocalDate

/** Date-indexed view over the bundled macro-event seed (FOMC / CPI / EIA). */
class MacroEventBook(events: List<MacroEvent>) {

    private val byDate: Map<LocalDate, List<MacroEvent>> = events
        .mapNotNull { event ->
            runCatching { LocalDate.parse(event.date) to event }.getOrNull()
        }
        .groupBy({ it.first }, { it.second })

    val hasData: Boolean = byDate.isNotEmpty()

    fun on(date: LocalDate): List<MacroEvent> = byDate[date].orEmpty()

    /** Compact per-day marker like "FOMC" or "CPI+EIA"; null when nothing lands. */
    fun marker(date: LocalDate): String? =
        on(date).map { it.type }.distinct().takeIf { it.isNotEmpty() }?.joinToString("+")

    /** High-impact events (FOMC/CPI) on [date]; EIA is routine and excluded. */
    fun highImpactOn(date: LocalDate): List<MacroEvent> =
        on(date).filter { it.type == "FOMC" || it.type == "CPI" }
}
