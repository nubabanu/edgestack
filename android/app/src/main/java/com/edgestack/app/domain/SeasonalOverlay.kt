package com.edgestack.app.domain

import com.edgestack.app.domain.model.TailwindCalendarCellV2
import com.edgestack.app.domain.model.TailwindCalendarV2
import java.time.LocalDate

/** Composite historical tailwind for one concrete date, built from server evidence only. */
data class DayEdge(
    /** 0..1 mean rank percentile of the matched slots; 1 = strongest history. */
    val percentile: Double,
    /** resolution label → matched server cell, in DAY/MONTH/YEAR order. */
    val components: List<Pair<String, TailwindCalendarCellV2>>,
)

/**
 * Projects server tailwind-calendar slots onto concrete dates.
 * Slot-key contract (server trade_calendar._choice_key):
 * DAY → weekday "0".."6" (Monday = 0); MONTH → week-of-month bucket
 * "0".."4" (= min(4, (day-1)/5)); YEAR → month "1".."12".
 */
class SeasonalOverlay(calendars: List<TailwindCalendarV2>) {

    private val byResolution: Map<String, Map<String, TailwindCalendarCellV2>> =
        calendars.associate { cal ->
            cal.resolution.uppercase() to cal.cells.associateBy { it.slotKey }
        }

    val hasData: Boolean = byResolution.values.any { it.isNotEmpty() }

    fun edgeFor(date: LocalDate): DayEdge? {
        val components = buildList {
            byResolution["DAY"]?.get((date.dayOfWeek.value - 1).toString())
                ?.let { add("Weekday" to it) }
            byResolution["MONTH"]?.get(minOf(4, (date.dayOfMonth - 1) / 5).toString())
                ?.let { add("Week of month" to it) }
            byResolution["YEAR"]?.get(date.monthValue.toString())
                ?.let { add("Month" to it) }
        }
        if (components.isEmpty()) return null
        return DayEdge(
            percentile = components.map { it.second.rankPercentile }.average(),
            components = components,
        )
    }
}
