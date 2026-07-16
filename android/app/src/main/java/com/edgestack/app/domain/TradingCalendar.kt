package com.edgestack.app.domain

import java.time.LocalDate
import java.time.YearMonth

/**
 * NYSE trading calendar backed by the bundled session list
 * (assets/seed/calendar.json, exported from exchange_calendars XNYS).
 */
class TradingCalendar(sessionDates: List<LocalDate>) {

    private val sessions: List<LocalDate> = sessionDates.sorted()
    private val index: Map<LocalDate, Int> =
        sessions.withIndex().associate { (i, d) -> d to i }

    val first: LocalDate get() = sessions.first()
    val last: LocalDate get() = sessions.last()

    fun isSession(d: LocalDate): Boolean = d in index

    /** Next session strictly after [d]; null past calendar end. */
    fun nextSession(d: LocalDate): LocalDate? {
        val pos = sessions.binarySearch(d)
        val next = if (pos >= 0) pos + 1 else -(pos + 1)
        return sessions.getOrNull(next)
    }

    /** Latest session on or before [d]; null before calendar start. */
    fun sessionOnOrBefore(d: LocalDate): LocalDate? {
        val pos = sessions.binarySearch(d)
        val at = if (pos >= 0) pos else -(pos + 1) - 1
        return sessions.getOrNull(at)
    }

    private fun monthSessions(ym: YearMonth): List<LocalDate> =
        sessions.filter { YearMonth.from(it) == ym }

    /** 1-based trading day of month; null if [d] is not a session. */
    fun tradingDayOfMonth(d: LocalDate): Int? {
        if (d !in index) return null
        return monthSessions(YearMonth.from(d)).indexOf(d) + 1
    }

    /** 1-based position counted from month end (1 = last session). */
    fun tradingDayFromMonthEnd(d: LocalDate): Int? {
        if (d !in index) return null
        val m = monthSessions(YearMonth.from(d))
        return m.size - m.indexOf(d)
    }

    fun nthSessionOfMonth(ym: YearMonth, n: Int): LocalDate? =
        monthSessions(ym).getOrNull(n - 1)

    fun lastSessionOfMonth(ym: YearMonth): LocalDate? =
        monthSessions(ym).lastOrNull()

    fun isLastSessionOfMonth(d: LocalDate): Boolean =
        d in index && lastSessionOfMonth(YearMonth.from(d)) == d

    /** Turn-of-month window: last session of a month or first 3 of the next. */
    fun isTurnOfMonthWindow(d: LocalDate): Boolean {
        val td = tradingDayOfMonth(d) ?: return false
        return td <= 3 || tradingDayFromMonthEnd(d) == 1
    }
}
