package com.edgestack.app

import com.edgestack.app.domain.TradingCalendar
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate
import java.time.YearMonth

class TradingCalendarTest {

    private val calendar: TradingCalendar by lazy {
        val text = requireNotNull(javaClass.getResourceAsStream("/fixtures/calendar.json")) {
            "missing fixture calendar.json — copy from assets/seed/"
        }.bufferedReader().readText()
        val sessions = Json.parseToJsonElement(text)
            .jsonObject["sessions"]!!.jsonArray.map { LocalDate.parse(it.jsonPrimitive.content) }
        TradingCalendar(sessions)
    }

    @Test
    fun knownSessionsAndHolidays() {
        assertTrue(calendar.isSession(LocalDate.of(2026, 7, 15)))
        assertFalse(calendar.isSession(LocalDate.of(2026, 7, 4)))   // Saturday
        assertFalse(calendar.isSession(LocalDate.of(2026, 7, 3)))   // July 4th observed
        assertFalse(calendar.isSession(LocalDate.of(2026, 11, 26))) // Thanksgiving
        assertFalse(calendar.isSession(LocalDate.of(2026, 12, 25))) // Christmas
        assertFalse(calendar.isSession(LocalDate.of(2026, 9, 7)))   // Labor Day
    }

    @Test
    fun seventhTradingDayOfNovember2026IsNov10() {
        assertEquals(LocalDate.of(2026, 11, 10),
                     calendar.nthSessionOfMonth(YearMonth.of(2026, 11), 7))
        assertEquals(7, calendar.tradingDayOfMonth(LocalDate.of(2026, 11, 10)))
    }

    @Test
    fun lastSessionOfJanuary2027() {
        assertEquals(LocalDate.of(2027, 1, 29),
                     calendar.lastSessionOfMonth(YearMonth.of(2027, 1)))
        assertTrue(calendar.isLastSessionOfMonth(LocalDate.of(2027, 1, 29)))
    }

    @Test
    fun turnOfMonthWindow() {
        // Last session of July 2026 (Fri Jul 31) + first three of August
        assertTrue(calendar.isTurnOfMonthWindow(LocalDate.of(2026, 7, 31)))
        assertTrue(calendar.isTurnOfMonthWindow(LocalDate.of(2026, 8, 3)))
        assertTrue(calendar.isTurnOfMonthWindow(LocalDate.of(2026, 8, 5)))
        assertFalse(calendar.isTurnOfMonthWindow(LocalDate.of(2026, 8, 6)))
        assertFalse(calendar.isTurnOfMonthWindow(LocalDate.of(2026, 7, 15)))
    }

    @Test
    fun nextAndPreviousSessionSkipWeekends() {
        assertEquals(LocalDate.of(2026, 7, 6),
                     calendar.nextSession(LocalDate.of(2026, 7, 2)))
        assertEquals(LocalDate.of(2026, 7, 2),
                     calendar.sessionOnOrBefore(LocalDate.of(2026, 7, 5)))
    }
}
