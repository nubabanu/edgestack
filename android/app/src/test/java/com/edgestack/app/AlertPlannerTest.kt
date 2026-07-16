package com.edgestack.app

import com.edgestack.app.domain.AlertPlanner
import com.edgestack.app.domain.AlertType
import com.edgestack.app.domain.TradingCalendar
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate
import java.time.ZoneId

class AlertPlannerTest {

    private val calendar: TradingCalendar by lazy {
        val text = requireNotNull(javaClass.getResourceAsStream("/fixtures/calendar.json"))
            .bufferedReader().readText()
        val sessions = Json.parseToJsonElement(text)
            .jsonObject["sessions"]!!.jsonArray.map { LocalDate.parse(it.jsonPrimitive.content) }
        TradingCalendar(sessions)
    }

    private fun types(d: LocalDate): List<AlertType> =
        AlertPlanner.alertsFor(calendar, d).map { it.type }

    @Test
    fun nonSessionDayHasNoAlerts() {
        assertTrue(types(LocalDate.of(2026, 7, 4)).isEmpty())
        assertTrue(types(LocalDate.of(2026, 11, 26)).isEmpty()) // Thanksgiving
    }

    @Test
    fun febSniperFiresOnLastJanuarySession() {
        assertTrue(AlertType.FEB_SNIPER in types(LocalDate.of(2027, 1, 29)))
    }

    @Test
    fun septemberDeriskFiresOnLastAugustSession() {
        assertTrue(AlertType.SEPTEMBER_DERISK in types(LocalDate.of(2026, 8, 31)))
    }

    @Test
    fun novWorstDayFiresOnSixthNovemberSession() {
        // 6th trading day of Nov 2026 = Nov 9; the worst day (td7) is Nov 10.
        assertTrue(AlertType.NOV_WORST_DAY_FLAT in types(LocalDate.of(2026, 11, 9)))
        assertTrue(AlertType.NOV_WORST_DAY_FLAT !in types(LocalDate.of(2026, 11, 10)))
    }

    @Test
    fun tomWindowStartFiresOnPenultimateSession() {
        // July 2026: last session Jul 31 -> penultimate Jul 30.
        assertTrue(AlertType.TOM_WINDOW_START in types(LocalDate.of(2026, 7, 30)))
    }

    @Test
    fun alertsAnchorAtMarketTimeDuringDstDivergence() {
        // Nov 2, 2026: US DST ended Nov 1, EU ended Oct 25 — both on winter time,
        // 15:45 ET == 21:45 Berlin.
        val nov = AlertPlanner.fireAt(LocalDate.of(2026, 11, 2))
            .withZoneSameInstant(ZoneId.of("Europe/Berlin"))
        assertEquals(21, nov.hour)
        // Oct 26, 2026: EU already on winter time, US still on DST ->
        // 15:45 ET == 20:45 Berlin. A fixed 21:45 Berlin alarm would be too late.
        val oct = AlertPlanner.fireAt(LocalDate.of(2026, 10, 26))
            .withZoneSameInstant(ZoneId.of("Europe/Berlin"))
        assertEquals(20, oct.hour)
        assertEquals(45, oct.minute)
    }
}
