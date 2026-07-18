package com.edgestack.app

import com.edgestack.app.domain.MacroEventBook
import com.edgestack.app.domain.model.MacroEvent
import java.time.LocalDate
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class MacroEventBookTest {

    private val book = MacroEventBook(
        listOf(
            MacroEvent("2026-07-29", "FOMC", "FOMC rate decision", "14:00"),
            MacroEvent("2026-07-29", "EIA", "EIA petroleum report", "10:30"),
            MacroEvent("2026-08-12", "CPI", "CPI release", "08:30"),
            MacroEvent("not-a-date", "CPI", "bad row ignored", "08:30"),
        ),
    )

    @Test
    fun `events are indexed by date`() {
        assertEquals(2, book.on(LocalDate.of(2026, 7, 29)).size)
        assertTrue(book.on(LocalDate.of(2026, 7, 30)).isEmpty())
    }

    @Test
    fun `marker joins distinct types and is null on quiet days`() {
        assertEquals("FOMC+EIA", book.marker(LocalDate.of(2026, 7, 29)))
        assertEquals("CPI", book.marker(LocalDate.of(2026, 8, 12)))
        assertNull(book.marker(LocalDate.of(2026, 8, 13)))
    }

    @Test
    fun `high impact excludes routine EIA`() {
        val impact = book.highImpactOn(LocalDate.of(2026, 7, 29))
        assertEquals(listOf("FOMC"), impact.map { it.type })
    }
}
