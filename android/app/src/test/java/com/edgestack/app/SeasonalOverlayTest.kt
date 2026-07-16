package com.edgestack.app

import com.edgestack.app.domain.SeasonalOverlay
import com.edgestack.app.domain.model.TailwindCalendarCellV2
import com.edgestack.app.domain.model.TailwindCalendarV2
import com.edgestack.app.domain.model.WinScoreV2
import java.time.LocalDate
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SeasonalOverlayTest {

    private fun cell(slotKey: String, percentile: Double) = TailwindCalendarCellV2(
        slotKey = slotKey,
        displayLabel = "slot $slotKey",
        entryWindow = "entry",
        exitWindow = "exit",
        horizon = "WEEK",
        score = WinScoreV2(
            netWinRate = 0.5,
            shrunkWinRate = 0.5,
            winScore = 50.0,
            expectedNetReturn = 0.0,
            lower95 = -0.01,
            observations = 100,
            effectiveSampleSize = 80.0,
            rank = 1,
            candidatesRanked = 5,
            multipleTestingAdjustedPvalue = 0.5,
            evidenceGrade = "INSUFFICIENT",
        ),
        rankPercentile = percentile,
    )

    private fun calendar(resolution: String, vararg cells: TailwindCalendarCellV2) =
        TailwindCalendarV2(
            resolution = resolution,
            timezone = "America/New_York",
            horizon = "WEEK",
            cells = cells.toList(),
            warning = "research only",
        )

    @Test
    fun `weekday slots use Monday=0 like the server`() {
        val overlay = SeasonalOverlay(listOf(calendar("DAY", cell("0", 1.0), cell("4", 0.0))))
        // 2026-07-20 is a Monday, 2026-07-24 a Friday.
        assertEquals(1.0, overlay.edgeFor(LocalDate.of(2026, 7, 20))!!.percentile, 1e-9)
        assertEquals(0.0, overlay.edgeFor(LocalDate.of(2026, 7, 24))!!.percentile, 1e-9)
        assertNull(overlay.edgeFor(LocalDate.of(2026, 7, 21))) // Tuesday: no slot present
    }

    @Test
    fun `week-of-month buckets cap at 4 and months are 1-based`() {
        val overlay = SeasonalOverlay(
            listOf(
                calendar("MONTH", cell("0", 1.0), cell("4", 0.2)),
                calendar("YEAR", cell("7", 0.6)),
            ),
        )
        // Day 3 → bucket 0; day 31 → bucket min(4, 30/5=6) = 4. July matches YEAR slot "7".
        val early = overlay.edgeFor(LocalDate.of(2026, 7, 3))!!
        assertEquals((1.0 + 0.6) / 2, early.percentile, 1e-9)
        val late = overlay.edgeFor(LocalDate.of(2026, 7, 31))!!
        assertEquals((0.2 + 0.6) / 2, late.percentile, 1e-9)
        assertEquals(listOf("Week of month", "Month"), late.components.map { it.first })
    }

    @Test
    fun `empty overlay reports no data`() {
        val overlay = SeasonalOverlay(emptyList())
        assertTrue(!overlay.hasData)
        assertNull(overlay.edgeFor(LocalDate.of(2026, 7, 20)))
    }
}
