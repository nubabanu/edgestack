package com.edgestack.app

import com.edgestack.app.domain.model.TrackedPosition
import com.edgestack.app.domain.pnlAt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class PositionMathTest {

    private val position = TrackedPosition(
        symbol = "SPY",
        entryPrice = 700.0,
        quantity = 10.0,
        entryDate = "2026-07-17",
    )

    @Test
    fun `profit and percent from a winning quote`() {
        val pnl = position.pnlAt(750.0)!!
        assertEquals(7500.0, pnl.marketValue, 1e-9)
        assertEquals(500.0, pnl.profit, 1e-9)
        assertEquals(50.0 / 700.0 * 100.0, pnl.profitPercent, 1e-9)
    }

    @Test
    fun `loss is negative`() {
        val pnl = position.pnlAt(630.0)!!
        assertEquals(-700.0, pnl.profit, 1e-9)
        assertEquals(-10.0, pnl.profitPercent, 1e-9)
    }

    @Test
    fun `missing or invalid quotes yield null`() {
        assertNull(position.pnlAt(null))
        assertNull(position.pnlAt(0.0))
        assertNull(position.copy(entryPrice = 0.0).pnlAt(750.0))
    }
}
