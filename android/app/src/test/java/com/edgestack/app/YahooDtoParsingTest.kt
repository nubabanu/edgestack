package com.edgestack.app

import com.edgestack.app.core.AppJson
import com.edgestack.app.data.remote.YahooChartResponse
import com.edgestack.app.data.remote.toBars
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class YahooDtoParsingTest {

    @Test
    fun parsesRealResponseAndDropsNullBars() {
        val text = requireNotNull(
            javaClass.getResourceAsStream("/fixtures/yahoo_chart_sample.json")) {
            "missing fixture — run scripts/export_overlay_fixture.py"
        }.bufferedReader().readText()
        val parsed = AppJson.decodeFromString(YahooChartResponse.serializer(), text)
        val result = requireNotNull(parsed.chart.result?.firstOrNull())
        assertEquals("SPY", result.meta.symbol)
        val n = result.timestamp!!.size
        assertEquals(30, n)
        val bars = result.toBars()
        // bar 5 has null open+close in the fixture and must be dropped
        assertEquals(n - 1, bars.size)
        assertTrue(bars.all { it.close > 0 && it.open > 0 && it.adj > 0 })
        assertTrue(bars.zipWithNext().all { (a, b) -> a.date < b.date })
    }
}
