package com.edgestack.app

import com.edgestack.app.domain.OverlayCalculator
import com.edgestack.app.domain.model.SpyBar
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate
import kotlin.math.abs

@Serializable
private data class FixtureBar(
    val date: String, val open: Double, val high: Double,
    val low: Double, val close: Double, val adj: Double,
)

@Serializable
private data class FixtureExposure(val date: String, @SerialName("L") val l: Double)

@Serializable
private data class OverlayFixture(
    val base: Double,
    val bars: List<FixtureBar>,
    @SerialName("applied_exposure") val appliedExposure: List<FixtureExposure>,
)

class OverlayCalculatorTest {

    private val json = Json { ignoreUnknownKeys = true }

    private fun load(name: String): OverlayFixture {
        val text = requireNotNull(javaClass.getResourceAsStream("/fixtures/$name")) {
            "missing fixture $name — run scripts/export_overlay_fixture.py"
        }.bufferedReader().readText()
        return json.decodeFromString(OverlayFixture.serializer(), text)
    }

    private fun assertParity(name: String) {
        val fx = load(name)
        val bars = fx.bars.map {
            SpyBar(LocalDate.parse(it.date), it.open, it.high, it.low, it.close, it.adj)
        }
        val result = OverlayCalculator.compute(bars, base = fx.base)
        assertEquals(fx.appliedExposure.size, result.applied.size)
        var maxDiff = 0.0
        var worst = ""
        for (i in fx.appliedExposure.indices) {
            assertEquals(fx.appliedExposure[i].date, result.dates[i].toString())
            val diff = abs(fx.appliedExposure[i].l - result.applied[i])
            if (diff > maxDiff) {
                maxDiff = diff
                worst = "${fx.appliedExposure[i].date}: py=${fx.appliedExposure[i].l} kt=${result.applied[i]}"
            }
        }
        assertTrue("max |dL| = $maxDiff at $worst", maxDiff < 1e-6)
    }

    @Test fun parityWithPythonBase10() = assertParity("overlay_spy_base10.json")

    @Test fun parityWithPythonBase13() = assertParity("overlay_spy_base13.json")

    @Test
    fun tomChipUsesExchangeCalendarNotBarBoundary() {
        val fx = load("overlay_spy_base10.json")
        // slice so the series ends mid-month: 2026-07-10 is July's 7th session
        val bars = fx.bars.map {
            SpyBar(LocalDate.parse(it.date), it.open, it.high, it.low, it.close, it.adj)
        }.filter { it.date <= LocalDate.of(2026, 7, 10) }
        val calText = requireNotNull(
            javaClass.getResourceAsStream("/fixtures/calendar.json"))
            .bufferedReader().readText()
        val sessions = kotlinx.serialization.json.Json.parseToJsonElement(calText)
            .jsonObject["sessions"]!!.jsonArray
            .map { LocalDate.parse(it.jsonPrimitive.content) }
        val calendar = com.edgestack.app.domain.TradingCalendar(sessions)

        // without a calendar the final bar looks like month-end -> chip fires
        val naive = requireNotNull(OverlayCalculator.state(bars))
        assertTrue(naive.firedRules.any { it.contains("Turn-of-month") })
        // with the bundled calendar, mid-month is correctly NOT turn-of-month
        val informed = requireNotNull(OverlayCalculator.state(bars, calendar = calendar))
        assertTrue(informed.firedRules.none { it.contains("Turn-of-month") })
    }

    @Test
    fun stateReportsFiredRules() {
        val fx = load("overlay_spy_base10.json")
        val bars = fx.bars.map {
            SpyBar(LocalDate.parse(it.date), it.open, it.high, it.low, it.close, it.adj)
        }
        val state = requireNotNull(OverlayCalculator.state(bars, base = 1.3))
        assertEquals(bars.last().date, state.date)
        assertTrue(state.appliedL > 0.0)
        assertTrue(state.history.isNotEmpty())
        assertTrue(state.firedRules.any { it.contains("1.3") })
    }
}
