package com.edgestack.app

import com.edgestack.app.core.AppJson
import com.edgestack.app.domain.OverlayCalculator
import com.edgestack.app.domain.model.SpyBar
import java.time.LocalDate
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Golden tests pinning the Kotlin port to the Python reference output. */
class OverlayCalculatorTest {

    private data class Fixture(
        val base: Double,
        val bars: List<SpyBar>,
        val applied: List<Pair<LocalDate, Double>>,
    )

    private fun load(name: String): Fixture {
        val text = requireNotNull(javaClass.getResourceAsStream("/fixtures/$name")) {
            "missing fixture $name — run scripts/export_overlay_fixture.py"
        }.bufferedReader().readText()
        val root = AppJson.parseToJsonElement(text).jsonObject
        val bars = root.getValue("bars").jsonArray.map { element ->
            val bar = element.jsonObject
            SpyBar(
                date = LocalDate.parse(bar.getValue("date").jsonPrimitive.content),
                open = bar.getValue("open").jsonPrimitive.double,
                high = bar.getValue("high").jsonPrimitive.double,
                low = bar.getValue("low").jsonPrimitive.double,
                close = bar.getValue("close").jsonPrimitive.double,
                adj = bar.getValue("adj").jsonPrimitive.double,
            )
        }
        val applied = root.getValue("applied_exposure").jsonArray.map { element ->
            val row = element.jsonObject
            LocalDate.parse(row.getValue("date").jsonPrimitive.content) to
                row.getValue("L").jsonPrimitive.double
        }
        return Fixture(root.getValue("base").jsonPrimitive.double, bars, applied)
    }

    private fun assertMatchesReference(name: String) {
        val fixture = load(name)
        val computed = OverlayCalculator.appliedExposure(fixture.bars, fixture.base)
        assertEquals(fixture.applied.size, computed.size)
        fixture.applied.zip(computed).forEach { (expected, actual) ->
            assertEquals("date drift at ${expected.first}", expected.first, actual.date)
            assertEquals(
                "exposure mismatch at ${expected.first}",
                expected.second,
                actual.exposure,
                1e-6,
            )
        }
    }

    @Test
    fun `base 1_0 matches python reference`() = assertMatchesReference("overlay_spy_base10.json")

    @Test
    fun `base 1_3 matches python reference`() = assertMatchesReference("overlay_spy_base13.json")

    @Test
    fun `latest regime reports sma distance and exposure`() {
        val fixture = load("overlay_spy_base10.json")
        val regime = requireNotNull(OverlayCalculator.latestRegime(fixture.bars))
        assertTrue(regime.currentExposure > 0.0)
        val sma = fixture.bars.takeLast(200).sumOf { it.close } / 200.0
        assertEquals(fixture.bars.last().close >= sma, regime.aboveSma200)
    }
}
