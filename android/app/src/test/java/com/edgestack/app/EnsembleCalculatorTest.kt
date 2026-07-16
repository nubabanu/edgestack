package com.edgestack.app

import com.edgestack.app.domain.EnsembleCalculator
import com.edgestack.app.domain.model.SpyBar
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate
import kotlin.math.abs

@Serializable
private data class EnsBar(
    val date: String, val open: Double, val high: Double,
    val low: Double, val close: Double, val adj: Double,
)

@Serializable
private data class EnsPoint(val date: String, @SerialName("e") val e: Double)

@Serializable
private data class EnsFixture(
    val bars: List<EnsBar>,
    val exposure: List<EnsPoint>,
    @SerialName("last_families") val lastFamilies: Map<String, Double>,
)

class EnsembleCalculatorTest {

    @Test
    fun parityWithPythonReference() {
        val text = requireNotNull(
            javaClass.getResourceAsStream("/fixtures/ensemble_spy.json")) {
            "missing fixture — run scripts/export_overlay_fixture.py"
        }.bufferedReader().readText()
        val fx = Json { ignoreUnknownKeys = true }
            .decodeFromString(EnsFixture.serializer(), text)
        val bars = fx.bars.map {
            SpyBar(LocalDate.parse(it.date), it.open, it.high, it.low,
                   it.close, it.adj)
        }
        val exp = EnsembleCalculator.exposureSeries(bars)
        assertEquals(fx.exposure.size, exp.size)
        var maxDiff = 0.0
        var worst = ""
        for (i in fx.exposure.indices) {
            val py = fx.exposure[i].e
            val kt = exp[i]
            val d = if (py.isNaN() && kt.isNaN()) 0.0 else abs(py - kt)
            if (d > maxDiff) { maxDiff = d; worst = fx.exposure[i].date }
        }
        assertTrue("max |dE| = $maxDiff at $worst", maxDiff < 1e-6)

        val state = requireNotNull(EnsembleCalculator.state(bars))
        for ((k, v) in fx.lastFamilies) {
            assertEquals("family $k", v, state.families[k]!!, 1e-6)
        }
    }
}
