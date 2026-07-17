package com.edgestack.app.domain

import com.edgestack.app.domain.model.SpyBar
import java.time.LocalDate
import kotlin.math.sqrt

/**
 * Kotlin port of the Python reference `scripts/calendar_overlay.py:build_exposure`.
 * Pinned to the committed golden fixtures; any change must keep them passing.
 *
 * Rules (decided at close t-1, applied to session t):
 * seasonal base (Sep 0.5, Oct/Nov 1.5, else 1.0) + 0.5 in the turn-of-month
 * window capped at 2.0; November's 7th trading day forces 0.0; below the
 * 200-bar SMA the exposure is clipped to 0.5; 20-day annualized vol above 20%
 * clips to 1.0; the first 200 bars are plain 1.0 warmup; a non-unit base
 * scales the applied series with a 2.5 cap.
 */
object OverlayCalculator {

    data class Applied(val date: LocalDate, val exposure: Double)

    fun appliedExposure(bars: List<SpyBar>, base: Double = 1.0): List<Applied> {
        if (bars.isEmpty()) return emptyList()
        val n = bars.size
        val decided = DoubleArray(n)

        val tdom = IntArray(n)
        val tdomFromEnd = IntArray(n)
        run {
            var count = 0
            var month: Pair<Int, Int>? = null
            for (i in 0 until n) {
                val key = bars[i].date.year to bars[i].date.monthValue
                count = if (key == month) count + 1 else 1
                month = key
                tdom[i] = count
            }
            count = 0
            month = null
            for (i in n - 1 downTo 0) {
                val key = bars[i].date.year to bars[i].date.monthValue
                count = if (key == month) count + 1 else 1
                month = key
                tdomFromEnd[i] = count
            }
        }

        for (i in 0 until n) {
            val m = bars[i].date.monthValue
            var level = when (m) {
                9 -> 0.5
                10, 11 -> 1.5
                else -> 1.0
            }
            if (tdom[i] <= 3 || tdomFromEnd[i] == 1) level += 0.5
            if (level > 2.0) level = 2.0
            if (m == 11 && tdom[i] == 7) level = 0.0
            decided[i] = level
        }

        var closeSum = 0.0
        for (i in 0 until n) {
            closeSum += bars[i].close
            if (i >= 200) closeSum -= bars[i - 200].close
            if (i >= 199) {
                val sma200 = closeSum / 200.0
                if (bars[i].close < sma200 && decided[i] > 0.5) decided[i] = 0.5
            }
        }

        val returns = DoubleArray(n)
        for (i in 1 until n) returns[i] = bars[i].adj / bars[i - 1].adj - 1.0
        for (i in 0 until n) {
            if (i < 20) continue  // needs 20 returns (indices i-19..i)
            var mean = 0.0
            for (j in i - 19..i) mean += returns[j]
            mean /= 20.0
            var variance = 0.0
            for (j in i - 19..i) {
                val d = returns[j] - mean
                variance += d * d
            }
            val vol20 = sqrt(variance / 19.0) * sqrt(252.0)
            if (vol20 > 0.20 && decided[i] > 1.0) decided[i] = 1.0
        }

        for (i in 0 until minOf(200, n)) decided[i] = 1.0

        return List(n) { i ->
            var applied = if (i == 0) 1.0 else decided[i - 1]
            if (base != 1.0) {
                applied *= base
                if (applied > 2.5) applied = 2.5
            }
            Applied(bars[i].date, applied)
        }
    }

    /** Latest regime snapshot for the UI card; null until 200 bars exist. */
    data class Regime(
        val aboveSma200: Boolean,
        val sma200DistancePercent: Double,
        val currentExposure: Double,
    )

    fun latestRegime(bars: List<SpyBar>, base: Double = 1.0): Regime? {
        if (bars.size < 201) return null
        val sma200 = bars.takeLast(200).sumOf { it.close } / 200.0
        val last = bars.last().close
        val exposure = appliedExposure(bars, base).last().exposure
        return Regime(
            aboveSma200 = last >= sma200,
            sma200DistancePercent = (last / sma200 - 1.0) * 100.0,
            currentExposure = exposure,
        )
    }
}
