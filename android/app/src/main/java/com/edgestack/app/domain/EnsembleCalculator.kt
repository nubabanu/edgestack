package com.edgestack.app.domain

import com.edgestack.app.domain.model.SpyBar
import java.time.LocalDate
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Kotlin port of edgestack.strategies.composite (ensemble4) — the validated
 * 4-family blend. Pinned to the Python reference by fixtures/ensemble_spy.json.
 *
 * Families (equal weight):
 *   trendOrDip   close > SMA200, OR RSI(2) < 10 panic day
 *   breakout20d  1.0 for 10 sessions after close > prior 20-session close-high
 *   volTarget    0.10 / (20d realized vol of adj returns), capped 1.5
 *   reversion3dn 1.0 the day after three consecutive down closes
 */
object EnsembleCalculator {

    data class State(
        val date: LocalDate,
        val exposure: Double,
        val families: Map<String, Double>,
        val history: List<Pair<LocalDate, Double>>,
    )

    fun familySeries(bars: List<SpyBar>): Map<String, DoubleArray> {
        val n = bars.size
        val close = DoubleArray(n) { bars[it].close }
        val adj = DoubleArray(n) { bars[it].adj }
        val ret = DoubleArray(n) { i ->
            if (i == 0) Double.NaN else adj[i] / adj[i - 1] - 1.0
        }
        val sma200 = rollingMean(close, 200)
        val rsi2 = rsi(close, 2)
        val hi20 = rollingMax(close, 20)

        val trendOrDip = DoubleArray(n) { i ->
            if (close[i] > sma200[i] || rsi2[i] < 10.0) 1.0 else 0.0
        }
        val breakoutTrigger = DoubleArray(n) { i ->
            if (i > 0 && !hi20[i - 1].isNaN() && close[i] > hi20[i - 1]) 1.0 else 0.0
        }
        val breakout = DoubleArray(n) { i ->
            var v = 0.0
            for (j in maxOf(0, i - 9)..i) if (breakoutTrigger[j] == 1.0) v = 1.0
            v
        }
        val vol20 = rollingStd(ret, 20)
        val volTarget = DoubleArray(n) { i ->
            val v = vol20[i] * sqrt(252.0)
            if (v.isNaN() || v <= 0) 0.0 else min(0.10 / v, 1.5)
        }
        val rev3 = DoubleArray(n) { i ->
            if (i >= 3 && ret[i] < 0 && ret[i - 1] < 0 && ret[i - 2] < 0) 1.0 else 0.0
        }
        return mapOf(
            "trend_or_dip" to trendOrDip,
            "breakout_20d" to breakout,
            "vol_target" to volTarget,
            "reversion_3dn" to rev3,
        )
    }

    fun exposureSeries(bars: List<SpyBar>): DoubleArray {
        val fams = familySeries(bars)
        val n = bars.size
        return DoubleArray(n) { i -> fams.values.sumOf { it[i] } / fams.size }
    }

    fun state(bars: List<SpyBar>, historySessions: Int = 120): State? {
        if (bars.size < 21) return null
        val fams = familySeries(bars)
        val exp = exposureSeries(bars)
        val i = bars.size - 1
        val start = maxOf(0, i - historySessions + 1)
        return State(
            date = bars[i].date,
            exposure = exp[i],
            families = fams.mapValues { it.value[i] },
            history = (start..i).map { bars[it].date to exp[it] },
        )
    }

    // --- helpers matching pandas semantics ---
    private fun rollingMean(x: DoubleArray, w: Int): DoubleArray {
        val out = DoubleArray(x.size) { Double.NaN }
        var sum = 0.0
        for (i in x.indices) {
            sum += x[i]
            if (i >= w) sum -= x[i - w]
            if (i >= w - 1) out[i] = sum / w
        }
        return out
    }

    private fun rollingMax(x: DoubleArray, w: Int): DoubleArray {
        val out = DoubleArray(x.size) { Double.NaN }
        for (i in w - 1 until x.size) {
            var m = x[i - w + 1]
            for (j in i - w + 2..i) if (x[j] > m) m = x[j]
            out[i] = m
        }
        return out
    }

    private fun rollingStd(x: DoubleArray, w: Int): DoubleArray {
        val out = DoubleArray(x.size) { Double.NaN }
        for (i in w - 1 until x.size) {
            var mean = 0.0
            var ok = true
            for (j in i - w + 1..i) {
                if (x[j].isNaN()) { ok = false; break }
                mean += x[j]
            }
            if (!ok) continue
            mean /= w
            var ss = 0.0
            for (j in i - w + 1..i) {
                val d = x[j] - mean
                ss += d * d
            }
            out[i] = sqrt(ss / (w - 1))
        }
        return out
    }

    /** Wilder RSI via EMA(alpha=1/n), matching pandas ewm(adjust=False). */
    private fun rsi(close: DoubleArray, n: Int): DoubleArray {
        val out = DoubleArray(close.size) { Double.NaN }
        if (close.size < 2) return out
        val alpha = 1.0 / n
        var up = 0.0
        var dn = 0.0
        for (i in 1 until close.size) {
            val d = close[i] - close[i - 1]
            val u = if (d > 0) d else 0.0
            val v = if (d < 0) -d else 0.0
            if (i == 1) { up = u; dn = v }
            else {
                up = alpha * u + (1 - alpha) * up
                dn = alpha * v + (1 - alpha) * dn
            }
            out[i] = if (dn == 0.0) 100.0 else 100.0 - 100.0 / (1.0 + up / dn)
        }
        return out
    }
}
