package com.edgestack.app.domain

import com.edgestack.app.domain.model.OverlayState
import com.edgestack.app.domain.model.SpyBar
import java.time.LocalDate
import java.time.YearMonth
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Exact Kotlin port of scripts/calendar_overlay.py::build_exposure.
 * Pinned to the Python reference by fixture tests (overlay_spy_base{10,13}.json).
 *
 * Rule order matters and mirrors the Python exactly:
 *   base 1.0 -> Sep 0.5 -> Oct/Nov 1.5 -> ToM +0.5 (cap 2.0) -> Nov td7 = 0.0
 *   -> gate: close < SMA200(raw close)  => min(L, 0.5)
 *   -> gate: 20d ann. vol (adj returns, ddof=1) > 0.20 => min(L, 1.0)
 *   -> first 200 rows forced to 1.0
 *   -> applied = shift(1), first value 1.0
 *   -> base != 1.0: applied = clip(applied * base, 2.5)
 * NaN SMA/vol must NOT trigger gates (pandas mask semantics).
 */
object OverlayCalculator {

    const val WARMUP = 200

    data class Result(
        val dates: List<LocalDate>,
        val applied: DoubleArray,       // exposure earning session t's return
        val sma200: DoubleArray,        // NaN until 200 bars
        val vol20: DoubleArray,         // NaN until ~21 bars
    )

    fun compute(bars: List<SpyBar>, base: Double = 1.0): Result {
        val n = bars.size
        val dates = bars.map { it.date }
        val close = DoubleArray(n) { bars[it].close }
        val adj = DoubleArray(n) { bars[it].adj }

        // trading-day-of-month indices from the bars themselves (like Python)
        val tdom = IntArray(n)
        val tdomEnd = IntArray(n)
        run {
            var i = 0
            while (i < n) {
                val ym = YearMonth.from(dates[i])
                var j = i
                while (j < n && YearMonth.from(dates[j]) == ym) j++
                for (k in i until j) {
                    tdom[k] = k - i + 1
                    tdomEnd[k] = j - k
                }
                i = j
            }
        }

        val l = DoubleArray(n)
        for (i in 0 until n) {
            val month = dates[i].monthValue
            var v = 1.0
            if (month == 9) v = 0.5
            if (month == 10 || month == 11) v = 1.5
            if (tdom[i] <= 3 || tdomEnd[i] == 1) v = min(v + 0.5, 2.0)
            if (month == 11 && tdom[i] == 7) v = 0.0
            l[i] = v
        }

        val sma200 = rollingMean(close, WARMUP)
        val ret = DoubleArray(n) { i ->
            if (i == 0) Double.NaN else adj[i] / adj[i - 1] - 1.0
        }
        val vol20 = rollingStd(ret, 20).also { arr ->
            for (i in arr.indices) arr[i] = arr[i] * sqrt(252.0)
        }

        for (i in 0 until n) {
            // NaN comparisons are false in Kotlin too -> gates skip warm-up rows
            if (close[i] < sma200[i]) l[i] = min(l[i], 0.5)
            if (vol20[i] > 0.20) l[i] = min(l[i], 1.0)
        }
        for (i in 0 until min(WARMUP, n)) l[i] = 1.0

        val applied = DoubleArray(n) { i -> if (i == 0) 1.0 else l[i - 1] }
        if (base != 1.0) {
            for (i in 0 until n) applied[i] = min(applied[i] * base, 2.5)
        }
        return Result(dates, applied, sma200, vol20)
    }

    /**
     * Today's state + which rules fired, for the dial screen.
     *
     * [calendar] (the bundled exchange calendar) decides month-position rules.
     * Bar data alone cannot: the last bar of a partial month always looks like
     * month-end, so mid-month days would wrongly light the turn-of-month chip.
     */
    fun state(
        bars: List<SpyBar>,
        base: Double = 1.0,
        historySessions: Int = 120,
        calendar: TradingCalendar? = null,
    ): OverlayState? {
        if (bars.size < 2) return null
        val r = compute(bars, base)
        val i = bars.size - 1
        val today = bars[i]
        val fired = buildList {
            val m = today.date.monthValue
            val cal = TradingCalendarFromBars(r.dates)
            val isTom = calendar?.isTurnOfMonthWindow(today.date) ?: cal.isTom(i)
            val tdom7 = (calendar?.tradingDayOfMonth(today.date) ?: cal.tdom(i)) == 7
            if (m == 9) add("September de-risk 0.5x")
            if (m == 10 || m == 11) add("Oct-Nov boost 1.5x")
            if (isTom) add("Turn-of-month +0.5x")
            if (m == 11 && tdom7) add("Nov worst-day: FLAT")
            if (today.close < r.sma200[i]) add("Below 200-DMA gate (max 0.5x)")
            if (r.vol20[i] > 0.20) add("Vol > 20% gate (max 1.0x)")
            if (base != 1.0) add("Base leverage ${base}x")
        }
        val start = maxOf(0, i - historySessions + 1)
        return OverlayState(
            date = today.date,
            appliedL = r.applied[i],
            firedRules = fired,
            spyClose = today.close,
            sma200 = r.sma200[i].takeIf { !it.isNaN() },
            vol20 = r.vol20[i].takeIf { !it.isNaN() },
            lastReturn = (bars[i].adj / bars[i - 1].adj - 1.0)
                .takeIf { it.isFinite() },
            history = (start..i).map { r.dates[it] to r.applied[it] },
        )
    }

    private fun rollingMean(x: DoubleArray, window: Int): DoubleArray {
        val out = DoubleArray(x.size) { Double.NaN }
        var sum = 0.0
        for (i in x.indices) {
            sum += x[i]
            if (i >= window) sum -= x[i - window]
            if (i >= window - 1) out[i] = sum / window
        }
        return out
    }

    /** Rolling stdev (ddof=1); NaN if any NaN inside the window (pandas min_periods). */
    private fun rollingStd(x: DoubleArray, window: Int): DoubleArray {
        val out = DoubleArray(x.size) { Double.NaN }
        for (i in window - 1 until x.size) {
            var mean = 0.0
            var ok = true
            for (j in i - window + 1..i) {
                if (x[j].isNaN()) { ok = false; break }
                mean += x[j]
            }
            if (!ok) continue
            mean /= window
            var ss = 0.0
            for (j in i - window + 1..i) {
                val d = x[j] - mean
                ss += d * d
            }
            out[i] = sqrt(ss / (window - 1))
        }
        return out
    }

    /** Minimal month-position helper over the bar dates (mirrors Python groupby). */
    private class TradingCalendarFromBars(private val dates: List<LocalDate>) {
        fun tdom(i: Int): Int {
            val ym = YearMonth.from(dates[i])
            var k = i
            while (k > 0 && YearMonth.from(dates[k - 1]) == ym) k--
            return i - k + 1
        }
        fun tdomEnd(i: Int): Int {
            val ym = YearMonth.from(dates[i])
            var k = i
            while (k < dates.size - 1 && YearMonth.from(dates[k + 1]) == ym) k++
            return k - i + 1
        }
        fun isTom(i: Int): Boolean = tdom(i) <= 3 || tdomEnd(i) == 1
    }
}
