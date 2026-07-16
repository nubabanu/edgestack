package com.edgestack.app.data.remote

import com.edgestack.app.domain.model.SpyBar
import kotlinx.serialization.Serializable
import java.time.Instant
import java.time.ZoneId

@Serializable
data class YahooChartResponse(val chart: YahooChart)

@Serializable
data class YahooChart(val result: List<YahooResult>? = null, val error: YahooError? = null)

@Serializable
data class YahooError(val code: String? = null, val description: String? = null)

@Serializable
data class YahooResult(
    val meta: YahooMeta,
    val timestamp: List<Long>? = null,
    val indicators: YahooIndicators,
)

@Serializable
data class YahooMeta(
    val symbol: String,
    val regularMarketPrice: Double? = null,
    val previousClose: Double? = null,
    val exchangeTimezoneName: String = "America/New_York",
)

@Serializable
data class YahooIndicators(
    val quote: List<YahooQuote>,
    val adjclose: List<YahooAdjClose>? = null,
)

@Serializable
data class YahooQuote(
    val open: List<Double?> = emptyList(),
    val high: List<Double?> = emptyList(),
    val low: List<Double?> = emptyList(),
    val close: List<Double?> = emptyList(),
    val volume: List<Long?> = emptyList(),
)

@Serializable
data class YahooAdjClose(val adjclose: List<Double?> = emptyList())

/** Convert to bars, dropping rows with null open/close (Yahoo emits them). */
fun YahooResult.toBars(): List<SpyBar> {
    val ts = timestamp ?: return emptyList()
    val q = indicators.quote.firstOrNull() ?: return emptyList()
    val adj = indicators.adjclose?.firstOrNull()?.adjclose
    val zone = ZoneId.of(meta.exchangeTimezoneName)
    val out = ArrayList<SpyBar>(ts.size)
    for (i in ts.indices) {
        val open = q.open.getOrNull(i) ?: continue
        val close = q.close.getOrNull(i) ?: continue
        val date = Instant.ofEpochSecond(ts[i]).atZone(zone).toLocalDate()
        out += SpyBar(
            date = date,
            open = open,
            high = q.high.getOrNull(i) ?: maxOf(open, close),
            low = q.low.getOrNull(i) ?: minOf(open, close),
            close = close,
            adj = adj?.getOrNull(i) ?: close,
        )
    }
    return out
}
