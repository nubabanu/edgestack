package com.edgestack.app.data.remote

import com.edgestack.app.core.AppJson
import com.edgestack.app.domain.model.SpyBar
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.IOException
import java.time.Instant
import java.time.temporal.ChronoUnit

/**
 * Yahoo v8 chart endpoint (no auth/crumb needed, unlike v7/quote).
 * Used for SPY daily history (overlay + 200-DMA) and board live quotes.
 */
class YahooChartClient(private val http: OkHttpClient) {

    private fun chartUrl(symbol: String, from: Instant, to: Instant, interval: String) =
        "https://query1.finance.yahoo.com/v8/finance/chart/$symbol".toHttpUrl()
            .newBuilder()
            .addQueryParameter("period1", from.epochSecond.toString())
            .addQueryParameter("period2", to.epochSecond.toString())
            .addQueryParameter("interval", interval)
            .addQueryParameter("events", "div,splits")
            .addQueryParameter("includeAdjustedClose", "true")
            .build()

    private suspend fun fetchResult(symbol: String, from: Instant, to: Instant): YahooResult =
        withContext(Dispatchers.IO) {
            val request = Request.Builder()
                .url(chartUrl(symbol, from, to, "1d"))
                .header("User-Agent", USER_AGENT)
                .build()
            http.newCall(request).execute().use { resp ->
                if (!resp.isSuccessful) throw IOException("yahoo ${resp.code} for $symbol")
                val body = resp.body?.string() ?: throw IOException("empty body for $symbol")
                val parsed = AppJson.decodeFromString(
                    YahooChartResponse.serializer(), body)
                parsed.chart.result?.firstOrNull()
                    ?: throw IOException(parsed.chart.error?.description ?: "no result for $symbol")
            }
        }

    /** Daily bars for the last [years] years (needs >=201 bars for a valid SMA200). */
    suspend fun dailyHistory(symbol: String, years: Long = 3): List<SpyBar> {
        val now = Instant.now()
        return fetchResult(symbol, now.minus(years * 365, ChronoUnit.DAYS), now).toBars()
    }

    /** Latest regular-market price per symbol; failed symbols are omitted. */
    suspend fun latestQuotes(symbols: List<String>): Map<String, Double> = coroutineScope {
        val now = Instant.now()
        symbols.map { symbol ->
            async {
                runCatching {
                    val r = fetchResult(symbol, now.minus(7, ChronoUnit.DAYS), now)
                    symbol to (r.meta.regularMarketPrice
                        ?: r.toBars().lastOrNull()?.close)
                }.getOrNull()
            }
        }.mapNotNull { it.await() }
            .mapNotNull { (s, p) -> p?.let { s to it } }
            .toMap()
    }

    companion object {
        const val USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
}
