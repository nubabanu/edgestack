package com.edgestack.app.data.remote

import com.edgestack.app.core.AppJson
import com.edgestack.app.domain.model.SpyBar
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
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

    /** One retried batched spark request; per-symbol charts as fallback. */
    suspend fun latestQuotes(symbols: List<String>): Map<String, Double> {
        if (symbols.isEmpty()) return emptyMap()
        val spark = runCatching { sparkQuotes(symbols) }.getOrDefault(emptyMap())
        if (spark.size == symbols.size) return spark
        val missing = symbols.filter { it !in spark }
        return spark + perSymbolQuotes(missing)
    }

    private suspend fun sparkQuotes(symbols: List<String>): Map<String, Double> =
        withRetry(attempts = 3) {
            withContext(Dispatchers.IO) {
                val url = "https://query1.finance.yahoo.com/v8/finance/spark".toHttpUrl()
                    .newBuilder()
                    .addQueryParameter("symbols", symbols.joinToString(","))
                    .addQueryParameter("range", "1d")
                    .addQueryParameter("interval", "1d")
                    .build()
                val request = Request.Builder().url(url)
                    .header("User-Agent", USER_AGENT).build()
                http.newCall(request).execute().use { resp ->
                    if (!resp.isSuccessful) throw IOException("spark ${resp.code}")
                    val body = resp.body?.string() ?: throw IOException("empty spark")
                    val parsed = AppJson.decodeFromString(SparkResponse.serializer(), body)
                    parsed.spark.result.orEmpty().mapNotNull { r ->
                        val price = r.response?.firstOrNull()?.let { res ->
                            res.meta.regularMarketPrice
                                ?: res.indicators.quote.firstOrNull()
                                    ?.close?.lastOrNull { c -> c != null }
                        }
                        price?.let { r.symbol to it }
                    }.toMap()
                }
            }
        }

    private suspend fun perSymbolQuotes(symbols: List<String>): Map<String, Double> =
        coroutineScope {
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

    private suspend fun <T> withRetry(attempts: Int, block: suspend () -> T): T {
        var last: Throwable? = null
        repeat(attempts) { k ->
            try {
                return block()
            } catch (e: Exception) {
                last = e
                delay(400L * (1L shl k))   // 400ms, 800ms, ...
            }
        }
        throw last ?: IOException("retry failed")
    }

    companion object {
        const val USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
}
