package com.edgestack.app.data.remote

import com.edgestack.app.core.AppJson
import com.edgestack.app.domain.model.MarketQuoteV1
import com.edgestack.app.domain.model.SpyBar
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.IOException
import java.time.Duration
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

    /** Backward-compatible price-only projection of [latestQuoteSnapshots]. */
    suspend fun latestQuotes(symbols: List<String>): Map<String, Double> {
        return latestQuoteSnapshots(symbols).mapValues { it.value.price }
    }

    /** One retried batched spark request; per-symbol charts as fallback. */
    suspend fun latestQuoteSnapshots(symbols: List<String>): Map<String, MarketQuoteV1> {
        val requested = symbols.map { it.trim().uppercase() }.filter { it.isNotBlank() }.distinct()
        if (requested.isEmpty()) return emptyMap()
        val spark = runCatching { sparkQuotes(requested) }.getOrDefault(emptyMap())
        if (spark.size == requested.size) return spark
        val missing = requested.filter { it !in spark }
        return spark + perSymbolQuotes(missing)
    }

    private suspend fun sparkQuotes(symbols: List<String>): Map<String, MarketQuoteV1> =
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
                    parseSparkQuotes(body, symbols)
                }
            }
        }

    private fun parseSparkQuotes(body: String, symbols: List<String>): Map<String, MarketQuoteV1> {
        val root = AppJson.parseToJsonElement(body).jsonObject
        if ("spark" in root) {
            val parsed = AppJson.decodeFromString(SparkResponse.serializer(), body)
            return parsed.spark.result.orEmpty().mapNotNull { result ->
                result.response?.firstOrNull()?.toMarketQuote(result.symbol)
                    ?.let { result.symbol to it }
            }.toMap()
        }
        return symbols.mapNotNull { symbol ->
            val item = root[symbol]?.jsonObject ?: return@mapNotNull null
            val timestamps = item["timestamp"]?.jsonArray.orEmpty()
            val closes = item["close"]?.jsonArray.orEmpty()
            val lastIndex = closes.indices.lastOrNull {
                closes[it].jsonPrimitive.doubleOrNull != null && it < timestamps.size
            } ?: return@mapNotNull null
            val price = closes[lastIndex].jsonPrimitive.doubleOrNull ?: return@mapNotNull null
            val observed = timestamps[lastIndex].jsonPrimitive.longOrNull ?: return@mapNotNull null
            val previous = item["chartPreviousClose"]?.jsonPrimitive?.doubleOrNull
                ?: item["previousClose"]?.jsonPrimitive?.doubleOrNull
            buildMarketQuote(symbol, price, previous, observed, null)?.let { symbol to it }
        }.toMap()
    }

    private suspend fun perSymbolQuotes(symbols: List<String>): Map<String, MarketQuoteV1> =
        coroutineScope {
            val now = Instant.now()
            symbols.map { symbol ->
                async {
                    runCatching {
                        val r = fetchResult(symbol, now.minus(7, ChronoUnit.DAYS), now)
                        symbol to r.toMarketQuote(symbol)
                    }.getOrNull()
                }
            }.mapNotNull { it.await() }
                .mapNotNull { (symbol, quote) -> quote?.let { symbol to it } }
                .toMap()
        }

    private fun YahooResult.toMarketQuote(requestedSymbol: String): MarketQuoteV1? {
        val lastClose = indicators.quote.firstOrNull()?.close?.lastOrNull { it != null }
        val price = meta.regularMarketPrice ?: lastClose ?: return null
        val observedEpoch = meta.regularMarketTime ?: timestamp?.lastOrNull() ?: return null
        val previous = meta.chartPreviousClose ?: meta.previousClose
        return buildMarketQuote(requestedSymbol, price, previous, observedEpoch, meta.marketState)
    }

    private fun buildMarketQuote(
        requestedSymbol: String,
        price: Double,
        previous: Double?,
        observedEpoch: Long,
        marketState: String?,
    ): MarketQuoteV1? {
        if (!price.isFinite() || price <= 0.0) return null
        val observed = Instant.ofEpochSecond(observedEpoch)
        val age = maxOf(Duration.between(observed, Instant.now()).toMillis() / 1000.0, 0.0)
        val marketSession = when (marketState?.uppercase()) {
            "PRE", "PREPRE" -> "PRE"
            "REGULAR" -> "REGULAR"
            "POST", "POSTPOST" -> "POST"
            "CLOSED" -> "CLOSED"
            else -> "UNKNOWN"
        }
        val freshness = when {
            marketSession == "CLOSED" -> "MARKET_CLOSED"
            marketSession == "UNKNOWN" -> "STALE"
            age > STALE_AFTER_SECONDS -> "STALE"
            else -> "FRESH"
        }
        val change = previous?.let { price - it }
        return MarketQuoteV1(
            symbol = requestedSymbol.uppercase(),
            price = price,
            previousClose = previous,
            change = change,
            changePercent = previous?.takeIf { it > 0.0 }?.let { (price - it) / it * 100.0 },
            observedAt = observed.toString(),
            marketSession = marketSession,
            freshnessStatus = freshness,
            ageSeconds = age,
            staleAfterSeconds = STALE_AFTER_SECONDS,
            provider = "yahoo_device",
            feed = "UNOFFICIAL_YAHOO",
            latency = "UNKNOWN",
            warnings = listOf(
                "Device fallback; unofficial quote may be delayed, throttled, or unavailable.",
            ),
        )
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
        const val STALE_AFTER_SECONDS = 120.0
    }
}
