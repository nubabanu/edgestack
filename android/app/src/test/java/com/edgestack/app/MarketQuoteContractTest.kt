package com.edgestack.app

import com.edgestack.app.core.AppJson
import com.edgestack.app.data.repo.QuoteRepository
import com.edgestack.app.domain.model.MarketQuoteV1
import com.edgestack.app.domain.model.QuoteBatchV1
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MarketQuoteContractTest {
    private fun quote(
        symbol: String,
        provider: String = "finnhub",
        freshness: String = "FRESH",
        session: String = "REGULAR",
    ) = MarketQuoteV1(
        symbol = symbol,
        price = 101.0,
        previousClose = 100.0,
        change = 1.0,
        changePercent = 1.0,
        observedAt = "2026-07-20T14:00:00Z",
        marketSession = session,
        freshnessStatus = freshness,
        ageSeconds = 2.0,
        provider = provider,
        feed = if (provider == "alpaca") "IEX_ONLY" else "FINNHUB_US_EQUITIES",
        latency = "REALTIME",
    )

    @Test
    fun gatewayContractParsesSourceAndFreshness() {
        val payload = """
            {
              "requested_at": "2026-07-20T14:00:02Z",
              "quotes": [{
                "symbol": "SPY",
                "price": 101.0,
                "previous_close": 100.0,
                "change": 1.0,
                "change_percent": 1.0,
                "observed_at": "2026-07-20T14:00:00Z",
                "market_session": "REGULAR",
                "freshness_status": "FRESH",
                "age_seconds": 2.0,
                "stale_after_seconds": 120.0,
                "provider": "alpaca",
                "feed": "IEX_ONLY",
                "latency": "REALTIME",
                "from_cache": false,
                "warnings": []
              }],
              "missing_symbols": ["ACN"],
              "providers_attempted": ["alpaca", "yahoo"]
            }
        """.trimIndent()

        val batch = AppJson.decodeFromString(QuoteBatchV1.serializer(), payload)

        assertEquals("alpaca / IEX_ONLY", batch.quotes.single().sourceLabel())
        assertEquals(listOf("ACN"), batch.missingSymbols)
        assertTrue(batch.quotes.single().isAlertEligible())
    }

    @Test
    fun staleAndClosedQuotesCannotTriggerAlerts() {
        assertFalse(quote("SPY", freshness = "STALE").isAlertEligible())
        assertFalse(
            quote("SPY", freshness = "MARKET_CLOSED", session = "CLOSED").isAlertEligible(),
        )
        assertFalse(quote("SPY", session = "UNKNOWN").isAlertEligible())
    }

    @Test
    fun repositoryFillsPartialBackendBatchFromDeviceYahoo() = runBlocking {
        val backend = QuoteBatchV1(
            requestedAt = "2026-07-20T14:00:00Z",
            quotes = listOf(quote("SPY")),
            missingSymbols = listOf("ACN"),
        )
        val repository = QuoteRepository(
            remoteQuotes = { Result.success(backend) },
            deviceQuotes = { symbols ->
                symbols.associateWith {
                    quote(it, provider = "yahoo_device", freshness = "STALE", session = "UNKNOWN")
                }
            },
        )

        val result = repository.latestQuotes(listOf("spy", "ACN", "SPY"))

        assertEquals(setOf("SPY", "ACN"), result.keys)
        assertEquals("finnhub", result.getValue("SPY").provider)
        assertEquals("yahoo_device", result.getValue("ACN").provider)
    }

    @Test
    fun repositoryUsesDeviceYahooWhenPcApiIsUnavailable() = runBlocking {
        val repository = QuoteRepository(
            remoteQuotes = { Result.failure(IllegalStateException("PC offline")) },
            deviceQuotes = { symbols -> symbols.associateWith { quote(it, "yahoo_device") } },
        )

        val result = repository.latestQuotes(listOf("SPY"))

        assertEquals("yahoo_device", result.getValue("SPY").provider)
    }
}
