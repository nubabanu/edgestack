package com.edgestack.app.data.repo

import com.edgestack.app.data.remote.YahooChartClient
import com.edgestack.app.domain.model.MarketQuoteV1
import com.edgestack.app.domain.model.QuoteBatchV1

/** Backend-first indicative quotes with the existing no-key device feed as fallback. */
class QuoteRepository(
    private val remoteQuotes: suspend (List<String>) -> Result<QuoteBatchV1>,
    private val deviceQuotes: suspend (List<String>) -> Map<String, MarketQuoteV1>,
) {
    constructor(syncRepo: SyncRepository, yahoo: YahooChartClient) : this(
        remoteQuotes = syncRepo::marketQuotes,
        deviceQuotes = yahoo::latestQuoteSnapshots,
    )

    suspend fun latestQuotes(symbols: List<String>): Map<String, MarketQuoteV1> {
        val requested = symbols.map { it.trim().uppercase() }.filter { it.isNotBlank() }.distinct()
        if (requested.isEmpty()) return emptyMap()

        val remoteResult = remoteQuotes(requested)
        val resolved = remoteResult.getOrNull()?.quotes
            ?.filter { it.symbol in requested }
            ?.associateBy { it.symbol }
            ?.toMutableMap()
            ?: mutableMapOf()
        val missing = requested.filter { it !in resolved }
        if (missing.isEmpty()) return resolved

        val fallbackResult = runCatching { deviceQuotes(missing) }
        fallbackResult.getOrNull()?.forEach { (symbol, quote) ->
            if (symbol in missing) resolved[symbol] = quote
        }
        if (resolved.isEmpty()) {
            throw fallbackResult.exceptionOrNull()
                ?: remoteResult.exceptionOrNull()
                ?: IllegalStateException("no quotes are currently available")
        }
        return resolved
    }
}
