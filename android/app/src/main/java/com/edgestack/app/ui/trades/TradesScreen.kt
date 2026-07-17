package com.edgestack.app.ui.trades

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.edgestack.app.data.remote.YahooChartClient
import com.edgestack.app.data.repo.PositionsRepository
import com.edgestack.app.data.repo.SyncRepository
import com.edgestack.app.domain.model.PaperResponse
import com.edgestack.app.domain.model.TrackedPosition
import com.edgestack.app.domain.pnlAt
import com.edgestack.app.ui.components.Refreshable
import com.edgestack.app.ui.components.Sparkline
import com.edgestack.app.ui.theme.Accent
import com.edgestack.app.ui.theme.AccentRed
import java.time.LocalDate
import kotlinx.coroutines.launch

class TradesViewModel(
    private val syncRepo: SyncRepository,
    private val positionsRepo: PositionsRepository,
    private val quotesClient: YahooChartClient,
) : ViewModel() {
    var paper by mutableStateOf<PaperResponse?>(null); private set
    var status by mutableStateOf(""); private set
    var loading by mutableStateOf(false); private set
    var positions by mutableStateOf(positionsRepo.load()); private set
    var quotes by mutableStateOf<Map<String, Double>>(emptyMap()); private set
    var symbolDraft by mutableStateOf("")
    var quantityDraft by mutableStateOf("")
    var entryPriceDraft by mutableStateOf("")
    var positionMessage by mutableStateOf(""); private set

    init { refresh() }

    fun refresh() {
        loading = true
        viewModelScope.launch {
            syncRepo.paper().onSuccess { paper = it }
                .onFailure { status = "Offline: ${it.message}" }
            refreshQuotes()
            loading = false
        }
    }

    private suspend fun refreshQuotes() {
        val symbols = positions.map { it.symbol }.distinct()
        if (symbols.isEmpty()) return
        runCatching { quotesClient.latestQuotes(symbols) }
            .onSuccess { quotes = it }
            .onFailure { positionMessage = "Quotes unavailable: ${it.message}" }
    }

    fun addPosition() {
        val symbol = symbolDraft.trim().uppercase()
        val quantity = quantityDraft.toDoubleOrNull()
        val entry = entryPriceDraft.toDoubleOrNull()
        if (symbol.isBlank() || quantity == null || quantity <= 0 || entry == null || entry <= 0) {
            positionMessage = "Enter a symbol, positive quantity, and positive entry price."
            return
        }
        positionsRepo.add(
            TrackedPosition(
                symbol = symbol,
                entryPrice = entry,
                quantity = quantity,
                entryDate = LocalDate.now().toString(),
            ),
        )
        positions = positionsRepo.load()
        symbolDraft = ""
        quantityDraft = ""
        entryPriceDraft = ""
        positionMessage = ""
        viewModelScope.launch { refreshQuotes() }
    }

    fun removePosition(position: TrackedPosition) {
        positionsRepo.remove(position.symbol, position.entryDate)
        positions = positionsRepo.load()
    }
}

@Composable
fun TradesScreen(vm: TradesViewModel) {
    Refreshable(refreshing = vm.loading, onRefresh = vm::refresh) {
        TradesContent(vm)
    }
}

@Composable
private fun TradesContent(vm: TradesViewModel) {
    Column(
        Modifier.fillMaxSize().padding(12.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("Canonical paper account", style = MaterialTheme.typography.titleMedium)
        Text(
            "Orders follow server target weights at the next open. Stops, targets, calendar " +
                "rules, and current fundamentals are not applied.",
            style = MaterialTheme.typography.bodySmall,
            color = Color.Gray,
        )
        Button(onClick = vm::refresh) { Text("Refresh from PC") }
        PaperSection(vm)
        MyPositionsSection(vm)
    }
}

@Composable
private fun PaperSection(vm: TradesViewModel) {
    val paper = vm.paper
    if (paper == null) {
        Text(vm.status.ifBlank { "No paper state cached." })
        return
    }
    val state = paper.state
    Card {
        Column(Modifier.padding(12.dp)) {
            Text(
                "equity ${"%.2f".format(state.currentEquity)}  •  " +
                    "cash ${"%.2f".format(state.cash)}",
                style = MaterialTheme.typography.titleMedium,
            )
            Text(
                "${state.positions.size} positions • ${state.fills.size} actual fills • " +
                    "as of ${state.lastSession ?: "pending"}",
            )
            if (paper.equityHistory.size >= 2) {
                Sparkline(values = paper.equityHistory.map { it.equity })
            }
        }
    }
    state.positions.forEach { position ->
        Card {
            Row(Modifier.padding(10.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(position.symbol, style = MaterialTheme.typography.titleMedium)
                Text(
                    "${"%.4f".format(position.quantity)} @ " +
                        "${"%.2f".format(position.averageFillPrice)}",
                )
                Text("last ${"%.2f".format(position.lastPrice)}")
            }
        }
    }
    state.realizedReturns.takeLast(5).forEach { realized ->
        Text(
            "${realized.session}: ${"%+.3f".format(realized.actualFillReturn * 100)}% • " +
                "cost ${"%.2f".format(realized.transactionCosts)} • " +
                "div ${"%.2f".format(realized.dividendCash)} • " +
                "fund ${"%+.2f".format(realized.financingCashFlow)}",
            style = MaterialTheme.typography.labelSmall,
        )
    }
}

@Composable
private fun MyPositionsSection(vm: TradesViewModel) {
    Text("My positions", style = MaterialTheme.typography.titleMedium)
    Text(
        "Your own tracked entries with delayed/indicative device quotes — " +
            "not the canonical server view.",
        style = MaterialTheme.typography.bodySmall,
        color = Color.Gray,
    )
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        OutlinedTextField(
            value = vm.symbolDraft,
            onValueChange = { vm.symbolDraft = it },
            modifier = Modifier.weight(1.1f),
            label = { Text("Symbol") },
            singleLine = true,
            keyboardOptions = KeyboardOptions(
                capitalization = KeyboardCapitalization.Characters,
                autoCorrect = false,
            ),
        )
        OutlinedTextField(
            value = vm.quantityDraft,
            onValueChange = { vm.quantityDraft = it },
            modifier = Modifier.weight(0.9f),
            label = { Text("Qty") },
            singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        )
        OutlinedTextField(
            value = vm.entryPriceDraft,
            onValueChange = { vm.entryPriceDraft = it },
            modifier = Modifier.weight(1f),
            label = { Text("Entry") },
            singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        )
    }
    Button(onClick = vm::addPosition) { Text("Add position") }
    if (vm.positionMessage.isNotBlank()) {
        Text(vm.positionMessage, style = MaterialTheme.typography.labelSmall)
    }
    if (vm.positions.isEmpty()) {
        Text(
            "No tracked positions yet — add one above.",
            style = MaterialTheme.typography.bodySmall,
            color = Color.Gray,
        )
    }
    vm.positions.forEach { position ->
        val pnl = position.pnlAt(vm.quotes[position.symbol])
        Card {
            Row(
                Modifier.fillMaxWidth().padding(10.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(Modifier.weight(1f)) {
                    Text(
                        "${position.symbol}  ${"%.4f".format(position.quantity)} @ " +
                            "${"%.2f".format(position.entryPrice)}",
                        style = MaterialTheme.typography.titleSmall,
                    )
                    Text(
                        "since ${position.entryDate}" +
                            (position.stop?.let { " • stop ${"%.2f".format(it)}" } ?: "") +
                            (position.target?.let { " • target ${"%.2f".format(it)}" } ?: ""),
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.Gray,
                    )
                    if (pnl == null) {
                        Text("last — • P&L —", style = MaterialTheme.typography.bodySmall)
                    } else {
                        Text(
                            "last ${"%.2f".format(vm.quotes[position.symbol])} • " +
                                "${"%+.2f".format(pnl.profit)} " +
                                "(${"%+.2f".format(pnl.profitPercent)}%)",
                            style = MaterialTheme.typography.bodySmall,
                            color = if (pnl.profit >= 0) Accent else AccentRed,
                        )
                    }
                }
                IconButton(onClick = { vm.removePosition(position) }) {
                    Icon(Icons.Filled.Delete, contentDescription = "Remove ${position.symbol}")
                }
            }
        }
    }
}
