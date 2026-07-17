package com.edgestack.app.ui.trades

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.edgestack.app.data.repo.SyncRepository
import com.edgestack.app.domain.model.PaperResponse
import com.edgestack.app.ui.components.Refreshable
import com.edgestack.app.ui.components.Sparkline
import kotlinx.coroutines.launch

class TradesViewModel(private val syncRepo: SyncRepository) : ViewModel() {
    var paper by mutableStateOf<PaperResponse?>(null); private set
    var status by mutableStateOf(""); private set
    var loading by mutableStateOf(false); private set

    init { refresh() }

    fun refresh() {
        loading = true
        viewModelScope.launch {
            syncRepo.paper().onSuccess { paper = it }
                .onFailure { status = "Offline: ${it.message}" }
            loading = false
        }
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
        val paper = vm.paper
        if (paper == null) {
            Text(vm.status.ifBlank { "No paper state cached." })
            return@Column
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
}
