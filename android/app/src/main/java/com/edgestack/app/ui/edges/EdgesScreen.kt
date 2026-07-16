package com.edgestack.app.ui.edges

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AssistChip
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
import com.edgestack.app.data.repo.EdgesRepository
import com.edgestack.app.data.repo.SyncRepository
import com.edgestack.app.domain.model.BacktestRunV2
import com.edgestack.app.domain.model.EdgeSummaryV2
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

class EdgesViewModel(
    repository: EdgesRepository,
    private val syncRepository: SyncRepository,
) : ViewModel() {
    var edges by mutableStateOf(repository.loadCatalog()); private set
    var monitoring by mutableStateOf(repository.loadMonitoring()); private set
    var backtests by mutableStateOf<List<BacktestRunV2>>(emptyList()); private set
    var message by mutableStateOf(""); private set
    var loading by mutableStateOf(false); private set

    fun refresh() {
        loading = true
        message = ""
        viewModelScope.launch {
            syncRepository.edges().fold(
                onSuccess = {
                    edges = it
                    message = "Loaded ${it.size} edges from the server."
                },
                onFailure = {
                    message = "Offline: ${it.message}. Showing the last cached catalog."
                },
            )
            syncRepository.monitoringEdges().onSuccess { monitoring = it }
            syncRepository.backtests().onSuccess { backtests = it }
            loading = false
        }
    }

    /** Deployed first, then by deflated Sharpe so the strongest evidence leads. */
    val ranked: List<EdgeSummaryV2>
        get() = edges.sortedWith(
            compareByDescending<EdgeSummaryV2> { it.status.equals("deployed", true) }
                .thenByDescending { it.deflatedSharpe ?: Double.NEGATIVE_INFINITY },
        )
}

@Composable
fun EdgesScreen(vm: EdgesViewModel) {
    LazyColumn(
        Modifier.fillMaxSize().padding(horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        item {
            Text("Edge catalog", style = MaterialTheme.typography.titleLarge)
            Text(
                "Validated edges with lifecycle status, monitoring health, and backtest runs. " +
                    "Statistics are historical evidence, not forecasts.",
                style = MaterialTheme.typography.bodySmall,
                color = Color.Gray,
            )
            Button(onClick = vm::refresh, enabled = !vm.loading) {
                Text(if (vm.loading) "Refreshing…" else "Refresh from server")
            }
            if (vm.message.isNotBlank()) {
                Text(vm.message, style = MaterialTheme.typography.labelSmall)
            }
        }
        if (vm.edges.isEmpty()) {
            item {
                Text(
                    "No cached edges yet — refresh while the server is reachable.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
        items(vm.ranked) { edge -> EdgeCard(edge) }
        vm.monitoring?.let { payload ->
            item { Text("Monitoring", style = MaterialTheme.typography.titleMedium) }
            item { MonitoringCard(payload) }
        }
        if (vm.backtests.isNotEmpty()) {
            item { Text("Backtest runs", style = MaterialTheme.typography.titleMedium) }
            items(vm.backtests) { run ->
                Card {
                    Column(Modifier.padding(10.dp)) {
                        Text(run.runId, style = MaterialTheme.typography.titleSmall)
                        Text(
                            "${run.createdAt} • ${run.costScenario}",
                            style = MaterialTheme.typography.labelSmall,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun EdgeCard(edge: EdgeSummaryV2) {
    Card {
        Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text(
                    edge.name.ifBlank { edge.edgeId },
                    style = MaterialTheme.typography.titleSmall,
                    modifier = Modifier.weight(1f),
                )
                AssistChip(onClick = {}, label = { Text(edge.status.uppercase()) })
            }
            Text(
                "${edge.family} • ${edge.direction} • horizon ${edge.horizon} sessions",
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                "Net mean ${edge.netMeanReturn?.let { "%+.3f%%".format(it * 100) } ?: "n/a"} • " +
                    "deflated Sharpe ${edge.deflatedSharpe?.let { "%.2f".format(it) } ?: "n/a"} • " +
                    "q ${edge.qValue?.let { "%.3f".format(it) } ?: "n/a"} • " +
                    "n ${edge.sampleSize ?: 0}",
                style = MaterialTheme.typography.labelSmall,
            )
        }
    }
}

@Composable
private fun MonitoringCard(payload: JsonObject) {
    Card {
        Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            payload.entries.take(40).forEach { (key, value) ->
                Text("$key: ${render(value)}", style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

private fun render(value: kotlinx.serialization.json.JsonElement): String = when (value) {
    is JsonPrimitive -> value.content
    is JsonArray -> "[${value.size} items]"
    is JsonObject -> value.entries.joinToString(prefix = "{", postfix = "}", limit = 6) {
        "${it.key}=${(it.value as? JsonPrimitive)?.content ?: "…"}"
    }
}
