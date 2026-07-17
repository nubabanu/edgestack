package com.edgestack.app.ui.edges

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.edgestack.app.data.repo.EdgesRepository
import com.edgestack.app.data.repo.SyncRepository
import com.edgestack.app.domain.model.BacktestRunV2
import com.edgestack.app.domain.model.EdgeSummaryV2
import com.edgestack.app.ui.components.Refreshable
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
    var query by mutableStateOf("")
    var statusFilter by mutableStateOf<String?>(null)
    var detailTitle by mutableStateOf<String?>(null); private set
    var detail by mutableStateOf<JsonObject?>(null); private set
    var detailError by mutableStateOf(""); private set

    fun openEdgeDetail(edge: EdgeSummaryV2) =
        openDetail(edge.name.ifBlank { edge.edgeId }) { syncRepository.edgeDetail(edge.edgeId) }

    fun openBacktestDetail(run: BacktestRunV2) =
        openDetail("Backtest ${run.runId}") { syncRepository.backtestDetail(run.runId) }

    private fun openDetail(title: String, fetch: suspend () -> Result<JsonObject>) {
        detailTitle = title
        detail = null
        detailError = ""
        viewModelScope.launch {
            fetch().fold(
                onSuccess = { detail = it },
                onFailure = { detailError = it.message ?: "unavailable" },
            )
        }
    }

    fun closeDetail() {
        detailTitle = null
        detail = null
        detailError = ""
    }

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

    /** Distinct statuses in catalog order of frequency, for the filter chips. */
    val statuses: List<String>
        get() = edges.groupingBy { it.status.uppercase() }.eachCount()
            .entries.sortedByDescending { it.value }.map { it.key }

    /** Deployed first, then by deflated Sharpe so the strongest evidence leads. */
    val ranked: List<EdgeSummaryV2>
        get() = edges
            .asSequence()
            .filter { statusFilter == null || it.status.equals(statusFilter, true) }
            .filter {
                query.isBlank() ||
                    it.name.contains(query, true) ||
                    it.family.contains(query, true) ||
                    it.direction.contains(query, true) ||
                    it.edgeId.contains(query, true)
            }
            .sortedWith(
                compareByDescending<EdgeSummaryV2> { it.status.equals("deployed", true) }
                    .thenByDescending { it.deflatedSharpe ?: Double.NEGATIVE_INFINITY },
            )
            .toList()

    companion object {
        /** Cards rendered at once; filters narrow the rest. */
        const val MAX_SHOWN = 200
    }
}

@Composable
fun EdgesScreen(vm: EdgesViewModel) {
    Refreshable(refreshing = vm.loading, onRefresh = vm::refresh) {
        EdgesList(vm)
    }
}

@Composable
private fun EdgesList(vm: EdgesViewModel) {
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
        } else {
            item {
                OutlinedTextField(
                    value = vm.query,
                    onValueChange = { vm.query = it },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Search name, family, direction") },
                    singleLine = true,
                )
                Row(
                    Modifier.horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    FilterChip(
                        selected = vm.statusFilter == null,
                        onClick = { vm.statusFilter = null },
                        label = { Text("ALL (${vm.edges.size})") },
                    )
                    vm.statuses.forEach { status ->
                        FilterChip(
                            selected = vm.statusFilter == status,
                            onClick = {
                                vm.statusFilter = if (vm.statusFilter == status) null else status
                            },
                            label = { Text(status) },
                        )
                    }
                }
            }
        }
        val shown = vm.ranked
        if (vm.edges.isNotEmpty()) {
            item {
                Text(
                    if (shown.size > EdgesViewModel.MAX_SHOWN) {
                        "Showing top ${EdgesViewModel.MAX_SHOWN} of ${shown.size} matches — " +
                            "narrow with search or a status filter."
                    } else {
                        "${shown.size} of ${vm.edges.size} edges match."
                    },
                    style = MaterialTheme.typography.labelSmall,
                    color = Color.Gray,
                )
            }
        }
        items(shown.take(EdgesViewModel.MAX_SHOWN)) { edge ->
            EdgeCard(edge, onClick = { vm.openEdgeDetail(edge) })
        }
        vm.monitoring?.let { payload ->
            item { Text("Monitoring", style = MaterialTheme.typography.titleMedium) }
            item { MonitoringCard(payload) }
        }
        if (vm.backtests.isNotEmpty()) {
            item { Text("Backtest runs", style = MaterialTheme.typography.titleMedium) }
            items(vm.backtests) { run ->
                Card(onClick = { vm.openBacktestDetail(run) }) {
                    Column(Modifier.padding(10.dp)) {
                        Text(run.runId, style = MaterialTheme.typography.titleSmall)
                        Text(
                            "${run.createdAt} • ${run.costScenario} • tap for detail",
                            style = MaterialTheme.typography.labelSmall,
                        )
                    }
                }
            }
        }
    }
    vm.detailTitle?.let { title ->
        DetailDialog(
            title = title,
            payload = vm.detail,
            error = vm.detailError,
            onClose = vm::closeDetail,
        )
    }
}

@Composable
private fun DetailDialog(
    title: String,
    payload: JsonObject?,
    error: String,
    onClose: () -> Unit,
) {
    Dialog(onDismissRequest = onClose) {
        Card {
            Column(Modifier.padding(14.dp)) {
                Text(title, style = MaterialTheme.typography.titleMedium)
                when {
                    error.isNotBlank() -> Text(
                        "Unavailable: $error",
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(vertical = 8.dp),
                    )
                    payload == null -> Text(
                        "Loading…",
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(vertical = 8.dp),
                    )
                    else -> LazyColumn(
                        Modifier.fillMaxWidth().heightIn(max = 480.dp).padding(top = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(3.dp),
                    ) {
                        items(payload.entries.toList()) { (key, value) ->
                            Text(
                                "$key: ${render(value)}",
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                    }
                }
                TextButton(onClick = onClose, modifier = Modifier.align(Alignment.End)) {
                    Text("Close")
                }
            }
        }
    }
}

@Composable
private fun EdgeCard(edge: EdgeSummaryV2, onClick: () -> Unit) {
    Card(onClick = onClick) {
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
