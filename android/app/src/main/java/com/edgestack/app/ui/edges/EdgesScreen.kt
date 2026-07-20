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
import androidx.compose.material3.LinearProgressIndicator
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
import com.edgestack.app.data.repo.ResearchRepository
import com.edgestack.app.data.repo.SyncRepository
import com.edgestack.app.domain.model.BacktestRunV2
import com.edgestack.app.domain.model.EdgeSummaryV2
import com.edgestack.app.domain.model.ResearchSnapshotV1
import com.edgestack.app.ui.components.Refreshable
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

class EdgesViewModel(
    repository: EdgesRepository,
    researchRepository: ResearchRepository,
    private val syncRepository: SyncRepository,
) : ViewModel() {
    var edges by mutableStateOf(repository.loadCatalog()); private set
    var monitoring by mutableStateOf(repository.loadMonitoring()); private set
    var research by mutableStateOf(researchRepository.load()); private set
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
            val updates = mutableListOf<String>()
            syncRepository.research().fold(
                onSuccess = {
                    research = it
                    updates += "Edge Factory updated"
                },
                onFailure = {
                    updates += "factory offline (${it.message})"
                },
            )
            syncRepository.edges().fold(
                onSuccess = {
                    edges = it
                    updates += "${it.size} catalog edges"
                },
                onFailure = {
                    updates += "catalog offline (${it.message})"
                },
            )
            syncRepository.monitoringEdges().onSuccess { monitoring = it }
            syncRepository.backtests().onSuccess { backtests = it }
            message = updates.joinToString(" • ") + ". Cached results remain available offline."
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
            Text("Edge Lab", style = MaterialTheme.typography.titleLarge)
            Text(
                "Server-owned acquisition, bounded research, shadow paper books, and canonical " +
                    "growth diagnostics. The device displays and caches results; it never " +
                    "computes signals or submits orders.",
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
        vm.research.growth?.let { growth ->
            item {
                Card {
                    Column(
                        Modifier.padding(10.dp),
                        verticalArrangement = Arrangement.spacedBy(3.dp),
                    ) {
                        Text("Canonical action: ${growth.action}", style = MaterialTheme.typography.titleMedium)
                        Text(
                            "Evidence ${growth.evidenceState} • leverage " +
                                "${"%.2f".format(growth.effectiveLeverage)}x • quarter-Kelly " +
                                "${"%.2f".format(growth.quarterKellyLimit)}x",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        Text(
                            "Expected log growth ${"%+.2f%%".format(growth.expectedLogGrowth * 100)} " +
                                "(lower 95% ${"%+.2f%%".format(growth.expectedLogGrowthLower95 * 100)})",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        Text(
                            "Binding: ${growth.bindingConstraints.joinToString().ifBlank { "none" }}",
                            style = MaterialTheme.typography.labelSmall,
                        )
                        growth.comparatorLogGrowth.toList().sortedBy { it.first }.forEach {
                            Text(
                                "vs ${it.first}: ${"%+.2f%%".format(it.second * 100)} log growth",
                                style = MaterialTheme.typography.labelSmall,
                            )
                        }
                        growth.constraintLimits.toList().sortedBy { it.second }.forEach {
                            Text("${it.first}: ${"%.2f".format(it.second)}x", style = MaterialTheme.typography.labelSmall)
                        }
                    }
                }
            }
        }
        vm.research.overview?.let { overview ->
            item {
                val worker = overview.worker
                Card {
                    Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                        Text("Worker ${worker.state}", style = MaterialTheme.typography.titleMedium)
                        Text(
                            "${worker.maxWorkers} workers • ${worker.processPriority} priority • " +
                                "${"%.2f".format(worker.storageUsedGb)} / " +
                                "${"%.0f".format(worker.storageCapGb)} GB",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        LinearProgressIndicator(
                            progress = {
                                (worker.storageUsedGb / worker.storageCapGb).coerceIn(0.0, 1.0).toFloat()
                            },
                            modifier = Modifier.fillMaxWidth(),
                        )
                        Text(
                            "Queue: ${overview.acquisitionJobCounts.entries.joinToString { "${it.key} ${it.value}" }}",
                            style = MaterialTheme.typography.labelSmall,
                        )
                        Text(
                            "Registered proposals: ${overview.proposalCount} • fresh/blind: " +
                                "${overview.freshProposalCount}",
                            style = MaterialTheme.typography.labelSmall,
                        )
                        overview.providerHealth.forEach {
                            Text("${it.key}: ${it.value}", style = MaterialTheme.typography.labelSmall)
                        }
                    }
                }
            }
        }
        if (vm.research.campaigns.isNotEmpty()) {
            item { Text("Campaign funnel", style = MaterialTheme.typography.titleMedium) }
            items(vm.research.campaigns) { campaign ->
                Card {
                    Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text(campaign.name, style = MaterialTheme.typography.titleSmall)
                        Text(
                            "${campaign.lifecycle} • ${campaign.completedTrials}/${campaign.trialCount} trials",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        Text(campaign.nextAction, style = MaterialTheme.typography.labelSmall)
                        campaign.failureReasons.take(2).forEach {
                            Text("Deficit: $it", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                        }
                        campaign.metrics.entries.sortedBy { it.key }.take(8).forEach {
                            Text(
                                "${it.key.replace('_', ' ')}: ${render(it.value)}",
                                style = MaterialTheme.typography.labelSmall,
                            )
                        }
                    }
                }
            }
        }
        val openGaps = vm.research.overview?.evidenceGaps.orEmpty()
            .filterNot { it.state == "READY" }
        if (openGaps.isNotEmpty()) {
            item { Text("Acquisition progress", style = MaterialTheme.typography.titleMedium) }
            items(openGaps.take(20)) { gap ->
                Card {
                    Column(Modifier.padding(10.dp)) {
                        Text("${gap.dataset} ${gap.frequency} • ${gap.state}")
                        Text(
                            "${gap.observedObservations}/${gap.requiredObservations} minimum observations • " +
                                "${gap.symbols.size} symbols • ${gap.provider ?: "no free provider"}",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        val progress = if (gap.requiredObservations > 0) {
                            gap.observedObservations.toFloat() / gap.requiredObservations
                        } else 0f
                        LinearProgressIndicator(
                            progress = { progress.coerceIn(0f, 1f) },
                            modifier = Modifier.fillMaxWidth(),
                        )
                        Text(gap.nextAction, style = MaterialTheme.typography.labelSmall)
                    }
                }
            }
        }
        if (vm.research.strategies.isNotEmpty()) {
            item { Text("Independent paper shadows", style = MaterialTheme.typography.titleMedium) }
            items(vm.research.strategies) { shadow ->
                Card {
                    Column(Modifier.padding(10.dp)) {
                        Text(shadow.strategyId, style = MaterialTheme.typography.titleSmall)
                        Text(
                            "${shadow.status} • ${shadow.sessions} sessions • ${shadow.trades} trades • " +
                                "${"%.1f".format(shadow.effectiveResolvedOutcomes)} resolved • " +
                                "net ${"%+.2f%%".format(shadow.netReturn * 100)}",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        shadow.benchmarkReturns.forEach {
                            Text("vs ${it.key}: ${"%+.2f%%".format(it.value * 100)}", style = MaterialTheme.typography.labelSmall)
                        }
                    }
                }
            }
        }
        item { Text("Validated edge catalog", style = MaterialTheme.typography.titleMedium) }
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
