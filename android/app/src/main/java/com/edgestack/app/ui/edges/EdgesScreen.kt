package com.edgestack.app.ui.edges

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import com.edgestack.app.data.repo.EdgesRepository
import com.edgestack.app.domain.model.Edge
import com.edgestack.app.ui.theme.Accent
import com.edgestack.app.ui.theme.AccentRed

class EdgesViewModel(edgesRepo: EdgesRepository) : ViewModel() {

    val edges: List<Edge> = runCatching { edgesRepo.load().edges }.getOrDefault(emptyList())
    val families: List<String> = edges.map { it.identity.family }.distinct().sorted()

    var familyFilter by mutableStateOf<String?>(null)
    var selected by mutableStateOf<Edge?>(null)

    fun visible(): List<Edge> =
        edges.filter { familyFilter == null || it.identity.family == familyFilter }
            .sortedBy { it.stats.qValue ?: 1.0 }
}

@Composable
fun EdgesScreen(vm: EdgesViewModel) {
    val detail = vm.selected
    if (detail != null) {
        EdgeDetail(detail) { vm.selected = null }
        return
    }
    Column(modifier = Modifier.fillMaxSize().padding(horizontal = 12.dp)) {
        Text(
            "Frozen validated edges (${vm.edges.size})",
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(vertical = 8.dp),
        )
        LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            items(vm.families) { fam ->
                FilterChip(
                    selected = vm.familyFilter == fam,
                    onClick = { vm.familyFilter = if (vm.familyFilter == fam) null else fam },
                    label = { Text(fam) },
                )
            }
        }
        LazyColumn(
            verticalArrangement = Arrangement.spacedBy(6.dp),
            modifier = Modifier.padding(top = 8.dp),
        ) {
            items(vm.visible()) { edge ->
                Card(modifier = Modifier.clickable { vm.selected = edge }) {
                    Column(Modifier.padding(10.dp)) {
                        Text(edge.identity.name.ifBlank { edge.identity.edgeId },
                            style = MaterialTheme.typography.bodyMedium, maxLines = 2)
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                "${edge.identity.family} · ${edge.identity.direction} · " +
                                    "${edge.identity.holdingHorizon}d",
                                style = MaterialTheme.typography.labelSmall,
                                color = Color.Gray,
                            )
                            Spacer(Modifier.weight(1f))
                            val net = edge.stats.netMeanReturn
                            if (net != null) {
                                Text("%+.2f%%".format(net * 100),
                                    style = MaterialTheme.typography.labelMedium,
                                    color = if (net >= 0) Accent else AccentRed)
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun EdgeDetail(edge: Edge, onBack: () -> Unit) {
    Column(modifier = Modifier.fillMaxSize().padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("< back", color = Accent,
            modifier = Modifier.clickable { onBack() }.padding(vertical = 4.dp))
        Text(edge.identity.name.ifBlank { edge.identity.edgeId },
            style = MaterialTheme.typography.titleMedium)
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                listOf(
                    "edge_id" to edge.identity.edgeId,
                    "family" to edge.identity.family,
                    "direction" to edge.identity.direction,
                    "horizon" to "${edge.identity.holdingHorizon} sessions",
                    "status" to (edge.currentStatus ?: "?"),
                    "net mean return" to edge.stats.netMeanReturn?.let {
                        "%+.3f%%".format(it * 100) },
                    "q-value" to edge.stats.qValue?.let { "%.4f".format(it) },
                    "deflated Sharpe" to edge.stats.deflatedSharpe?.let { "%.2f".format(it) },
                    "sample size" to edge.stats.sampleSize?.toString(),
                    "P(net>0)" to edge.stats.probPositive?.let { "%.0f%%".format(it * 100) },
                ).forEach { (k, v) ->
                    if (v != null) {
                        Row {
                            Text(k, color = Color.Gray,
                                style = MaterialTheme.typography.labelMedium)
                            Spacer(Modifier.weight(1f))
                            Text(v, style = MaterialTheme.typography.labelMedium)
                        }
                    }
                }
            }
        }
    }
}
