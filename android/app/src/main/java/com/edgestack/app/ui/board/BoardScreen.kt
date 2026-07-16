package com.edgestack.app.ui.board

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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.edgestack.app.data.repo.RecommendationRepository
import com.edgestack.app.data.repo.SyncRepository
import com.edgestack.app.domain.model.CanonicalRecommendationBundleV2
import kotlinx.coroutines.launch

class BoardViewModel(
    private val recommendationRepo: RecommendationRepository,
    private val syncRepo: SyncRepository,
) : ViewModel() {
    var bundle by mutableStateOf<CanonicalRecommendationBundleV2?>(null); private set
    var status by mutableStateOf(""); private set
    var refreshing by mutableStateOf(false); private set

    init { load() }

    fun load() {
        bundle = runCatching { recommendationRepo.loadBundle() }.getOrNull()
    }

    fun sync() {
        refreshing = true
        viewModelScope.launch {
            status = syncRepo.syncAll().fold({ "synced: $it" }, { "offline: ${it.message}" })
            load()
            refreshing = false
        }
    }
}

@Composable
fun BoardScreen(vm: BoardViewModel) {
    val bundle = vm.bundle
    Column(Modifier.fillMaxSize().padding(horizontal = 12.dp)) {
        if (bundle == null) {
            Text("No canonical recommendation available")
            return@Column
        }
        val base = bundle.baseRecommendation
        Row(
            Modifier.fillMaxWidth().padding(vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                "Portfolio ${bundle.session}",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.weight(1f),
            )
            Button(onClick = vm::sync, enabled = !vm.refreshing) {
                Text(if (vm.refreshing) "Syncing…" else "Sync", maxLines = 1)
            }
        }
        AssistChip(onClick = {}, label = { Text(base.status) })
        if (vm.status.isNotBlank()) {
            Text(vm.status, style = MaterialTheme.typography.labelSmall, color = Color.Gray)
        }
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            item {
                Card {
                    Column(Modifier.padding(12.dp)) {
                        Text("Unlevered canonical base", style = MaterialTheme.typography.titleMedium)
                        base.unleveredBaseWeights.forEach { weight ->
                            Text("${weight.symbol}  ${"%.2f".format(weight.weight * 100)}%")
                        }
                        Text(
                            "Modeled active net return ${"%+.2f".format(base.expectedNetReturn * 100)}% " +
                                "• volatility ${"%.1f".format(base.expectedVolatility * 100)}%",
                            style = MaterialTheme.typography.labelSmall,
                            color = Color.Gray,
                        )
                    }
                }
            }
            if (base.promotedSleeves.isNotEmpty() || base.promotedCompoundSleeves.isNotEmpty()) {
                item { Text("Promoted sleeves", style = MaterialTheme.typography.titleMedium) }
                items(base.promotedSleeves + base.promotedCompoundSleeves) { sleeve ->
                    Card {
                        Column(Modifier.padding(10.dp)) {
                            Text(sleeve.sleeveId)
                            Text(
                                "${sleeve.family} • ${sleeve.horizonSessions} sessions • " +
                                    "${sleeve.evidenceGrade}",
                                style = MaterialTheme.typography.labelSmall,
                            )
                        }
                    }
                }
            }
            if (base.watchlist.isNotEmpty()) {
                item { Text("Watchlist only — zero weight", style = MaterialTheme.typography.titleMedium) }
                items(base.watchlist) { idea ->
                    Card {
                        Column(Modifier.padding(10.dp)) {
                            Text("${idea.symbol} • ${idea.evidenceGrade}")
                            Text(idea.thesis, style = MaterialTheme.typography.bodySmall)
                            Text(
                                "${idea.prospectiveSessions}/252 sessions; " +
                                    "ESS ${"%.0f".format(idea.effectiveResolvedOutcomes)}/100",
                                style = MaterialTheme.typography.labelSmall,
                                color = Color.Gray,
                            )
                            Text(idea.zeroWeightReason, style = MaterialTheme.typography.labelSmall)
                        }
                    }
                }
            }
            item {
                Text(
                    "Fresh ${base.freshness.isFresh} • data ${bundle.dataVersion.take(10)} • " +
                        "artifact ${bundle.artifactVersion.take(10)}",
                    style = MaterialTheme.typography.labelSmall,
                    color = Color.Gray,
                )
            }
        }
    }
}
