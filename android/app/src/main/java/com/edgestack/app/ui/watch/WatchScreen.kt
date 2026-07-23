package com.edgestack.app.ui.watch

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AssistChip
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
import com.edgestack.app.data.repo.WatchersRepository
import com.edgestack.app.domain.model.OilSurgeV1
import com.edgestack.app.domain.model.PaperBookV1
import com.edgestack.app.domain.model.TrancheSymbolV1
import com.edgestack.app.domain.model.TrancheWatchV1
import com.edgestack.app.ui.components.Refreshable
import kotlinx.coroutines.launch

class WatchViewModel(
    private val repository: WatchersRepository,
    private val syncRepository: SyncRepository,
) : ViewModel() {
    var tranche by mutableStateOf<TrancheWatchV1?>(repository.loadTranche()); private set
    var oil by mutableStateOf<OilSurgeV1?>(repository.loadOil()); private set
    var message by mutableStateOf(""); private set
    var loading by mutableStateOf(false); private set

    fun refresh() {
        loading = true
        viewModelScope.launch {
            val notes = mutableListOf<String>()
            syncRepository.trancheWatch().fold(
                onSuccess = { tranche = it },
                onFailure = { notes += "tranche: ${it.message}" },
            )
            syncRepository.oilSurge().fold(
                onSuccess = { oil = it },
                onFailure = { notes += "oil: ${it.message}" },
            )
            message = if (notes.isEmpty()) "Watchers refreshed from server."
            else "Offline or missing: ${notes.joinToString("; ")}. Showing cached state."
            loading = false
        }
    }
}

@Composable
fun WatchScreen(vm: WatchViewModel) {
    Refreshable(refreshing = vm.loading, onRefresh = vm::refresh) {
        LazyColumn(
            Modifier.fillMaxSize().padding(horizontal = 12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            item {
                Text("Watchers — PC-side alert programs", style = MaterialTheme.typography.titleLarge)
                Text(
                    "Read-only mirror of the tranche and oil-surge watchers running on the PC. " +
                        "Alerts originate there (Telegram + toast); nothing here trades.",
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.Gray,
                )
                if (vm.message.isNotBlank()) {
                    Text(vm.message, style = MaterialTheme.typography.labelSmall)
                }
            }
            item { OilCard(vm.oil) }
            item { Text("Tranche triggers", style = MaterialTheme.typography.titleMedium) }
            val tranche = vm.tranche
            if (tranche == null) {
                item {
                    Card {
                        Text(
                            "No tranche watcher data yet. Configure the server URL and refresh.",
                            Modifier.padding(12.dp),
                        )
                    }
                }
            } else {
                item {
                    Text(
                        "Run ${tranche.runDate ?: "?"}" +
                            (if (tranche.stale) " — STALE" else "") +
                            (tranche.breadth?.let {
                                " • breadth ${it.count}/${it.total}"
                            } ?: ""),
                        style = MaterialTheme.typography.labelSmall,
                        color = if (tranche.stale) MaterialTheme.colorScheme.error else Color.Gray,
                    )
                }
                items(tranche.symbols) { symbol -> SymbolCard(symbol) }
                item { PaperCard("Tranche paper book", tranche.paperBook) }
            }
            item {
                val oil = vm.oil
                PaperCard("Oil paper book", oil?.paperBook)
            }
        }
    }
}

@Composable
private fun OilCard(oil: OilSurgeV1?) {
    Card {
        Column(Modifier.fillMaxWidth().padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Oil surge watch", style = MaterialTheme.typography.titleMedium)
                AssistChip(onClick = {}, label = { Text(oil?.state?.phase ?: "no data") })
            }
            if (oil == null) {
                Text("No oil surge data yet. Refresh with the server reachable.")
                return@Card
            }
            val episode = oil.state.episode
            if (episode != null) {
                Text(
                    "Shock ${episode.shockDate} (${"%+.1f%%".format(episode.shockRet * 100)}) • " +
                        "session ${episode.sessions}/10 • dips ${episode.dips}",
                )
                episode.postShockHigh?.let {
                    Text("Post-shock high ${"%.2f".format(it)}", style = MaterialTheme.typography.bodySmall)
                }
            } else {
                Text("No active surge regime.", style = MaterialTheme.typography.bodySmall)
            }
            oil.latestCloses.forEach { (symbol, close) ->
                Text("$symbol ${close.close} (${close.date})", style = MaterialTheme.typography.bodySmall)
            }
            Text(
                if (oil.dipTicketsEnabled) {
                    "Dip entries: paper-ticketed (study PASS)."
                } else {
                    "Dip alerts DISPLAY-ONLY — study verdict " +
                        "${oil.studyVerdict?.verdict ?: "not run"}; no entry rule validated."
                },
                style = MaterialTheme.typography.labelSmall,
                color = if (oil.dipTicketsEnabled) Color.Unspecified else MaterialTheme.colorScheme.error,
            )
            if (oil.stale) {
                Text("STALE — last EOD pass is old.", color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.labelSmall)
            }
        }
    }
}

@Composable
private fun SymbolCard(symbol: TrancheSymbolV1) {
    Card {
        Column(Modifier.fillMaxWidth().padding(10.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(symbol.symbol, style = MaterialTheme.typography.titleMedium)
                symbol.close?.let { Text("close ${"%.2f".format(it)}") }
                symbol.go?.let {
                    Text(
                        "GO $it/100 (display-only)",
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.Gray,
                    )
                }
            }
            symbol.triggers().forEach { (label, state) ->
                Text(
                    (if (state.fired) "FIRED  " else "wait   ") + "$label — ${state.detail}",
                    style = MaterialTheme.typography.bodySmall,
                    color = if (state.fired) MaterialTheme.colorScheme.primary else Color.Gray,
                )
            }
            symbol.cal?.let { Text("CAL: $it", style = MaterialTheme.typography.bodySmall) }
            if (symbol.windowOpen) {
                Text("Post-earnings window OPEN", style = MaterialTheme.typography.bodySmall)
            }
            symbol.earnings?.let {
                Text("earnings $it", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
            }
        }
    }
}

@Composable
private fun PaperCard(title: String, book: PaperBookV1?) {
    Card {
        Column(Modifier.fillMaxWidth().padding(10.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            val trades = book?.trades.orEmpty()
            if (trades.isEmpty()) {
                Text("No paper trades.", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
            } else {
                trades.forEach { trade ->
                    Text(
                        "${trade.symbol} ${trade.trigger} EUR ${"%.0f".format(trade.eur)} " +
                            (trade.fillPrice?.let { "@ ${"%.2f".format(it)} (${trade.fillDate})" }
                                ?: "(pending fill)"),
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
            Text(
                "Paper autopilot only — no live orders ever.",
                style = MaterialTheme.typography.labelSmall,
                color = Color.Gray,
            )
        }
    }
}
