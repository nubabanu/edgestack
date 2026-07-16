package com.edgestack.app.ui.board

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
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
import com.edgestack.app.data.repo.BoardRepository
import com.edgestack.app.data.repo.SyncRepository
import com.edgestack.app.domain.model.Board
import com.edgestack.app.domain.model.BoardRow
import com.edgestack.app.domain.model.QuoteStatus
import com.edgestack.app.ui.theme.Accent
import com.edgestack.app.ui.theme.AccentAmber
import com.edgestack.app.ui.theme.AccentRed
import kotlinx.coroutines.launch

class BoardViewModel(
    private val boardRepo: BoardRepository,
    private val syncRepo: SyncRepository,
) : ViewModel() {

    var board by mutableStateOf<Board?>(null); private set
    var quotes by mutableStateOf<Map<String, Double>>(emptyMap()); private set
    var status by mutableStateOf(""); private set
    var refreshing by mutableStateOf(false); private set

    init { load() }

    fun load() {
        board = runCatching { boardRepo.load() }.getOrNull()
    }

    fun refreshQuotes() {
        val b = board ?: return
        refreshing = true
        viewModelScope.launch {
            quotes = boardRepo.liveQuotes(b)
            status = if (quotes.isEmpty()) "quote refresh failed (offline?)"
                     else "live quotes: ${quotes.size}/${b.rows.size}"
            refreshing = false
        }
    }

    fun sync() {
        refreshing = true
        viewModelScope.launch {
            status = syncRepo.syncAll().fold({ "synced ($it)" }, { "sync failed: ${it.message}" })
            load()
            refreshing = false
        }
    }

    fun quoteStatus(row: BoardRow): Pair<QuoteStatus, Double?> {
        val last = quotes[row.symbol] ?: return QuoteStatus.OPEN to null
        return when {
            last <= row.stop -> QuoteStatus.STOPPED to last
            last >= row.target -> QuoteStatus.TARGET_HIT to last
            else -> QuoteStatus.OPEN to last
        }
    }
}

@Composable
fun BoardScreen(vm: BoardViewModel) {
    val board = vm.board
    Column(modifier = Modifier.fillMaxSize().padding(horizontal = 12.dp)) {
        if (board == null) {
            Text("No board available", style = MaterialTheme.typography.titleMedium)
            return@Column
        }
        Row(
            modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Board ${board.asOf}", style = MaterialTheme.typography.titleMedium)
            AssistChip(onClick = {}, label = { Text(board.regime.trend) })
            AssistChip(onClick = {}, label = { Text("vol ${board.regime.vol}") })
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = vm::refreshQuotes, enabled = !vm.refreshing) {
                Text("Refresh quotes")
            }
            OutlinedButton(onClick = vm::sync, enabled = !vm.refreshing) {
                Text("Sync from PC")
            }
        }
        if (vm.status.isNotBlank()) {
            Text(vm.status, style = MaterialTheme.typography.labelSmall, color = Color.Gray)
        }
        LazyColumn(
            verticalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.padding(top = 8.dp),
        ) {
            items(board.rows) { row -> BoardRowCard(row, vm) }
        }
    }
}

@Composable
private fun BoardRowCard(row: BoardRow, vm: BoardViewModel) {
    val (qs, last) = vm.quoteStatus(row)
    Card {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(row.symbol, style = MaterialTheme.typography.titleLarge)
                Spacer(Modifier.width(10.dp))
                Column {
                    Text("close %.2f".format(row.close),
                        style = MaterialTheme.typography.labelMedium)
                    if (last != null) {
                        Text("last %.2f".format(last),
                            style = MaterialTheme.typography.labelMedium,
                            color = if (last >= row.close) Accent else AccentRed)
                    }
                }
                Spacer(Modifier.weight(1f))
                when (qs) {
                    QuoteStatus.STOPPED -> Text("STOPPED", color = AccentRed,
                        style = MaterialTheme.typography.titleSmall)
                    QuoteStatus.TARGET_HIT -> Text("TARGET", color = Accent,
                        style = MaterialTheme.typography.titleSmall)
                    QuoteStatus.OPEN -> Text("conv %.0f".format(row.conviction),
                        color = AccentAmber, style = MaterialTheme.typography.titleSmall)
                }
            }
            LinearProgressIndicator(
                progress = { (row.conviction / 100.0).toFloat() },
                modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp),
            )
            Text(
                "E[net] 10d %+.2f%%   hit %.0f%%   edges %d/%d fam".format(
                    row.eNet10d * 100, row.hit * 100, row.nEdges, row.families),
                style = MaterialTheme.typography.labelMedium,
            )
            Text(
                "stop %.2f   target %.2f".format(row.stop, row.target),
                style = MaterialTheme.typography.labelMedium, color = Color.Gray,
            )
        }
    }
}
