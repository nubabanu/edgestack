package com.edgestack.app.ui.trades

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
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
import com.edgestack.app.data.remote.YahooChartClient
import com.edgestack.app.data.repo.PositionsRepository
import com.edgestack.app.data.repo.SyncRepository
import com.edgestack.app.domain.TradingCalendar
import com.edgestack.app.domain.model.PaperResponse
import com.edgestack.app.domain.model.TrackedPosition
import com.edgestack.app.ui.components.Sparkline
import com.edgestack.app.ui.theme.Accent
import com.edgestack.app.ui.theme.AccentRed
import kotlinx.coroutines.launch
import java.time.LocalDate

class TradesViewModel(
    private val positionsRepo: PositionsRepository,
    private val syncRepo: SyncRepository,
    private val yahoo: YahooChartClient,
    private val calendar: TradingCalendar,
) : ViewModel() {

    var positions by mutableStateOf<List<TrackedPosition>>(emptyList()); private set
    var quotes by mutableStateOf<Map<String, Double>>(emptyMap()); private set
    var paper by mutableStateOf<PaperResponse?>(null); private set
    var status by mutableStateOf(""); private set

    init { reload() }

    fun reload() {
        positions = positionsRepo.load()
        refreshQuotes()
    }

    fun refreshQuotes() {
        viewModelScope.launch {
            if (positions.isNotEmpty()) {
                quotes = yahoo.latestQuotes(positions.map { it.symbol }.distinct())
            }
            syncRepo.paper().onSuccess { paper = it }
                .onFailure { status = "paper: ${it.message}" }
        }
    }

    fun add(symbol: String, entry: Double, qty: Double,
            stop: Double?, target: Double?) {
        positionsRepo.add(TrackedPosition(
            symbol = symbol.uppercase().trim(), entryPrice = entry,
            quantity = qty, entryDate = LocalDate.now().toString(),
            stop = stop, target = target))
        reload()
    }

    fun close(p: TrackedPosition) {
        positionsRepo.remove(p.symbol, p.entryDate)
        reload()
    }

    fun sessionsHeld(p: TrackedPosition): Int =
        calendar.sessionsBetween(LocalDate.parse(p.entryDate), LocalDate.now())
}

@Composable
fun TradesScreen(vm: TradesViewModel) {
    Column(
        modifier = Modifier.fillMaxSize().padding(12.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("My positions", style = MaterialTheme.typography.titleMedium)
        AddPositionRow(onAdd = vm::add)
        if (vm.positions.isEmpty()) {
            Text("No tracked positions. Add what you actually bought — the app " +
                "then watches stops, targets and the 10-session clock.",
                style = MaterialTheme.typography.bodySmall, color = Color.Gray)
        }
        vm.positions.forEach { p ->
            val last = vm.quotes[p.symbol]
            Card {
                Column(Modifier.padding(10.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(p.symbol, style = MaterialTheme.typography.titleMedium)
                        Spacer(Modifier.width(8.dp))
                        Text("${p.quantity} @ %.2f".format(p.entryPrice),
                            style = MaterialTheme.typography.labelMedium)
                        Spacer(Modifier.weight(1f))
                        if (last != null) {
                            val pnl = (last - p.entryPrice) * p.quantity
                            val pct = last / p.entryPrice - 1
                            Text("%+.2f (%.1f%%)".format(pnl, pct * 100),
                                color = if (pnl >= 0) Accent else AccentRed,
                                style = MaterialTheme.typography.titleSmall)
                        }
                    }
                    val held = vm.sessionsHeld(p)
                    val leftTxt = (p.horizonSessions - held).let {
                        if (it > 0) "$it sessions to time-exit" else "TIME EXIT DUE"
                    }
                    val stopTxt = p.stop?.let { s ->
                        last?.let { "stop %.2f (%.1f%% away)".format(
                            s, (it / s - 1) * 100) } ?: "stop %.2f".format(s)
                    } ?: "no stop"
                    Text("$stopTxt   ·   $leftTxt",
                        style = MaterialTheme.typography.labelSmall,
                        color = if (last != null && p.stop != null && last <= p.stop)
                            AccentRed else Color.Gray)
                    OutlinedButton(onClick = { vm.close(p) }) { Text("Close") }
                }
            }
        }

        Text("Paper account (PC experiment)",
            style = MaterialTheme.typography.titleMedium)
        val paper = vm.paper
        if (paper == null) {
            Text("Sync with your PC to see the live paper experiment. ${vm.status}",
                style = MaterialTheme.typography.bodySmall, color = Color.Gray)
            Button(onClick = vm::refreshQuotes) { Text("Refresh") }
        } else {
            val st = paper.state
            val equity = paper.equityHistory.lastOrNull()?.equity
            Card {
                Column(Modifier.padding(10.dp)) {
                    Text("equity %.2f   cash %.2f   open %d   trades %d".format(
                        equity ?: st.cash, st.cash, st.positions.size,
                        st.trades.size), style = MaterialTheme.typography.bodyMedium)
                    val realized = st.trades.sumOf { it.netPnl }
                    Text("realized P&L %+.2f   as of ${st.lastSession ?: "-"}"
                        .format(realized),
                        style = MaterialTheme.typography.labelMedium,
                        color = if (realized >= 0) Accent else AccentRed)
                    if (paper.equityHistory.size >= 2) {
                        Sparkline(values = paper.equityHistory.map { it.equity })
                    }
                    st.positions.forEach { p ->
                        Text("${p.symbol}  ${p.quantity} @ %.2f  stop %.2f  tgt %.2f"
                            .format(p.entryPrice, p.stopPrice, p.targetPrice),
                            style = MaterialTheme.typography.labelSmall,
                            color = Color.Gray)
                    }
                }
            }
        }
    }
}

@Composable
private fun AddPositionRow(onAdd: (String, Double, Double, Double?, Double?) -> Unit) {
    var sym by androidx.compose.runtime.remember { mutableStateOf("") }
    var entry by androidx.compose.runtime.remember { mutableStateOf("") }
    var qty by androidx.compose.runtime.remember { mutableStateOf("") }
    var stop by androidx.compose.runtime.remember { mutableStateOf("") }
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            OutlinedTextField(value = sym, onValueChange = { sym = it },
                label = { Text("Symbol") }, modifier = Modifier.width(110.dp),
                singleLine = true)
            OutlinedTextField(value = entry, onValueChange = { entry = it },
                label = { Text("Entry") }, modifier = Modifier.width(100.dp),
                singleLine = true)
            OutlinedTextField(value = qty, onValueChange = { qty = it },
                label = { Text("Qty") }, modifier = Modifier.width(90.dp),
                singleLine = true)
            OutlinedTextField(value = stop, onValueChange = { stop = it },
                label = { Text("Stop") }, modifier = Modifier.width(100.dp),
                singleLine = true)
        }
        Button(onClick = {
            val e = entry.toDoubleOrNull(); val q = qty.toDoubleOrNull()
            if (sym.isNotBlank() && e != null && q != null) {
                onAdd(sym, e, q, stop.toDoubleOrNull(), null)
                sym = ""; entry = ""; qty = ""; stop = ""
            }
        }) { Text("Add position") }
    }
}
