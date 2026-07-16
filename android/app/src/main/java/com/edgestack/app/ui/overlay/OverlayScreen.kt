package com.edgestack.app.ui.overlay

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.FilterChip
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
import com.edgestack.app.data.local.SettingsStore
import com.edgestack.app.data.repo.OverlayRepository
import com.edgestack.app.domain.model.OverlayState
import com.edgestack.app.ui.components.ExposureDial
import com.edgestack.app.ui.components.Sparkline
import kotlinx.coroutines.launch

class OverlayViewModel(
    private val overlayRepo: OverlayRepository,
    private val settingsStore: SettingsStore,
) : ViewModel() {

    var state by mutableStateOf<OverlayState?>(null); private set
    var base by mutableStateOf(1.0); private set
    var loading by mutableStateOf(false); private set
    var error by mutableStateOf(""); private set

    init {
        viewModelScope.launch {
            base = settingsStore.current().baseLeverage
            refresh()
        }
    }

    fun setBaseLeverage(v: Double) {
        base = v
        viewModelScope.launch {
            settingsStore.setBaseLeverage(v)
            recomputeOnly()
        }
    }

    private suspend fun recomputeOnly() {
        state = overlayRepo.state(base)
        if (state == null) error = "need SPY history — connect to the internet once"
    }

    fun refresh() {
        loading = true
        error = ""
        viewModelScope.launch {
            state = overlayRepo.state(base, forceRefresh = true)
            if (state == null) error = "SPY fetch failed (offline?)"
            loading = false
        }
    }
}

@Composable
fun OverlayScreen(vm: OverlayViewModel) {
    val s = vm.state
    Column(
        modifier = Modifier.fillMaxSize().padding(12.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("Calendar-leverage overlay", style = MaterialTheme.typography.titleMedium)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(selected = vm.base == 1.0, onClick = { vm.setBaseLeverage(1.0) },
                label = { Text("base 1.0x") })
            FilterChip(selected = vm.base == 1.3, onClick = { vm.setBaseLeverage(1.3) },
                label = { Text("base 1.3x") })
            Button(onClick = vm::refresh, enabled = !vm.loading) { Text("Refresh") }
        }
        if (vm.error.isNotBlank()) {
            Text(vm.error, color = MaterialTheme.colorScheme.error)
        }
        if (s != null) {
            ExposureDial(value = s.appliedL)
            Text("as of ${s.date} (exposure applied to the NEXT session)",
                style = MaterialTheme.typography.labelMedium, color = Color.Gray)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                s.firedRules.forEach { AssistChip(onClick = {}, label = { Text(it) }) }
                if (s.firedRules.isEmpty()) {
                    AssistChip(onClick = {}, label = { Text("No special rules today") })
                }
            }
            Card {
                Column(Modifier.padding(12.dp)) {
                    val smaText = s.sma200?.let { sma ->
                        val pct = (s.spyClose / sma - 1) * 100
                        "SPY %.2f vs 200-DMA %.2f (%+.1f%%)".format(s.spyClose, sma, pct)
                    } ?: "200-DMA warming up"
                    Text(smaText, style = MaterialTheme.typography.bodyMedium)
                    Text(
                        s.vol20?.let { "20d realized vol %.1f%%".format(it * 100) }
                            ?: "vol warming up",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }
            Text("Exposure — last ${s.history.size} sessions",
                style = MaterialTheme.typography.labelMedium)
            Sparkline(values = s.history.map { it.second },
                modifier = Modifier.fillMaxWidth())
        } else if (!vm.loading) {
            Text("No overlay state yet.", color = Color.Gray)
        }
    }
}
