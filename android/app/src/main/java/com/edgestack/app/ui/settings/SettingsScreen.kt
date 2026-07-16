package com.edgestack.app.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
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
import com.edgestack.app.data.local.Settings
import com.edgestack.app.data.local.SettingsStore
import com.edgestack.app.data.repo.SyncRepository
import com.edgestack.app.domain.model.RiskProfileV2
import kotlinx.coroutines.launch

class SettingsViewModel(
    private val settingsStore: SettingsStore,
    private val syncRepo: SyncRepository,
) : ViewModel() {
    var settings by mutableStateOf(Settings()); private set
    var urlDraft by mutableStateOf(""); private set
    var targetVolDraft by mutableStateOf("12"); private set
    var drawdownDraft by mutableStateOf("15"); private set
    var leverageDraft by mutableStateOf("1"); private set
    var fundingDraft by mutableStateOf("200"); private set
    var stockCapDraft by mutableStateOf("3"); private set
    var sectorCapDraft by mutableStateOf("20"); private set
    var status by mutableStateOf(""); private set
    private var initialized = false

    init {
        viewModelScope.launch {
            settingsStore.settings.collect {
                settings = it
                if (!initialized) {
                    urlDraft = it.baseUrl
                    targetVolDraft = (it.targetVolatility * 100).toString()
                    drawdownDraft = (it.maximumDrawdown * 100).toString()
                    leverageDraft = it.maximumGrossLeverage.toString()
                    fundingDraft = it.fundingSpreadBps.toString()
                    stockCapDraft = (it.perStockCap * 100).toString()
                    sectorCapDraft = (it.sectorCap * 100).toString()
                    initialized = true
                }
            }
        }
    }

    fun editUrl(value: String) { urlDraft = value }
    fun editTargetVol(value: String) { targetVolDraft = value }
    fun editDrawdown(value: String) { drawdownDraft = value }
    fun editLeverage(value: String) { leverageDraft = value }
    fun editFunding(value: String) { fundingDraft = value }
    fun editStockCap(value: String) { stockCapDraft = value }
    fun editSectorCap(value: String) { sectorCapDraft = value }

    fun save() = viewModelScope.launch {
        val profile = runCatching {
            RiskProfileV2(
                targetVolatility = targetVolDraft.toDouble() / 100,
                maximumDrawdown = drawdownDraft.toDouble() / 100,
                maximumGrossLeverage = leverageDraft.toDouble(),
                fundingSpreadBps = fundingDraft.toDouble(),
                perStockCap = stockCapDraft.toDouble() / 100,
                sectorCap = sectorCapDraft.toDouble() / 100,
            )
        }.getOrElse {
            status = "Enter valid numbers."
            return@launch
        }
        settingsStore.setBaseUrl(urlDraft.trim())
        settingsStore.setRiskProfile(profile)
        status = "Risk profile saved; refresh the server preview to apply sizing."
    }

    fun testConnection() = viewModelScope.launch {
        status = syncRepo.testConnection().fold({ "OK: $it" }, { "failed: ${it.message}" })
    }

    fun syncNow() = viewModelScope.launch {
        status = syncRepo.syncAll().fold({ "synced: $it" }, { "sync failed: ${it.message}" })
    }
}

@Composable
fun SettingsScreen(vm: SettingsViewModel) {
    Column(
        Modifier.fillMaxSize().padding(12.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("PC server", style = MaterialTheme.typography.titleMedium)
        OutlinedTextField(
            value = vm.urlDraft,
            onValueChange = vm::editUrl,
            label = { Text("Base URL, e.g. http://192.168.1.20:8000") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
        )
        Text("Risk sizing (alpha selection is unchanged)", style = MaterialTheme.typography.titleMedium)
        NumberField("Target volatility (%)", vm.targetVolDraft, vm::editTargetVol)
        NumberField("Maximum drawdown (%)", vm.drawdownDraft, vm::editDrawdown)
        NumberField("Maximum gross leverage (0–5x)", vm.leverageDraft, vm::editLeverage)
        NumberField("Funding spread (bps)", vm.fundingDraft, vm::editFunding)
        NumberField("Per-stock cap after leverage (%)", vm.stockCapDraft, vm::editStockCap)
        NumberField("Sector cap after leverage (%)", vm.sectorCapDraft, vm::editSectorCap)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = vm::save) { Text("Save") }
            OutlinedButton(onClick = vm::testConnection) { Text("Test") }
            OutlinedButton(onClick = vm::syncNow) { Text("Sync & preview") }
        }
        if (vm.status.isNotBlank()) Text(vm.status, color = Color.Gray)
        Text("Notifications", style = MaterialTheme.typography.titleMedium)
        Text(
            "Calendar, sniper, and on-device leverage alerts are disabled. Notifications are " +
                "generated only when canonical status, targets, freshness, or risk state changes.",
            style = MaterialTheme.typography.bodySmall,
        )
        Text("Research and paper trading only", style = MaterialTheme.typography.titleMedium)
        Text(
            "Custom profiles require the stateless preview endpoint. Offline mode displays the " +
                "last server/default result and never computes a replacement signal.",
            style = MaterialTheme.typography.bodySmall,
            color = Color.Gray,
        )
    }
}

@Composable
private fun NumberField(label: String, value: String, onChange: (String) -> Unit) {
    OutlinedTextField(
        value = value,
        onValueChange = onChange,
        label = { Text(label) },
        modifier = Modifier.fillMaxWidth(),
        singleLine = true,
    )
}
