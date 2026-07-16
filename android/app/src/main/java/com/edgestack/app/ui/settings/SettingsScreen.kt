package com.edgestack.app.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
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
import com.edgestack.app.data.local.Settings
import com.edgestack.app.data.local.SettingsStore
import com.edgestack.app.data.repo.SyncRepository
import kotlinx.coroutines.launch

class SettingsViewModel(
    private val settingsStore: SettingsStore,
    private val syncRepo: SyncRepository,
) : ViewModel() {

    var settings by mutableStateOf(Settings()); private set
    var urlDraft by mutableStateOf(""); private set
    var status by mutableStateOf(""); private set

    init {
        viewModelScope.launch {
            settingsStore.settings.collect {
                settings = it
                if (urlDraft.isBlank()) urlDraft = it.baseUrl
            }
        }
    }

    fun editUrl(v: String) { urlDraft = v }

    fun saveUrl() = viewModelScope.launch {
        settingsStore.setBaseUrl(urlDraft.trim())
        status = "server URL saved"
    }

    fun testConnection() = viewModelScope.launch {
        status = syncRepo.testConnection().fold({ "OK: $it" }, { "failed: ${it.message}" })
    }

    fun syncNow() = viewModelScope.launch {
        status = syncRepo.syncAll().fold({ "synced: $it" }, { "sync failed: ${it.message}" })
    }

    fun toggle(type: String, enabled: Boolean) = viewModelScope.launch {
        settingsStore.setAlert(type, enabled)
    }
}

@Composable
fun SettingsScreen(vm: SettingsViewModel, onTestNotification: () -> Unit) {
    val s = vm.settings
    Column(
        modifier = Modifier.fillMaxSize().padding(12.dp).verticalScroll(rememberScrollState()),
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
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = vm::saveUrl) { Text("Save") }
            OutlinedButton(onClick = vm::testConnection) { Text("Test connection") }
            OutlinedButton(onClick = vm::syncNow) { Text("Sync now") }
        }
        if (vm.status.isNotBlank()) {
            Text(vm.status, style = MaterialTheme.typography.labelMedium, color = Color.Gray)
        }

        Text("Alerts (fire 15:45 New York time)", style = MaterialTheme.typography.titleMedium)
        AlertToggle("Buy-at-close reminder (turn-of-month sessions)",
            s.alertsBuyAtClose) { vm.toggle("buyAtClose", it) }
        AlertToggle("Turn-of-month window start", s.alertsTurnOfMonth) {
            vm.toggle("turnOfMonth", it) }
        AlertToggle("September de-risk (last August session)", s.alertsSeptember) {
            vm.toggle("september", it) }
        AlertToggle("Feb-1 sniper (last January session)", s.alertsFebSniper) {
            vm.toggle("febSniper", it) }
        AlertToggle("November worst-day flat (6th trading day)", s.alertsNovWorstDay) {
            vm.toggle("novWorstDay", it) }
        OutlinedButton(onClick = onTestNotification) { Text("Send test notification") }

        Text("About", style = MaterialTheme.typography.titleMedium)
        Text(
            "EdgeStack companion. Data: bundled seed + Yahoo daily bars + your " +
                "PC-hosted research API. The overlay and all alerts implement the " +
                "validated calendar/risk rules from the EdgeStack research campaign.",
            style = MaterialTheme.typography.bodySmall, color = Color.Gray,
        )
    }
}

@Composable
private fun AlertToggle(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
        Text(label, style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.weight(1f))
        Spacer(Modifier.padding(4.dp))
        Switch(checked = checked, onCheckedChange = onChange)
    }
}
