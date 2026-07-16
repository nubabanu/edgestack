package com.edgestack.app.ui.instrument

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
import com.edgestack.app.data.repo.InstrumentAnalysisRepository
import com.edgestack.app.data.repo.SyncRepository
import com.edgestack.app.domain.model.EdgeEffectV2
import com.edgestack.app.domain.model.InstrumentAnalysisV2
import com.edgestack.app.domain.model.TimingWindowV2
import kotlinx.coroutines.launch

class InstrumentViewModel(
    private val repository: InstrumentAnalysisRepository,
    private val syncRepository: SyncRepository,
) : ViewModel() {
    var symbol by mutableStateOf("")
    var intendedEntry by mutableStateOf("")
    var analysis by mutableStateOf<InstrumentAnalysisV2?>(repository.loadLast()); private set
    var message by mutableStateOf(""); private set
    var loading by mutableStateOf(false); private set

    fun analyze() {
        if (symbol.isBlank()) {
            message = "Enter a ticker, GOLD, OIL, WTI, BRENT, or SILVER."
            return
        }
        loading = true
        message = ""
        viewModelScope.launch {
            syncRepository.analyzeInstrument(
                symbol,
                intendedEntry.trim().ifBlank { null },
            ).fold(
                onSuccess = {
                    analysis = it
                    symbol = it.resolution.requestedSymbol
                    message = "Analyzed against canonical ${it.asOf.take(10)} evidence."
                },
                onFailure = { message = "Analysis unavailable: ${it.message}" },
            )
            loading = false
        }
    }
}

@Composable
fun InstrumentScreen(vm: InstrumentViewModel) {
    val analysis = vm.analysis
    LazyColumn(
        Modifier.fillMaxSize().padding(horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        item {
            Text("Instrument timing", style = MaterialTheme.typography.titleLarge)
            Text(
                "The server may abstain. Only frozen promoted windows are actionable.",
                style = MaterialTheme.typography.bodySmall,
                color = Color.Gray,
            )
            OutlinedTextField(
                value = vm.symbol,
                onValueChange = { vm.symbol = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Ticker or commodity (for example AAPL, GLD, GOLD, OIL)") },
                singleLine = true,
            )
            OutlinedTextField(
                value = vm.intendedEntry,
                onValueChange = { vm.intendedEntry = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Optional intended entry with offset") },
                supportingText = { Text("Example: 2026-07-20T09:30:00-04:00") },
                singleLine = true,
            )
            Button(onClick = vm::analyze, enabled = !vm.loading) {
                Text(if (vm.loading) "Analyzing…" else "Analyze")
            }
            if (vm.message.isNotBlank()) {
                Text(vm.message, style = MaterialTheme.typography.labelSmall)
            }
        }
        if (analysis != null) {
            item { AnalysisSummary(analysis) }
            items(analysis.horizonAnalyses) { horizon ->
                Card {
                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text(horizon.horizon, style = MaterialTheme.typography.titleMedium)
                            AssistChip(
                                onClick = {},
                                label = { Text(if (horizon.actionable) "PROMOTED" else "RESEARCH") },
                            )
                        }
                        Text("Resolution: ${horizon.dataResolution}")
                        Window("Best", horizon.bestWindow)
                        Window("Worst", horizon.worstWindow)
                        horizon.intendedEntryAssessment?.let { Text("Your time: $it") }
                        horizon.warning?.let {
                            Text("⚠ $it", style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
            if (analysis.tailwinds.isNotEmpty()) {
                item { Text("Tailwinds", style = MaterialTheme.typography.titleMedium) }
                items(analysis.tailwinds) { Effect(it) }
            }
            if (analysis.headwinds.isNotEmpty()) {
                item { Text("Headwinds", style = MaterialTheme.typography.titleMedium) }
                items(analysis.headwinds) { Effect(it) }
            }
            if (analysis.mixedEffects.isNotEmpty()) {
                item { Text("Mixed and policy context", style = MaterialTheme.typography.titleMedium) }
                items(analysis.mixedEffects) { Effect(it) }
            }
            if (analysis.news.isNotEmpty()) {
                item { Text("Frozen news context", style = MaterialTheme.typography.titleMedium) }
                items(analysis.news) { item ->
                    Card {
                        Column(Modifier.padding(10.dp)) {
                            Text(item.headline)
                            Text(
                                "${item.source} • ${item.sentimentLabel} • fresh ${item.isFresh}",
                                style = MaterialTheme.typography.labelSmall,
                            )
                            Text(item.warning, style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
            item {
                Card {
                    Column(Modifier.padding(12.dp)) {
                        Text("What to watch", style = MaterialTheme.typography.titleMedium)
                        analysis.whatToWatch.forEach { Text("• $it") }
                        Text("Current year", style = MaterialTheme.typography.titleMedium)
                        analysis.currentYearNotes.forEach { Text("• $it") }
                        analysis.warnings.forEach {
                            Text("⚠ $it", style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun AnalysisSummary(analysis: InstrumentAnalysisV2) {
    Card {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                "${analysis.resolution.requestedSymbol} → ${analysis.resolution.resolvedSymbol}",
                style = MaterialTheme.typography.titleMedium,
            )
            analysis.resolution.proxyFor?.let { Text("Tradable proxy for $it") }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                AssistChip(onClick = {}, label = { Text(analysis.status) })
                AssistChip(onClick = {}, label = { Text(analysis.overallRating) })
            }
            Text(
                if (analysis.alignment.alignedTrade) {
                    "ALL PROMOTED HORIZONS ALIGNED — paper-only"
                } else {
                    "Not all stars aligned: missing ${analysis.alignment.missingInputs.joinToString()}"
                },
            )
            Text(analysis.alignment.explanation, style = MaterialTheme.typography.bodySmall)
            Text(
                "Canonical weight ${"%.2f".format(analysis.canonicalPortfolioWeight * 100)}% • " +
                    "price ${analysis.currentPrice?.let { "%.2f".format(it) } ?: "unavailable"}",
                style = MaterialTheme.typography.labelSmall,
            )
        }
    }
}

@Composable
private fun Window(name: String, window: TimingWindowV2?) {
    if (window == null) {
        Text("$name: unavailable — no compatible evidence")
        return
    }
    Text("$name: ${window.entryWindow}", style = MaterialTheme.typography.titleSmall)
    Text("Exit: ${window.exitWindow}")
    Text(
        "Net mean ${window.expectedNetReturn?.let { "%+.2f".format(it * 100) + "%" } ?: "n/a"} • " +
            "95% low ${window.lower95?.let { "%+.2f".format(it * 100) + "%" } ?: "n/a"} • " +
            "ESS ${"%.0f".format(window.effectiveSampleSize)} • ${window.evidenceGrade}",
        style = MaterialTheme.typography.labelSmall,
    )
    window.whatInvalidatesIt.forEach {
        Text("Caution: $it", style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun Effect(effect: EdgeEffectV2) {
    Card {
        Column(Modifier.fillMaxWidth().padding(10.dp)) {
            Text(
                "${effect.family}${if (effect.compound) " • compound" else ""} • " +
                    "${effect.direction} • ${effect.evidenceGrade}",
            )
            Text(effect.observation)
            Text(
                "Positive ${"%+.2f".format(effect.positiveContribution * 100)}% • " +
                    "negative/uncertainty ${"%+.2f".format(effect.negativeContribution * 100)}% • " +
                    "conservative net ${"%+.2f".format(effect.netContribution * 100)}%",
                style = MaterialTheme.typography.labelSmall,
            )
            if (effect.protectiveAvoidanceValue > 0) {
                Text(
                    "Modeled avoidance value ${"%.2f".format(effect.protectiveAvoidanceValue * 100)}%",
                    style = MaterialTheme.typography.labelSmall,
                )
            }
            Text("Downside of this effect: ${effect.adverseCounterEffect}")
            Text("Protective/opposite value: ${effect.protectiveCounterEffect}")
            Text("Invalidation: ${effect.invalidation}", style = MaterialTheme.typography.labelSmall)
        }
    }
}
