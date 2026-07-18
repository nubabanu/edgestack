package com.edgestack.app.ui.instrument

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.edgestack.app.data.repo.InstrumentAnalysisRepository
import com.edgestack.app.data.repo.OilDecisionRepository
import com.edgestack.app.data.repo.SyncRepository
import com.edgestack.app.domain.MacroEventBook
import com.edgestack.app.domain.model.EdgeEffectV2
import com.edgestack.app.domain.model.MacroEvent
import com.edgestack.app.domain.model.InstrumentAnalysisV2
import com.edgestack.app.domain.model.OilBrokerQuoteV2
import com.edgestack.app.domain.model.OilDecisionRequestV2
import com.edgestack.app.domain.model.OilDecisionSnapshotV2
import com.edgestack.app.domain.model.PatternLeaderBoardV2
import com.edgestack.app.domain.model.TimingWindowV2
import com.edgestack.app.work.WorkScheduler
import java.time.LocalDate
import java.time.Instant
import java.time.OffsetDateTime
import kotlinx.coroutines.launch

class InstrumentViewModel(
    private val repository: InstrumentAnalysisRepository,
    private val oilRepository: OilDecisionRepository,
    private val syncRepository: SyncRepository,
    private val macroEvents: MacroEventBook,
) : ViewModel() {

    /** Macro releases landing on the intended-entry date, if it parses. */
    val intendedEntryEvents: List<MacroEvent>
        get() {
            val match = Regex("\\d{4}-\\d{2}-\\d{2}").find(intendedEntry) ?: return emptyList()
            val date = runCatching { LocalDate.parse(match.value) }.getOrNull()
                ?: return emptyList()
            return macroEvents.on(date)
        }
    var symbol by mutableStateOf("")
    var intendedEntry by mutableStateOf("")
    var analysis by mutableStateOf<InstrumentAnalysisV2?>(repository.loadLast()); private set
    var oilDecision by mutableStateOf<OilDecisionSnapshotV2?>(oilRepository.load()); private set
    var leaders by mutableStateOf<PatternLeaderBoardV2?>(null); private set
    var message by mutableStateOf(""); private set
    var loading by mutableStateOf(false); private set
    var oilBid by mutableStateOf("")
    var oilAsk by mutableStateOf("")
    var oilOfferedLeverage by mutableStateOf("10")
    var oilModeledLeverage by mutableStateOf("10")
    var oilEventFlags by mutableStateOf<Set<String>>(emptySet())

    fun toggleOilEvent(flag: String) {
        oilEventFlags = if (flag in oilEventFlags) oilEventFlags - flag else oilEventFlags + flag
    }

    fun checkOil() {
        val entry = runCatching { OffsetDateTime.parse(intendedEntry.trim()) }.getOrNull()
        val bid = oilBid.toDoubleOrNull()
        val ask = oilAsk.toDoubleOrNull()
        val offered = oilOfferedLeverage.toDoubleOrNull()
        val modeled = oilModeledLeverage.toDoubleOrNull()
        if (entry == null || bid == null || ask == null || offered == null || modeled == null) {
            message = "OIL check needs a timezone-aware entry plus numeric bid, ask, and leverage."
            return
        }
        if (bid <= 0 || ask <= bid || offered !in 1.0..10.0 || modeled !in 1.0..offered) {
            message = "OIL quote/leverage invalid: ask > bid > 0 and modeled ≤ offered ≤ 10."
            return
        }
        loading = true
        message = ""
        viewModelScope.launch {
            syncRepository.oilDecision(
                OilDecisionRequestV2(
                    intendedEntryAt = entry.toString(),
                    quote = OilBrokerQuoteV2(
                        observedAt = Instant.now().toString(),
                        bid = bid,
                        ask = ask,
                        offeredLeverage = offered,
                    ),
                    modeledLeverage = modeled,
                    eventFlags = oilEventFlags.sorted(),
                ),
            ).fold(
                onSuccess = {
                    oilDecision = it
                    symbol = "OIL"
                    message = "Paper-only OIL snapshot ${it.snapshotId.take(8)}: ${it.status}."
                },
                onFailure = { message = "OIL decision unavailable: ${it.message}" },
            )
            loading = false
        }
    }

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

    fun recheck() {
        loading = true
        viewModelScope.launch {
            syncRepository.recheckInstrument().fold(
                onSuccess = {
                    analysis = it.analysis
                    message = if (it.recommendationStillHolds) {
                        "Server recheck: selected timing still holds."
                    } else {
                        "Server recheck changed: ${it.changes.joinToString()}"
                    }
                },
                onFailure = { message = "Recheck unavailable: ${it.message}" },
            )
            loading = false
        }
    }

    var leaderSymbolsDraft by mutableStateOf("")
    var leaderHorizon by mutableStateOf("WEEK")

    fun scanLeaders() {
        val custom = leaderSymbolsDraft
            .split(',', ' ', ';')
            .map { it.trim().uppercase() }
            .filter { it.isNotBlank() }
            .distinct()
            .take(50)
        val symbols = custom.ifEmpty { QUICK_SYMBOLS }
        loading = true
        viewModelScope.launch {
            syncRepository.patternLeaders(symbols, horizon = leaderHorizon).fold(
                onSuccess = {
                    leaders = it
                    message = "Scanned ${it.searchedSymbols.size} symbols at ${it.horizon} " +
                        "horizon; research-only."
                },
                onFailure = { message = "Pattern scan unavailable: ${it.message}" },
            )
            loading = false
        }
    }

    companion object {
        val QUICK_SYMBOLS = listOf(
            "SPY", "QQQ", "GLD", "USO", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
        )
    }
}

@Composable
fun InstrumentScreen(vm: InstrumentViewModel) {
    val analysis = vm.analysis
    val oilDecision = vm.oilDecision
    val context = LocalContext.current
    LaunchedEffect(analysis?.analysisId) {
        WorkScheduler.scheduleInstrumentRecheck(context, analysis?.recheckPlan?.nextCheckAt)
    }
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
                keyboardOptions = KeyboardOptions(
                    capitalization = KeyboardCapitalization.Characters,
                    autoCorrect = false,
                ),
            )
            OutlinedTextField(
                value = vm.intendedEntry,
                onValueChange = { vm.intendedEntry = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Optional intended entry with offset") },
                supportingText = {
                    Text("Date only: 2026-07-20; or hour: 2026-07-20T09:30:00-04:00")
                },
                singleLine = true,
                keyboardOptions = KeyboardOptions(autoCorrect = false),
                isError = vm.intendedEntryEvents.any { it.type != "EIA" },
            )
            vm.intendedEntryEvents.forEach { event ->
                Text(
                    "⚡ ${event.label} at ${event.timeEt} ET lands on that date.",
                    style = MaterialTheme.typography.labelSmall,
                )
            }
            Text("Quick instruments", style = MaterialTheme.typography.labelMedium)
            Row(
                Modifier.horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                InstrumentViewModel.QUICK_SYMBOLS.forEach { quick ->
                    AssistChip(onClick = { vm.symbol = quick }, label = { Text(quick) })
                }
            }
            Button(onClick = vm::analyze, enabled = !vm.loading) {
                Text(if (vm.loading) "Analyzing…" else "Analyze")
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = vm::recheck, enabled = !vm.loading && analysis != null) {
                    Text("Recheck now")
                }
                Button(onClick = vm::scanLeaders, enabled = !vm.loading) {
                    Text("Pattern leaders")
                }
            }
            OutlinedTextField(
                value = vm.leaderSymbolsDraft,
                onValueChange = { vm.leaderSymbolsDraft = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Pattern scan symbols (comma-separated, blank = defaults)") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(
                    capitalization = KeyboardCapitalization.Characters,
                    autoCorrect = false,
                ),
            )
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                listOf("DAY", "WEEK", "MONTH").forEach { horizon ->
                    AssistChip(
                        onClick = { vm.leaderHorizon = horizon },
                        label = {
                            Text(if (vm.leaderHorizon == horizon) "✓ $horizon" else horizon)
                        },
                    )
                }
            }
            if (vm.message.isNotBlank()) {
                Text(vm.message, style = MaterialTheme.typography.labelSmall)
            }
            OilTicketControls(vm)
        }
        if (oilDecision != null) item { OilDecisionCard(oilDecision) }
        if (analysis != null) {
            item { AnalysisSummary(analysis) }
            if (analysis.chosenTimeRatings.isNotEmpty()) {
                item { Text("Your chosen time", style = MaterialTheme.typography.titleMedium) }
                items(analysis.chosenTimeRatings) { rating ->
                    Card {
                        Column(Modifier.padding(10.dp)) {
                            Text("${rating.resolution} • ${rating.horizon} • ${rating.rating}")
                            Text(
                                "Matched ${rating.matchedSlot ?: "none"} • win score " +
                                    (rating.score?.let { "%.1f".format(it.winScore) } ?: "n/a") +
                                    " • rank ${rating.score?.rank ?: 0}/${rating.score?.candidatesRanked ?: 0}",
                            )
                            Text(rating.recommendation, style = MaterialTheme.typography.bodySmall)
                            rating.betterAlternative?.let {
                                Text(
                                    "Better historical slot: ${it.entryWindow} • " +
                                        "score ${"%.1f".format(it.score.winScore)}",
                                )
                            }
                        }
                    }
                }
            }
            if (analysis.exitPlans.isNotEmpty()) {
                item { Text("Conditional exit map", style = MaterialTheme.typography.titleMedium) }
                items(analysis.exitPlans) { exit ->
                    Card {
                        Column(Modifier.padding(10.dp)) {
                            Text("${exit.horizon}: enter ${exit.entrySlot}")
                            Text("Preferred exit: ${exit.preferredExit ?: "unavailable"}")
                            Text(
                                "Win score ${exit.score?.let { "%.1f".format(it.winScore) } ?: "n/a"} • " +
                                    "${if (exit.actionable) "PROMOTED" else "RESEARCH"}",
                                style = MaterialTheme.typography.labelSmall,
                            )
                            Text(exit.rationale, style = MaterialTheme.typography.bodySmall)
                            exit.warning?.let { Text("⚠ $it") }
                        }
                    }
                }
            }
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
                        Text("Automatic recheck", style = MaterialTheme.typography.titleMedium)
                        Text(analysis.recheckPlan.reason)
                        Text(
                            "Next: ${analysis.recheckPlan.nextCheckAt ?: "disabled"} • " +
                                "resolution ${analysis.recheckPlan.requiredResolution ?: "n/a"}",
                        )
                    }
                }
            }
            analysis.tailwindCalendars.forEach { calendar ->
                item {
                    Text(
                        "${calendar.resolution} tailwind calendar • ${calendar.horizon}",
                        style = MaterialTheme.typography.titleMedium,
                    )
                }
                items(calendar.cells.take(12)) { cell ->
                    Card {
                        Row(
                            Modifier.fillMaxWidth().padding(8.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                        ) {
                            Text("#${cell.score.rank} ${cell.displayLabel}")
                            Text("win score ${"%.1f".format(cell.score.winScore)}")
                        }
                    }
                }
                item { Text(calendar.warning, style = MaterialTheme.typography.bodySmall) }
            }
        }
        vm.leaders?.let { board ->
            item { Text("Pattern leaders — research only", style = MaterialTheme.typography.titleMedium) }
            items(board.leaders) { leader ->
                Card {
                    Column(Modifier.padding(10.dp)) {
                        Text("#${leader.rank} ${leader.symbol}")
                        Text(
                            "${leader.strongestSlot.entryWindow} • win score " +
                                "${"%.1f".format(leader.strongestSlot.score.winScore)}",
                        )
                    }
                }
            }
            item { Text(board.warning, style = MaterialTheme.typography.bodySmall) }
        }
    }
}

@Composable
private fun OilTicketControls(vm: InstrumentViewModel) {
    Text("eToro OIL paper gate", style = MaterialTheme.typography.titleMedium)
    Text(
        "Manual ticket inputs are sent to the PC. The result never contains a live order or size.",
        style = MaterialTheme.typography.bodySmall,
        color = Color.Gray,
    )
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedTextField(
            value = vm.oilBid,
            onValueChange = { vm.oilBid = it },
            modifier = Modifier.weight(1f),
            label = { Text("OIL bid") },
            singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        )
        OutlinedTextField(
            value = vm.oilAsk,
            onValueChange = { vm.oilAsk = it },
            modifier = Modifier.weight(1f),
            label = { Text("OIL ask") },
            singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        )
    }
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedTextField(
            value = vm.oilOfferedLeverage,
            onValueChange = { vm.oilOfferedLeverage = it },
            modifier = Modifier.weight(1f),
            label = { Text("Offered leverage") },
            singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        )
        OutlinedTextField(
            value = vm.oilModeledLeverage,
            onValueChange = { vm.oilModeledLeverage = it },
            modifier = Modifier.weight(1f),
            label = { Text("Model up to") },
            singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        )
    }
    Text("Manual hard vetoes", style = MaterialTheme.typography.labelMedium)
    Row(
        Modifier.horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        OIL_EVENT_FLAGS.forEach { (flag, label) ->
            AssistChip(
                onClick = { vm.toggleOilEvent(flag) },
                label = { Text(if (flag in vm.oilEventFlags) "✓ $label" else label) },
            )
        }
    }
    Button(onClick = vm::checkOil, enabled = !vm.loading) {
        Text(if (vm.loading) "Checking…" else "Check OIL — paper only")
    }
}

@Composable
private fun OilDecisionCard(snapshot: OilDecisionSnapshotV2) {
    val statusColor = when (snapshot.status) {
        "BLOCKED" -> MaterialTheme.colorScheme.error
        "PAPER_ONLY" -> Color(0xFF8A6D00)
        else -> MaterialTheme.colorScheme.primary
    }
    Card {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("eToro OIL • ${snapshot.status}", color = statusColor,
                style = MaterialTheme.typography.titleLarge)
            Text("PAPER ONLY • actionable ${snapshot.actionable} • canonical weight 0%")
            Text(
                "${snapshot.brokerProfile.productType} • ${snapshot.brokerProfile.weeklySession} " +
                    "${snapshot.brokerProfile.marketTimezone} • break ${snapshot.brokerProfile.dailyBreak}",
                style = MaterialTheme.typography.bodySmall,
            )
            Text("Snapshot ${snapshot.snapshotId.take(12)} • next ${snapshot.nextRecheckAt ?: "manual"}")
            snapshot.decisionReasons.forEach { Text("• $it") }
            snapshot.hardBlockReasons.forEach { Text("⛔ $it", color = statusColor) }
            Text("Friction sensitivity", style = MaterialTheme.typography.titleSmall)
            snapshot.frictionSensitivity.forEach { scenario ->
                val mean = scenario.expectedNetReturn?.let { "%+.3f%%".format(it * 100) } ?: "n/a"
                val lower = scenario.lower95?.let { "%+.3f%%".format(it * 100) } ?: "n/a"
                Text(
                    "${scenario.name}: ${"%.1f".format(scenario.roundTripCostBps)} bp • " +
                        "${scenario.matchedSlot ?: "no slot"} • mean $mean • low $lower • " +
                        if (scenario.survives) "survives (still paper)" else "fails",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            Text(
                "Sources ${snapshot.sourceAlignment.state} • quote age " +
                    "${"%.0f".format(snapshot.dataFreshness.quoteAgeSeconds)}s • " +
                    "canonical match ${snapshot.dataFreshness.canonicalMatchesCatalog}",
            )
            snapshot.stressTable.firstOrNull {
                it.leverage == 10 && it.adverseMoveFraction == 0.1
            }?.let {
                Text(
                    "10× with a 10% adverse move models ${"%.0f".format(it.equityLossFraction * 100)}% " +
                        "equity loss; liquidation may occur earlier.",
                    color = MaterialTheme.colorScheme.error,
                )
            }
            snapshot.manualInputsRequired.forEach { Text("Confirm: $it") }
            snapshot.warnings.forEach { Text("⚠ $it", style = MaterialTheme.typography.bodySmall) }
            Text(snapshot.disclaimer, style = MaterialTheme.typography.labelSmall)
        }
    }
}

private val OIL_EVENT_FLAGS = listOf(
    "WEEKEND_SUPPLY_ESCALATION" to "Supply escalation",
    "SHIPPING_DISRUPTION" to "Shipping disruption",
    "WTI_ROLLOVER_EXPIRY" to "Rollover/expiry",
    "BROKER_MAINTENANCE" to "Broker maintenance",
)

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
