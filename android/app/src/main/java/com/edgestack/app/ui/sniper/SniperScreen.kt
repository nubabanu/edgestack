package com.edgestack.app.ui.sniper

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
import com.edgestack.app.data.repo.SniperRepository
import com.edgestack.app.data.repo.SyncRepository
import com.edgestack.app.domain.model.SniperCandidateV2
import com.edgestack.app.domain.model.SniperPlanV2
import kotlinx.coroutines.launch

class SniperViewModel(
    private val repository: SniperRepository,
    private val syncRepository: SyncRepository,
) : ViewModel() {
    var plan by mutableStateOf<SniperPlanV2?>(repository.load()); private set
    var accountEquity by mutableStateOf(plan?.accountEquity?.toString() ?: "100000")
    var maxTolerableLoss by mutableStateOf(plan?.maxTolerableLoss?.toString() ?: "250")
    var vehicle by mutableStateOf(plan?.requestedVehicle ?: "SPY")
    var message by mutableStateOf(""); private set
    var loading by mutableStateOf(false); private set

    fun refresh() = request { syncRepository.latestSniper() }

    fun preview() {
        val equity = accountEquity.toDoubleOrNull()
        val loss = maxTolerableLoss.toDoubleOrNull()
        if (equity == null || loss == null) {
            message = "Enter valid numeric equity and loss values."
            return
        }
        request { syncRepository.previewSniper(equity, loss, vehicle) }
    }

    private fun request(call: suspend () -> Result<SniperPlanV2>) {
        loading = true
        viewModelScope.launch {
            call().fold(
                onSuccess = {
                    plan = it
                    vehicle = it.requestedVehicle
                    message = "Server shadow plan ${it.session} loaded."
                },
                onFailure = {
                    message = "Offline: ${it.message}. Showing the last server plan if available."
                },
            )
            loading = false
        }
    }

    companion object {
        val VEHICLES = listOf("SPY", "USMV", "SPLV", "XLV", "XLP")
    }
}

@Composable
fun SniperScreen(vm: SniperViewModel) {
    val plan = vm.plan
    LazyColumn(
        Modifier.fillMaxSize().padding(horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        item {
            Text("Sniper — staged shadow plan", style = MaterialTheme.typography.titleLarge)
            Text(
                "Rare-trade, loss-first research. Nothing on this screen creates a live order " +
                    "or changes canonical portfolio weights.",
                style = MaterialTheme.typography.bodySmall,
                color = Color.Gray,
            )
            OutlinedTextField(
                value = vm.accountEquity,
                onValueChange = { vm.accountEquity = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Account equity") },
                singleLine = true,
            )
            OutlinedTextField(
                value = vm.maxTolerableLoss,
                onValueChange = { vm.maxTolerableLoss = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Maximum tolerable modeled loss") },
                supportingText = { Text("Sizing uses the adverse 5th-percentile or a −4% fallback.") },
                singleLine = true,
            )
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                SniperViewModel.VEHICLES.forEach { symbol ->
                    AssistChip(
                        onClick = { vm.vehicle = symbol },
                        label = { Text(if (vm.vehicle == symbol) "✓ $symbol" else symbol) },
                    )
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = vm::preview, enabled = !vm.loading) {
                    Text(if (vm.loading) "Checking…" else "Preview sizing")
                }
                Button(onClick = vm::refresh, enabled = !vm.loading) { Text("Refresh default") }
            }
            if (vm.message.isNotBlank()) {
                Text(vm.message, style = MaterialTheme.typography.labelSmall)
            }
        }
        if (plan == null) {
            item {
                Card {
                    Text(
                        "No cached sniper plan. Configure the server URL and refresh.",
                        Modifier.padding(12.dp),
                    )
                }
            }
        } else {
            item {
                Card {
                    Column(
                        Modifier.fillMaxWidth().padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(4.dp),
                    ) {
                        Text("Stage 1 — available in shadow", style = MaterialTheme.typography.titleMedium)
                        Text(
                            "${plan.requestedVehicle} • equity ${money(plan.accountEquity)} • " +
                                "loss budget ${money(plan.maxTolerableLoss)}",
                        )
                        Text(
                            if (plan.stage1PromotionSatisfied) {
                                "Compatible Stage 1 sleeve found; candidates still remain paper-only here."
                            } else {
                                "No promoted Stage 1 sleeve: zero canonical weight."
                            },
                            style = MaterialTheme.typography.labelSmall,
                            color = Color.Gray,
                        )
                    }
                }
            }
            items(plan.stage1Candidates) { candidate -> CandidateCard(candidate) }
            item { Text("Filters and confirmations", style = MaterialTheme.typography.titleMedium) }
            items(plan.overlays) { overlay ->
                Card {
                    Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                        Text("${overlay.strategyId} • ${overlay.state}")
                        Text(overlay.effect)
                        Text("Threshold: ${overlay.threshold}", style = MaterialTheme.typography.labelSmall)
                        overlay.warning?.let { Text("⚠ $it", style = MaterialTheme.typography.bodySmall) }
                        Text(
                            "Cannot initiate: ${if (overlay.canInitiate) "POLICY ERROR" else "yes"}",
                            style = MaterialTheme.typography.labelSmall,
                        )
                    }
                }
            }
            item { Text("Stage 2 — blocked", style = MaterialTheme.typography.titleMedium) }
            items(plan.stage2Candidates) { candidate -> CandidateCard(candidate) }
            item { Text("Ranked policy", style = MaterialTheme.typography.titleMedium) }
            items(plan.policyRanking) { policy ->
                Card {
                    Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                        Text("#${policy.rank} ${policy.strategyId} • ${policy.activation}")
                        Text("${policy.conviction} • ${policy.role}", style = MaterialTheme.typography.labelSmall)
                        Text(policy.rule, style = MaterialTheme.typography.bodySmall)
                        Text(policy.reason, style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                    }
                }
            }
            item {
                Card {
                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text("Hard exclusions", style = MaterialTheme.typography.titleMedium)
                        Text(plan.excludedStrategyIds.joinToString())
                        Text("These strategies cannot create candidates, overlays, or weights.")
                        plan.warnings.forEach { Text("⚠ $it", style = MaterialTheme.typography.bodySmall) }
                        Text(plan.disclaimer, style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                    }
                }
            }
        }
    }
}

@Composable
private fun CandidateCard(candidate: SniperCandidateV2) {
    Card {
        Column(Modifier.fillMaxWidth().padding(10.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(candidate.strategyId, style = MaterialTheme.typography.titleMedium)
                AssistChip(onClick = {}, label = { Text(candidate.status) })
            }
            if (candidate.componentTriggers.isNotEmpty()) {
                Text("Triggered by ${candidate.componentTriggers.joinToString()}")
            }
            candidate.entryWindow?.let { Text("Entry: $it") }
            Text("Exit: ${candidate.exitRule}")
            candidate.sizing?.let { sizing ->
                Text(
                    "Shadow notional ${money(sizing.cappedNotional)} " +
                        "(${"%.1f".format(sizing.portfolioWeight * 100)}%) • " +
                        "about ${sizing.estimatedShares} shares",
                )
                Text(
                    "Adverse p5 ${"%.2f".format(sizing.adverseMoveP05 * 100)}% • " +
                        "${sizing.resolvedSignalOutcomes} resolved samples",
                    style = MaterialTheme.typography.labelSmall,
                )
                Text("⚠ ${sizing.warning}", style = MaterialTheme.typography.bodySmall)
            }
            candidate.vetoReasons.forEach { Text("Veto/block: $it") }
            Text(
                "Evidence ${candidate.evidence.evidenceGrade} • observations " +
                    "${candidate.evidence.observations} • ESS " +
                    "${"%.1f".format(candidate.evidence.effectiveSampleSize)}",
                style = MaterialTheme.typography.labelSmall,
            )
            Text(
                if (!candidate.actionable && candidate.paperOnly) "SHADOW / PAPER ONLY" else "POLICY ERROR",
                style = MaterialTheme.typography.labelSmall,
                color = Color.Gray,
            )
        }
    }
}

private fun money(value: Double): String = "\$${"%,.0f".format(value)}"
