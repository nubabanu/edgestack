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
import com.edgestack.app.data.local.Settings
import com.edgestack.app.data.local.SettingsStore
import com.edgestack.app.data.repo.RecommendationRepository
import com.edgestack.app.data.repo.SyncRepository
import com.edgestack.app.domain.model.PortfolioRecommendationV2
import com.edgestack.app.data.remote.YahooChartClient
import com.edgestack.app.domain.OverlayCalculator
import com.edgestack.app.ui.components.Refreshable
import com.edgestack.app.ui.theme.Accent
import com.edgestack.app.ui.theme.AccentRed
import kotlinx.coroutines.launch

class OverlayViewModel(
    private val recommendationRepo: RecommendationRepository,
    private val syncRepo: SyncRepository,
    private val settingsStore: SettingsStore,
    private val quotesClient: YahooChartClient,
) : ViewModel() {
    var recommendation by mutableStateOf<PortfolioRecommendationV2?>(null); private set
    var settings by mutableStateOf(Settings()); private set
    var loading by mutableStateOf(false); private set
    var error by mutableStateOf(""); private set
    var regime by mutableStateOf<OverlayCalculator.Regime?>(null); private set

    init {
        recommendation = runCatching { recommendationRepo.displayedRecommendation() }.getOrNull()
        viewModelScope.launch { settings = settingsStore.current() }
        refreshRegime()
    }

    fun preview() {
        loading = true
        error = ""
        viewModelScope.launch {
            syncRepo.preview().fold(
                {
                    recommendation = it
                    error = ""
                },
                { error = "Preview requires the PC API; showing last canonical/default result." },
            )
            loading = false
        }
        refreshRegime()
    }

    private fun refreshRegime() {
        viewModelScope.launch {
            regime = runCatching {
                OverlayCalculator.latestRegime(quotesClient.dailyHistory("SPY", years = 2L))
            }.getOrNull()
        }
    }
}

@Composable
fun OverlayScreen(vm: OverlayViewModel) {
    Refreshable(refreshing = vm.loading, onRefresh = vm::preview) {
        OverlayContent(vm)
    }
}

@Composable
private fun RegimeCard(regime: OverlayCalculator.Regime?) {
    Card {
        Column(Modifier.fillMaxWidth().padding(12.dp)) {
            Text("SPY regime (device)", style = MaterialTheme.typography.titleMedium)
            if (regime == null) {
                Text(
                    "— quote history unavailable; pull to refresh with internet.",
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.Gray,
                )
            } else {
                Text(
                    if (regime.aboveSma200) "RISK-ON • above 200-day average" else
                        "RISK-OFF • below 200-day average",
                    style = MaterialTheme.typography.titleSmall,
                    color = if (regime.aboveSma200) Accent else AccentRed,
                )
                Text(
                    "SPY ${"%+.1f".format(regime.sma200DistancePercent)}% vs SMA200 • " +
                        "overlay exposure ${"%.2f".format(regime.currentExposure)}x",
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    "Device-computed from the reference overlay logic — context only, " +
                        "not the canonical signal.",
                    style = MaterialTheme.typography.labelSmall,
                    color = Color.Gray,
                )
            }
        }
    }
}

@Composable
private fun OverlayContent(vm: OverlayViewModel) {
    val recommendation = vm.recommendation
    Column(
        Modifier.fillMaxSize().padding(12.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("Portfolio & risk", style = MaterialTheme.typography.titleMedium)
        Text(
            "Server-calculated only. Offline mode never recalculates signals or leverage.",
            style = MaterialTheme.typography.labelSmall,
            color = Color.Gray,
        )
        Button(onClick = vm::preview, enabled = !vm.loading) { Text("Refresh custom preview") }
        if (vm.error.isNotBlank()) Text(vm.error, color = MaterialTheme.colorScheme.error)
        RegimeCard(vm.regime)
        if (recommendation == null) {
            Text("No canonical recommendation cached.")
            return@Column
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            AssistChip(onClick = {}, label = { Text(recommendation.status) })
            AssistChip(
                onClick = {},
                label = { Text(recommendation.outputRiskState.drawdownState) },
            )
        }
        Card {
            Column(Modifier.fillMaxWidth().padding(12.dp)) {
                Text("Leverage", style = MaterialTheme.typography.titleMedium)
                Text(
                    "requested ${"%.2f".format(vm.settings.maximumGrossLeverage)}x  •  " +
                        "effective ${"%.2f".format(recommendation.effectiveLeverage)}x",
                    style = MaterialTheme.typography.titleLarge,
                )
                Text(
                    "Binding: ${recommendation.bindingConstraints.ifEmpty { listOf("none") }.joinToString()}",
                    style = MaterialTheme.typography.bodySmall,
                )
                recommendation.constraints.filter { it.binding }.forEach {
                    Text("${it.name}: limit ${"%.2f".format(it.leverageLimit)}x")
                }
            }
        }
        Card {
            Column(Modifier.padding(12.dp)) {
                Text("Stress & financing", style = MaterialTheme.typography.titleMedium)
                Text("One-day 99.5% loss  ${"%.2f".format(recommendation.oneDayStressLoss * 100)}%")
                Text("20-session path loss  ${"%.2f".format(recommendation.multiSessionStressLoss * 100)}%")
                Text("Expected volatility  ${"%.2f".format(recommendation.expectedVolatility * 100)}%")
                Text("Annual funding cost  ${"%.2f".format(recommendation.fundingCost * 100)}%")
                Text("Modeled active net  ${"%+.2f".format(recommendation.expectedNetReturn * 100)}%")
            }
        }
        Card {
            Column(Modifier.padding(12.dp)) {
                Text("Canonical targets", style = MaterialTheme.typography.titleMedium)
                recommendation.personalizedTargetWeights.forEach {
                    Text("${it.symbol}  ${"%+.2f".format(it.weight * 100)}%")
                }
            }
        }
        Text(
            "Evidence ${recommendation.evidenceGrade} • fresh ${recommendation.freshness.isFresh} " +
                "(${recommendation.freshness.ageBusinessDays} business days) • " +
                "drawdown ${"%.2f".format(recommendation.outputRiskState.currentDrawdown * 100)}%",
            style = MaterialTheme.typography.bodySmall,
        )
        if (recommendation.outputRiskState.cashLatched) {
            Text(
                "Cash latch active. Reset eligible: ${recommendation.outputRiskState.resetEligible}; " +
                    "sessions latched: ${recommendation.outputRiskState.sessionsSinceLatch}",
                color = MaterialTheme.colorScheme.error,
            )
        }
        recommendation.warnings.forEach { Text("⚠ $it", style = MaterialTheme.typography.bodySmall) }
    }
}
