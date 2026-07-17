package com.edgestack.app.ui.calendar

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.edgestack.app.data.repo.CalendarRepository
import com.edgestack.app.data.repo.InstrumentAnalysisRepository
import com.edgestack.app.data.repo.SyncRepository
import kotlinx.coroutines.launch
import com.edgestack.app.domain.DayEdge
import com.edgestack.app.domain.SeasonalOverlay
import com.edgestack.app.domain.TradingCalendar
import java.time.LocalDate
import java.time.YearMonth
import java.time.format.TextStyle
import java.util.Locale

class CalendarViewModel(
    calendarRepo: CalendarRepository,
    private val instrumentRepo: InstrumentAnalysisRepository,
    private val syncRepository: SyncRepository,
) : ViewModel() {
    val calendar: TradingCalendar = calendarRepo.calendar
    var month by mutableStateOf(YearMonth.now())
    var selected by mutableStateOf<LocalDate?>(null)
    var overlay by mutableStateOf(SeasonalOverlay(emptyList())); private set
    var overlaySymbol by mutableStateOf<String?>(null); private set
    var symbolDraft by mutableStateOf("")
    var loading by mutableStateOf(false); private set
    var message by mutableStateOf(""); private set

    /** Re-read the last instrument analysis so the overlay follows the Analyze tab. */
    fun reload() {
        val analysis = instrumentRepo.loadLast()
        overlay = SeasonalOverlay(analysis?.tailwindCalendars.orEmpty())
        overlaySymbol = analysis?.resolution?.resolvedSymbol
    }

    /** Fetch server tailwind evidence for [symbol] and shade the grid with it. */
    fun shade(symbol: String = symbolDraft) {
        val cleaned = symbol.trim().uppercase()
        if (cleaned.isBlank()) return
        loading = true
        message = ""
        viewModelScope.launch {
            syncRepository.analyzeInstrument(cleaned).fold(
                onSuccess = {
                    reload()
                    symbolDraft = ""
                    message = "Shading ${it.resolution.resolvedSymbol} evidence."
                },
                onFailure = { message = "Shading unavailable: ${it.message}" },
            )
            loading = false
        }
    }

    companion object {
        val QUICK_SYMBOLS = listOf("SPY", "QQQ", "GLD", "USO")
    }
}

@Composable
fun CalendarScreen(vm: CalendarViewModel) {
    LaunchedEffect(Unit) { vm.reload() }
    val today = LocalDate.now()
    val month = vm.month
    LazyColumn(
        Modifier.fillMaxSize().padding(horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        item {
            Text("Trading calendar", style = MaterialTheme.typography.titleLarge)
            Text(
                if (vm.overlay.hasData) {
                    "NYSE sessions with historical tailwind shading for " +
                        "${vm.overlaySymbol ?: "the last analyzed instrument"} — research only."
                } else {
                    "NYSE sessions and turn-of-month windows. Pick an instrument to " +
                        "shade days by its historical tailwind evidence."
                },
                style = MaterialTheme.typography.bodySmall,
                color = Color.Gray,
            )
            Row(
                Modifier.horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                CalendarViewModel.QUICK_SYMBOLS.forEach { quick ->
                    AssistChip(
                        onClick = { vm.shade(quick) },
                        enabled = !vm.loading,
                        label = {
                            Text(if (vm.overlaySymbol == quick) "✓ $quick" else quick)
                        },
                    )
                }
            }
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                OutlinedTextField(
                    value = vm.symbolDraft,
                    onValueChange = { vm.symbolDraft = it },
                    modifier = Modifier.weight(1f),
                    label = { Text("Other ticker or commodity") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(
                        capitalization = KeyboardCapitalization.Characters,
                        autoCorrect = false,
                    ),
                )
                TextButton(onClick = { vm.shade() }, enabled = !vm.loading) {
                    Text(if (vm.loading) "…" else "Shade")
                }
            }
            if (vm.message.isNotBlank()) {
                Text(vm.message, style = MaterialTheme.typography.labelSmall)
            }
        }
        item {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                TextButton(onClick = { vm.month = month.minusMonths(1) }) { Text("◀") }
                Text(
                    "${month.month.getDisplayName(TextStyle.FULL, Locale.ENGLISH)} ${month.year}",
                    style = MaterialTheme.typography.titleMedium,
                )
                Row {
                    if (month != YearMonth.now()) {
                        TextButton(onClick = {
                            vm.month = YearMonth.now()
                            vm.selected = today
                        }) { Text("Today") }
                    }
                    TextButton(onClick = { vm.month = month.plusMonths(1) }) { Text("▶") }
                }
            }
        }
        item {
            MonthGrid(
                month = month,
                today = today,
                calendar = vm.calendar,
                overlay = vm.overlay,
                selected = vm.selected,
                onSelect = { vm.selected = it },
            )
        }
        item { Legend(vm.overlay.hasData) }
        if (vm.overlay.hasData) {
            item { WeekAheadCard(today, vm.calendar, vm.overlay, vm.overlaySymbol) }
        }
        item { MonthFacts(month, today, vm.calendar, vm.overlay) }
        vm.selected?.let { date ->
            item { DayDetails(date, vm.calendar, vm.overlay.edgeFor(date)) }
        }
    }
}

@Composable
private fun MonthGrid(
    month: YearMonth,
    today: LocalDate,
    calendar: TradingCalendar,
    overlay: SeasonalOverlay,
    selected: LocalDate?,
    onSelect: (LocalDate) -> Unit,
) {
    Card {
        Column(Modifier.padding(6.dp)) {
            Row(Modifier.fillMaxWidth()) {
                listOf("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su").forEach { label ->
                    Text(
                        label,
                        Modifier.weight(1f),
                        textAlign = TextAlign.Center,
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.Gray,
                    )
                }
            }
            val first = month.atDay(1)
            val leadingBlanks = first.dayOfWeek.value - 1
            val days = (1..month.lengthOfMonth()).map(month::atDay)
            val cells: List<LocalDate?> = List(leadingBlanks) { null } + days
            cells.chunked(7).forEach { week ->
                Row(Modifier.fillMaxWidth()) {
                    week.forEach { date ->
                        DayCell(
                            date = date,
                            today = today,
                            calendar = calendar,
                            overlay = overlay,
                            isSelected = date != null && date == selected,
                            onSelect = onSelect,
                            modifier = Modifier.weight(1f),
                        )
                    }
                    repeat(7 - week.size) { Box(Modifier.weight(1f)) }
                }
            }
        }
    }
}

@Composable
private fun DayCell(
    date: LocalDate?,
    today: LocalDate,
    calendar: TradingCalendar,
    overlay: SeasonalOverlay,
    isSelected: Boolean,
    onSelect: (LocalDate) -> Unit,
    modifier: Modifier = Modifier,
) {
    if (date == null) {
        Box(modifier)
        return
    }
    val isSession = calendar.isSession(date)
    val edge = if (isSession) overlay.edgeFor(date) else null
    val heat = edge?.let { heatColor(it.percentile) } ?: Color.Transparent
    val outline = when {
        isSelected -> MaterialTheme.colorScheme.primary
        date == today -> MaterialTheme.colorScheme.onSurface
        else -> Color.Transparent
    }
    Box(
        modifier
            .padding(2.dp)
            .aspectRatio(1f)
            .clip(RoundedCornerShape(8.dp))
            .background(heat)
            .border(1.dp, outline, RoundedCornerShape(8.dp))
            .clickable { onSelect(date) },
        contentAlignment = Alignment.Center,
    ) {
        Text(
            "${date.dayOfMonth}",
            style = MaterialTheme.typography.bodySmall,
            color = if (isSession) MaterialTheme.colorScheme.onSurface else Color.Gray,
        )
        if (isSession && calendar.isTurnOfMonthWindow(date)) {
            Box(
                Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 3.dp)
                    .size(4.dp)
                    .clip(CircleShape)
                    .background(MaterialTheme.colorScheme.tertiary),
            )
        }
    }
}

@Composable
private fun heatColor(percentile: Double): Color {
    val p = percentile.toFloat().coerceIn(0f, 1f)
    return if (p >= 0.5f) {
        MaterialTheme.colorScheme.primary.copy(alpha = (p - 0.5f) * 0.8f)
    } else {
        MaterialTheme.colorScheme.error.copy(alpha = (0.5f - p) * 0.8f)
    }
}

@Composable
private fun Legend(hasOverlay: Boolean) {
    Card {
        Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    Modifier.size(10.dp).clip(CircleShape)
                        .background(MaterialTheme.colorScheme.tertiary),
                )
                Text(
                    "  Turn-of-month window (last session or first 3 of the month)",
                    style = MaterialTheme.typography.labelSmall,
                )
            }
            Text("Gray dates are non-sessions.", style = MaterialTheme.typography.labelSmall)
            if (hasOverlay) {
                Text(
                    "Shading: blue = historically stronger slot, red = weaker, by server " +
                        "rank percentile. Past frequencies, not forecasts.",
                    style = MaterialTheme.typography.labelSmall,
                )
            }
        }
    }
}

/** The next five sessions ranked by historical tailwind — "when this week?" */
@Composable
private fun WeekAheadCard(
    today: LocalDate,
    calendar: TradingCalendar,
    overlay: SeasonalOverlay,
    symbol: String?,
) {
    val sessions = buildList {
        var d: LocalDate? = if (calendar.isSession(today)) today else calendar.nextSession(today)
        while (d != null && size < 5) {
            add(d)
            d = calendar.nextSession(d)
        }
    }
    val scored = sessions.mapNotNull { d -> overlay.edgeFor(d)?.let { d to it } }
    if (scored.isEmpty()) return
    val best = scored.maxByOrNull { it.second.percentile }?.first
    val worst = scored.minByOrNull { it.second.percentile }?.first
    Card {
        Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Text(
                "Next ${scored.size} sessions${symbol?.let { " — $it" } ?: ""}",
                style = MaterialTheme.typography.titleSmall,
            )
            scored.forEach { (d, edge) ->
                val marker = when (d) {
                    best -> "  ← strongest"
                    worst -> "  ← weakest"
                    else -> ""
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        Modifier.size(10.dp).clip(CircleShape).background(heatColor(edge.percentile)),
                    )
                    Text(
                        "  ${d.dayOfWeek.getDisplayName(TextStyle.SHORT, Locale.ENGLISH)} $d — " +
                            "percentile ${"%.0f".format(edge.percentile * 100)}$marker",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
            Text(
                "Historical tilt for entries you already planned — not a forecast.",
                style = MaterialTheme.typography.labelSmall,
                color = Color.Gray,
            )
        }
    }
}

@Composable
private fun MonthFacts(
    month: YearMonth,
    today: LocalDate,
    calendar: TradingCalendar,
    overlay: SeasonalOverlay,
) {
    val sessions = (1..month.lengthOfMonth()).map(month::atDay).filter(calendar::isSession)
    Card {
        Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text("This month", style = MaterialTheme.typography.titleSmall)
            Text("${sessions.size} trading sessions", style = MaterialTheme.typography.bodySmall)
            calendar.nextSession(today)?.let {
                Text("Next session after today: $it", style = MaterialTheme.typography.bodySmall)
            }
            if (overlay.hasData && sessions.isNotEmpty()) {
                val ranked = sessions.mapNotNull { d -> overlay.edgeFor(d)?.let { d to it } }
                ranked.maxByOrNull { it.second.percentile }?.let { (d, e) ->
                    Text(
                        "Strongest historical slot: $d " +
                            "(percentile ${"%.0f".format(e.percentile * 100)})",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                ranked.minByOrNull { it.second.percentile }?.let { (d, e) ->
                    Text(
                        "Weakest historical slot: $d " +
                            "(percentile ${"%.0f".format(e.percentile * 100)})",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }
    }
}

@Composable
private fun DayDetails(date: LocalDate, calendar: TradingCalendar, edge: DayEdge?) {
    Card {
        Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text("$date", style = MaterialTheme.typography.titleSmall)
            if (!calendar.isSession(date)) {
                Text("Not a trading session.", style = MaterialTheme.typography.bodySmall)
                return@Column
            }
            calendar.tradingDayOfMonth(date)?.let { n ->
                val fromEnd = calendar.tradingDayFromMonthEnd(date)
                Text(
                    "Trading day $n of the month ($fromEnd from month end)",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            if (calendar.isTurnOfMonthWindow(date)) {
                Text("Inside the turn-of-month window.", style = MaterialTheme.typography.bodySmall)
            }
            if (edge == null) {
                Text(
                    "No tailwind evidence for this date — analyze an instrument first.",
                    style = MaterialTheme.typography.bodySmall,
                )
            } else {
                edge.components.forEach { (label, cell) ->
                    Text(
                        "$label ${cell.displayLabel}: win score " +
                            "${"%.1f".format(cell.score.winScore)}, rank " +
                            "${cell.score.rank}/${cell.score.candidatesRanked} " +
                            "(${cell.score.evidenceGrade})",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                Text(
                    "Historical frequencies only; the server decides what is actionable.",
                    style = MaterialTheme.typography.labelSmall,
                    color = Color.Gray,
                )
            }
        }
    }
}
