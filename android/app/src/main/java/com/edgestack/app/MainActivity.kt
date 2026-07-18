package com.edgestack.app

import android.Manifest
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.edgestack.app.ui.board.BoardScreen
import com.edgestack.app.ui.board.BoardViewModel
import com.edgestack.app.ui.calendar.CalendarScreen
import com.edgestack.app.ui.calendar.CalendarViewModel
import com.edgestack.app.ui.components.DisclaimerFooter
import com.edgestack.app.ui.edges.EdgesScreen
import com.edgestack.app.ui.edges.EdgesViewModel
import com.edgestack.app.ui.instrument.InstrumentScreen
import com.edgestack.app.ui.instrument.InstrumentViewModel
import com.edgestack.app.ui.overlay.OverlayScreen
import com.edgestack.app.ui.overlay.OverlayViewModel
import com.edgestack.app.ui.settings.SettingsScreen
import com.edgestack.app.ui.settings.SettingsViewModel
import com.edgestack.app.ui.sniper.SniperScreen
import com.edgestack.app.ui.sniper.SniperViewModel
import com.edgestack.app.ui.trades.TradesScreen
import com.edgestack.app.ui.trades.TradesViewModel
import com.edgestack.app.ui.theme.EdgeStackTheme

class MainActivity : ComponentActivity() {

    private val requestNotifications =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) {}

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (Build.VERSION.SDK_INT >= 33) {
            requestNotifications.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
        val container = (application as EdgeStackApp).container
        val factory = object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T = when (modelClass) {
                BoardViewModel::class.java ->
                    BoardViewModel(
                        container.recommendationRepo,
                        container.syncRepo,
                        container.settings,
                    ) as T
                OverlayViewModel::class.java ->
                    OverlayViewModel(
                        container.recommendationRepo,
                        container.syncRepo,
                        container.settings,
                        container.quotes,
                    ) as T
                TradesViewModel::class.java ->
                    TradesViewModel(
                        container.syncRepo,
                        container.positionsRepo,
                        container.quotes,
                    ) as T
                InstrumentViewModel::class.java ->
                    InstrumentViewModel(
                        container.instrumentRepo,
                        container.oilRepo,
                        container.syncRepo,
                        container.calendarRepo.macroEvents,
                    ) as T
                CalendarViewModel::class.java ->
                    CalendarViewModel(
                        container.calendarRepo,
                        container.instrumentRepo,
                        container.syncRepo,
                    ) as T
                SniperViewModel::class.java ->
                    SniperViewModel(container.sniperRepo, container.syncRepo) as T
                EdgesViewModel::class.java ->
                    EdgesViewModel(container.edgesRepo, container.syncRepo) as T
                SettingsViewModel::class.java ->
                    SettingsViewModel(container.settings, container.syncRepo) as T
                else -> throw IllegalArgumentException("unknown $modelClass")
            }
        }

        setContent {
            EdgeStackTheme {
                var tab by remember { mutableIntStateOf(0) }
                val tabs = listOf(
                    "Portfolio", "Calendar", "Risk", "Analyze",
                    "Sniper", "Edges", "Trades", "Settings",
                )
                Scaffold(
                    bottomBar = { DisclaimerFooter() },
                ) { padding ->
                    Column(Modifier.fillMaxSize().padding(padding)) {
                        ScrollableTabRow(selectedTabIndex = tab, edgePadding = 8.dp) {
                            tabs.forEachIndexed { i, label ->
                                Tab(
                                    selected = tab == i,
                                    onClick = { tab = i },
                                    text = { Text(label) },
                                )
                            }
                        }
                        when (tab) {
                            0 -> BoardScreen(viewModel(factory = factory))
                            1 -> CalendarScreen(viewModel(factory = factory))
                            2 -> OverlayScreen(viewModel(factory = factory))
                            3 -> InstrumentScreen(viewModel(factory = factory))
                            4 -> SniperScreen(viewModel(factory = factory))
                            5 -> EdgesScreen(viewModel(factory = factory))
                            6 -> TradesScreen(viewModel(factory = factory))
                            else -> SettingsScreen(viewModel(factory = factory))
                        }
                    }
                }
            }
        }
    }
}
