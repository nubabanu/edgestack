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
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.ShoppingCart
import androidx.compose.material.icons.filled.Star
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.edgestack.app.ui.board.BoardScreen
import com.edgestack.app.ui.board.BoardViewModel
import com.edgestack.app.ui.components.DisclaimerFooter
import com.edgestack.app.ui.edges.EdgesScreen
import com.edgestack.app.ui.edges.EdgesViewModel
import com.edgestack.app.ui.overlay.OverlayScreen
import com.edgestack.app.ui.overlay.OverlayViewModel
import com.edgestack.app.ui.settings.SettingsScreen
import com.edgestack.app.ui.settings.SettingsViewModel
import com.edgestack.app.ui.trades.TradesScreen
import com.edgestack.app.ui.trades.TradesViewModel
import com.edgestack.app.ui.theme.EdgeStackTheme
import com.edgestack.app.work.AlertNotifier

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
                    BoardViewModel(container.boardRepo, container.syncRepo) as T
                OverlayViewModel::class.java ->
                    OverlayViewModel(container.overlayRepo, container.settings,
                                     container.calendarRepo.calendar) as T
                TradesViewModel::class.java ->
                    TradesViewModel(container.positionsRepo, container.syncRepo,
                                    container.yahoo,
                                    container.calendarRepo.calendar) as T
                EdgesViewModel::class.java -> EdgesViewModel(container.edgesRepo) as T
                SettingsViewModel::class.java ->
                    SettingsViewModel(container.settings, container.syncRepo) as T
                else -> throw IllegalArgumentException("unknown $modelClass")
            }
        }

        setContent {
            EdgeStackTheme {
                var tab by remember { mutableIntStateOf(0) }
                val tabs = listOf(
                    "Board" to Icons.Filled.Home,
                    "Overlay" to Icons.Filled.Star,
                    "Trades" to Icons.Filled.ShoppingCart,
                    "Edges" to Icons.AutoMirrored.Filled.List,
                    "Settings" to Icons.Filled.Settings,
                )
                Scaffold(
                    bottomBar = {
                        Column {
                            DisclaimerFooter()
                            NavigationBar {
                                tabs.forEachIndexed { i, (label, icon) ->
                                    NavigationBarItem(
                                        selected = tab == i,
                                        onClick = { tab = i },
                                        icon = { Icon(icon, contentDescription = label) },
                                        label = { Text(label) },
                                    )
                                }
                            }
                        }
                    },
                ) { padding ->
                    Column(Modifier.fillMaxSize().padding(padding)) {
                        when (tab) {
                            0 -> BoardScreen(viewModel(factory = factory))
                            1 -> OverlayScreen(viewModel(factory = factory))
                            2 -> TradesScreen(viewModel(factory = factory))
                            3 -> EdgesScreen(viewModel(factory = factory))
                            else -> SettingsScreen(viewModel(factory = factory)) {
                                AlertNotifier.notify(
                                    this@MainActivity, AlertNotifier.CHANNEL_CALENDAR,
                                    999, "Test notification",
                                    "Channels work. Alerts fire 15:45 New York time.")
                            }
                        }
                    }
                }
            }
        }
    }
}
