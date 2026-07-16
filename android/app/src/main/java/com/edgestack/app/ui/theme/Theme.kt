package com.edgestack.app.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

val Accent = Color(0xFF4CC38A)
val AccentRed = Color(0xFFE5484D)
val AccentAmber = Color(0xFFFFB224)
val Surface1 = Color(0xFF101418)
val Surface2 = Color(0xFF1A2027)

private val DarkColors = darkColorScheme(
    primary = Accent,
    onPrimary = Color.Black,
    secondary = AccentAmber,
    background = Surface1,
    surface = Surface2,
    error = AccentRed,
)

@Composable
fun EdgeStackTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = DarkColors, content = content)
}
