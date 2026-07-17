package com.edgestack.app.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.edgestack.app.ui.theme.Accent
import com.edgestack.app.ui.theme.AccentAmber
import com.edgestack.app.ui.theme.AccentRed
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

/** Standard swipe-down-to-refresh wrapper used by every server-backed tab. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Refreshable(
    refreshing: Boolean,
    onRefresh: () -> Unit,
    modifier: Modifier = Modifier,
    content: @Composable BoxScope.() -> Unit,
) {
    PullToRefreshBox(
        isRefreshing = refreshing,
        onRefresh = onRefresh,
        modifier = modifier.fillMaxSize(),
        content = content,
    )
}

@Composable
fun DisclaimerFooter() {
    Text(
        text = "Research output only. Not investment advice.",
        style = MaterialTheme.typography.labelSmall,
        color = Color.Gray,
        textAlign = TextAlign.Center,
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp),
    )
}

/** Arc gauge 0..2.5x with a needle at [value]. */
@Composable
fun ExposureDial(value: Double, modifier: Modifier = Modifier) {
    val maxV = 2.5
    Box(modifier = modifier.fillMaxWidth().height(190.dp),
        contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.fillMaxWidth().height(170.dp)) {
            val stroke = 22f
            val d = min(size.width, size.height * 2f) - stroke * 2
            val topLeft = Offset((size.width - d) / 2f, size.height - d / 2f - stroke)
            val arcSize = Size(d, d)
            val startAngle = 180f
            val sweep = 180f
            fun band(fromV: Double, toV: Double, color: Color) {
                drawArc(
                    color = color,
                    startAngle = startAngle + (fromV / maxV * sweep).toFloat(),
                    sweepAngle = ((toV - fromV) / maxV * sweep).toFloat(),
                    useCenter = false,
                    topLeft = topLeft,
                    size = arcSize,
                    style = Stroke(width = stroke, cap = StrokeCap.Butt),
                )
            }
            band(0.0, 0.5, AccentRed.copy(alpha = 0.75f))
            band(0.5, 1.0, AccentAmber.copy(alpha = 0.75f))
            band(1.0, 1.6, Accent.copy(alpha = 0.75f))
            band(1.6, 2.5, Accent.copy(alpha = 0.35f))
            // needle
            val angleRad = Math.toRadians(
                (startAngle + (value.coerceIn(0.0, maxV) / maxV * sweep)))
            val center = Offset(topLeft.x + d / 2f, topLeft.y + d / 2f)
            val r = d / 2f - stroke
            drawLine(
                color = Color.White,
                start = center,
                end = Offset(
                    center.x + (r * cos(angleRad)).toFloat(),
                    center.y + (r * sin(angleRad)).toFloat(),
                ),
                strokeWidth = 6f,
                cap = StrokeCap.Round,
            )
            drawCircle(Color.White, radius = 9f, center = center)
        }
        Text(
            text = "%.2fx".format(value),
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.align(Alignment.BottomCenter),
        )
    }
}

/** Minimal line sparkline for a numeric series. */
@Composable
fun Sparkline(values: List<Double>, modifier: Modifier = Modifier, color: Color = Accent) {
    if (values.size < 2) return
    Canvas(modifier = modifier.fillMaxWidth().height(72.dp)) {
        val lo = values.min()
        val hi = values.max()
        val span = (hi - lo).takeIf { it > 1e-12 } ?: 1.0
        val stepX = size.width / (values.size - 1)
        val path = Path()
        values.forEachIndexed { i, v ->
            val x = i * stepX
            val y = size.height - ((v - lo) / span * size.height).toFloat()
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, color = color, style = Stroke(width = 4f, cap = StrokeCap.Round))
    }
}
