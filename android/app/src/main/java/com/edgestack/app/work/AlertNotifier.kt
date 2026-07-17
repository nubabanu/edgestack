package com.edgestack.app.work

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import com.edgestack.app.MainActivity

object AlertNotifier {

    const val CHANNEL_CANONICAL = "canonical_recommendations"
    const val CHANNEL_RISK = "risk_alerts"
    const val CHANNEL_TIMING = "timing_windows"

    fun createChannels(context: Context) {
        val nm = context.getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_CANONICAL, "Canonical recommendations",
                NotificationManager.IMPORTANCE_HIGH).apply {
                description = "Canonical status, target, and freshness changes"
            })
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_RISK, "Risk alerts",
                NotificationManager.IMPORTANCE_HIGH).apply {
                description = "Canonical drawdown and latch-state changes"
            })
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_TIMING, "Timing windows",
                NotificationManager.IMPORTANCE_DEFAULT).apply {
                description = "Sniper candidate triggers, scheduled entries, turn-of-month windows"
            })
    }

    fun notify(context: Context, channel: String, id: Int, title: String, text: String) {
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED
        ) return
        val intent = PendingIntent.getActivity(
            context, 0, Intent(context, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE)
        val notification = NotificationCompat.Builder(context, channel)
            .setSmallIcon(android.R.drawable.stat_notify_chat)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(
                "$text\n\nResearch output only. Not investment advice."))
            .setContentIntent(intent)
            .setAutoCancel(true)
            .build()
        NotificationManagerCompat.from(context).notify(id, notification)
    }
}
