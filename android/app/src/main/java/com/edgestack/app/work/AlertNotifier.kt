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

    const val CHANNEL_CALENDAR = "calendar_alerts"
    const val CHANNEL_RISK = "risk_alerts"

    fun createChannels(context: Context) {
        val nm = context.getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_CALENDAR, "Calendar alerts",
                NotificationManager.IMPORTANCE_HIGH).apply {
                description = "Turn-of-month, September de-risk, Feb-1 sniper, buy-at-close"
            })
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_RISK, "Risk alerts",
                NotificationManager.IMPORTANCE_HIGH).apply {
                description = "200-DMA breach and volatility gate transitions"
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
