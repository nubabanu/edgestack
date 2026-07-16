package com.edgestack.app.work

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import com.edgestack.app.EdgeStackApp
import com.edgestack.app.domain.AlertPlanner
import com.edgestack.app.domain.AlertSettings
import com.edgestack.app.domain.AlertType
import java.time.Duration
import java.time.LocalDate
import java.time.ZonedDateTime

/**
 * Runs near 15:45 America/New_York on trading days:
 *  1. refresh SPY -> recompute overlay
 *  2. notify RISK channel on gate TRANSITIONS (200-DMA cross, vol gate flip)
 *  3. post today's calendar alerts from AlertPlanner
 *  4. re-enqueue itself for the next session (self-rechaining)
 */
class DailyCheckWorker(
    context: Context,
    params: WorkerParameters,
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        val app = applicationContext as EdgeStackApp
        val c = app.container
        val settings = c.settings.current()
        val calendar = c.calendarRepo.calendar
        val today = LocalDate.now(AlertPlanner.MARKET_ZONE)

        if (calendar.isSession(today)) {
            // risk transitions from fresh SPY data (best effort)
            runCatching {
                val state = c.overlayRepo.state(settings.baseLeverage, forceRefresh = true)
                if (state?.sma200 != null) {
                    val above = state.spyClose >= state.sma200
                    val volOn = (state.vol20 ?: 0.0) > 0.20
                    val prevAbove = settings.lastAbove200Dma
                    val prevVol = settings.lastVolGateOn
                    if (prevAbove != null && prevAbove != above) {
                        AlertNotifier.notify(
                            applicationContext, AlertNotifier.CHANNEL_RISK, 100,
                            if (above) "SPY back above 200-DMA" else "SPY BROKE its 200-DMA",
                            if (above) "Overlay gate released; normal exposure allowed."
                            else "Overlay caps exposure at 0.5x. Best drawdown gate in 14y of data.",
                        )
                    }
                    if (prevVol != null && prevVol != volOn) {
                        AlertNotifier.notify(
                            applicationContext, AlertNotifier.CHANNEL_RISK, 101,
                            if (volOn) "Volatility gate ON (>20%)" else "Volatility gate off",
                            if (volOn) "20d realized vol above 20%: no leverage; suspend dip-buying."
                            else "20d realized vol back under 20%.",
                        )
                    }
                    // red-close sniper: SPY down today, in calm uptrend
                    // (t=5.41 finding: buy the close, sell at tomorrow's open;
                    // price at 15:45 ET approximates the close)
                    if ((state.lastReturn ?: 0.0) < 0.0 && above && !volOn) {
                        AlertNotifier.notify(
                            applicationContext, AlertNotifier.CHANNEL_CALENDAR, 210,
                            "Sniper: red close in calm uptrend",
                            "SPY is down today inside an uptrend. Historically the " +
                                "gentlest trade: buy near the close, sell at " +
                                "tomorrow's open (60% hit, typical bad case -0.8%).",
                        )
                    }
                    c.settings.setRiskState(above, volOn)
                }
            }

            val alertSettings = AlertSettings(
                buyAtClose = settings.alertsBuyAtClose,
                turnOfMonth = settings.alertsTurnOfMonth,
                septemberDerisk = settings.alertsSeptember,
                febSniper = settings.alertsFebSniper,
                novWorstDay = settings.alertsNovWorstDay,
            )
            AlertPlanner.alertsFor(calendar, today, alertSettings).forEach { alert ->
                AlertNotifier.notify(
                    applicationContext, AlertNotifier.CHANNEL_CALENDAR,
                    200 + alert.type.ordinal, alert.title, alert.message)
            }

            // tracked-position checks: stop breach + time-exit due
            runCatching {
                val positions = c.positionsRepo.load()
                if (positions.isNotEmpty()) {
                    val quotes = c.yahoo.latestQuotes(
                        positions.map { it.symbol }.distinct())
                    positions.forEachIndexed { i, p ->
                        val last = quotes[p.symbol] ?: return@forEachIndexed
                        if (p.stop != null && last <= p.stop) {
                            AlertNotifier.notify(
                                applicationContext, AlertNotifier.CHANNEL_RISK,
                                300 + i, "STOP BREACHED: ${p.symbol}",
                                "Last %.2f <= stop %.2f. Exit at/near the close — "
                                    .format(last, p.stop) +
                                    "no averaging down.")
                        }
                        val held = calendar.sessionsBetween(
                            java.time.LocalDate.parse(p.entryDate), today)
                        if (held >= p.horizonSessions) {
                            AlertNotifier.notify(
                                applicationContext, AlertNotifier.CHANNEL_CALENDAR,
                                340 + i, "Time exit due: ${p.symbol}",
                                "Held $held sessions (plan: ${p.horizonSessions}). " +
                                    "The edge was measured to here — exit at the close.")
                        }
                    }
                }
            }
        }

        WorkScheduler.scheduleNext(applicationContext, calendar)
        return Result.success()
    }

    companion object {
        const val UNIQUE_NAME = "daily-alert-check"
    }
}

object WorkScheduler {

    /** Enqueue the next run at 15:45 ET on the next session (or today if early). */
    fun scheduleNext(context: Context, calendar: com.edgestack.app.domain.TradingCalendar) {
        val nowEt = ZonedDateTime.now(AlertPlanner.MARKET_ZONE)
        var target = nowEt.toLocalDate()
        if (!calendar.isSession(target) || !nowEt.toLocalTime().isBefore(AlertPlanner.ALERT_TIME)) {
            target = calendar.nextSession(target) ?: return
        }
        val fireAt = AlertPlanner.fireAt(target)
        val delay = Duration.between(nowEt, fireAt).let {
            if (it.isNegative) Duration.ofMinutes(1) else it
        }
        val request = OneTimeWorkRequestBuilder<DailyCheckWorker>()
            .setInitialDelay(delay)
            .build()
        WorkManager.getInstance(context).enqueueUniqueWork(
            DailyCheckWorker.UNIQUE_NAME, ExistingWorkPolicy.REPLACE, request)
    }
}
