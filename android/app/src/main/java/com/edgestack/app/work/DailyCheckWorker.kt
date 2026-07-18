package com.edgestack.app.work

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import com.edgestack.app.EdgeStackApp
import java.time.Duration
import java.time.LocalDate
import java.time.LocalTime
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.ZonedDateTime

/** Syncs server-owned recommendations; the device never computes a signal or overlay. */
class DailyCheckWorker(
    context: Context,
    params: WorkerParameters,
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        val container = (applicationContext as EdgeStackApp).container
        val before = container.settings.current()
        val beforeRisk = container.settings.decodedRiskState(before)
        val beforePlan = container.sniperRepo.load()
        container.syncRepo.syncAll().onSuccess {
            val after = container.settings.current()
            val afterRisk = container.settings.decodedRiskState(after)
            if (before.lastCanonicalStatus.isNotBlank() &&
                before.lastCanonicalStatus != after.lastCanonicalStatus
            ) {
                AlertNotifier.notify(
                    applicationContext,
                    AlertNotifier.CHANNEL_CANONICAL,
                    100,
                    "Canonical status changed",
                    "${before.lastCanonicalStatus} → ${after.lastCanonicalStatus}",
                )
            }
            if (before.lastCanonicalFingerprint.isNotBlank() &&
                before.lastCanonicalFingerprint != after.lastCanonicalFingerprint
            ) {
                val recommendation = container.recommendationRepo.displayedRecommendation()
                AlertNotifier.notify(
                    applicationContext,
                    AlertNotifier.CHANNEL_CANONICAL,
                    101,
                    "Canonical targets changed",
                    "Effective leverage ${"%.2f".format(recommendation.effectiveLeverage)}x; " +
                        "open the app for target weights and binding constraints.",
                )
            }
            if (before.lastFresh != null && before.lastFresh != after.lastFresh) {
                AlertNotifier.notify(
                    applicationContext,
                    AlertNotifier.CHANNEL_CANONICAL,
                    102,
                    "Recommendation freshness changed",
                    if (after.lastFresh == true) "Canonical inputs are fresh again."
                    else "Canonical inputs are stale; position increases are disabled.",
                )
            }
            if (beforeRisk != null && afterRisk != null &&
                beforeRisk.drawdownState != afterRisk.drawdownState
            ) {
                AlertNotifier.notify(
                    applicationContext,
                    AlertNotifier.CHANNEL_RISK,
                    103,
                    "Risk state ${afterRisk.drawdownState}",
                    "Drawdown ${"%.2f".format(afterRisk.currentDrawdown * 100)}%; " +
                        "cash latch ${afterRisk.cashLatched}; reset eligible ${afterRisk.resetEligible}.",
                )
            }
            if (after.timingAlerts) {
                notifyTimingWindows(container, beforePlan)
            }
        }
        if (container.settings.current().timingAlerts) {
            notifyStopTargetBreaches(container)
        }

        WorkScheduler.scheduleNext(applicationContext, container.calendarRepo.calendar)
        return Result.success()
    }

    /** Alert when a tracked position's stop or target level is crossed. */
    private suspend fun notifyStopTargetBreaches(container: com.edgestack.app.di.AppContainer) {
        val positions = container.positionsRepo.load()
        if (positions.isEmpty()) return
        val quotes = runCatching {
            container.quotes.latestQuotes(positions.map { it.symbol }.distinct())
        }.getOrElse { return }
        positions.forEachIndexed { index, position ->
            val last = quotes[position.symbol] ?: return@forEachIndexed
            position.stop?.takeIf { last <= it }?.let { stop ->
                AlertNotifier.notify(
                    applicationContext,
                    AlertNotifier.CHANNEL_TIMING,
                    160 + index,
                    "${position.symbol} at stop level",
                    "Last ${"%.2f".format(last)} <= stop ${"%.2f".format(stop)} " +
                        "(entry ${"%.2f".format(position.entryPrice)}). Delayed device quote.",
                )
            }
            position.target?.takeIf { last >= it }?.let { target ->
                AlertNotifier.notify(
                    applicationContext,
                    AlertNotifier.CHANNEL_TIMING,
                    180 + index,
                    "${position.symbol} at target level",
                    "Last ${"%.2f".format(last)} >= target ${"%.2f".format(target)} " +
                        "(entry ${"%.2f".format(position.entryPrice)}). Delayed device quote.",
                )
            }
        }
    }

    /** Sniper transitions and upcoming windows; states come from the server only. */
    private fun notifyTimingWindows(
        container: com.edgestack.app.di.AppContainer,
        beforePlan: com.edgestack.app.domain.model.SniperPlanV2?,
    ) {
        val plan = container.sniperRepo.load() ?: return
        val calendar = container.calendarRepo.calendar
        val zone = ZoneId.of("America/New_York")
        val today = ZonedDateTime.now(zone).toLocalDate()
        val nextSession = calendar.nextSession(today)

        val beforeStates = beforePlan?.stage1Candidates
            ?.associate { it.strategyId to it.status }.orEmpty()
        plan.stage1Candidates.forEachIndexed { index, candidate ->
            val previous = beforeStates[candidate.strategyId]
            if (candidate.status.equals("TRIGGERED", true) &&
                !previous.equals("TRIGGERED", true)
            ) {
                AlertNotifier.notify(
                    applicationContext,
                    AlertNotifier.CHANNEL_TIMING,
                    130 + index,
                    "Sniper candidate triggered: ${candidate.strategyId}",
                    "${candidate.symbol} • ${candidate.entryWindow ?: "see plan"} • " +
                        "paper-only shadow plan; open Sniper for sizing.",
                )
            }
            val entryDate = candidate.entryWindow?.let { window ->
                DATE_PATTERN.find(window)?.value?.let { LocalDate.parse(it) }
            }
            if (entryDate != null && (entryDate == today || entryDate == nextSession)) {
                AlertNotifier.notify(
                    applicationContext,
                    AlertNotifier.CHANNEL_TIMING,
                    140 + index,
                    "Sniper entry window ${if (entryDate == today) "today" else "next session"}",
                    "${candidate.strategyId} (${candidate.symbol}): planned entry $entryDate. " +
                        "Exit: ${candidate.exitRule}",
                )
            }
        }

        if (nextSession != null &&
            calendar.isLastSessionOfMonth(nextSession) &&
            !calendar.isTurnOfMonthWindow(today)
        ) {
            AlertNotifier.notify(
                applicationContext,
                AlertNotifier.CHANNEL_TIMING,
                150,
                "Turn-of-month window starts next session",
                "$nextSession is the month's last session; the historical turn-of-month " +
                    "window runs through the first 3 sessions of next month.",
            )
        }

        if (nextSession != null) {
            container.calendarRepo.macroEvents.highImpactOn(nextSession)
                .forEachIndexed { index, event ->
                    AlertNotifier.notify(
                        applicationContext,
                        AlertNotifier.CHANNEL_TIMING,
                        200 + index,
                        "${event.type} next session",
                        "$nextSession: ${event.label} at ${event.timeEt} ET — " +
                            "expect volatility around the release.",
                    )
                }
        }
    }

    companion object {
        const val UNIQUE_NAME = "daily-canonical-check"
        private val DATE_PATTERN = Regex("\\d{4}-\\d{2}-\\d{2}")
    }
}

/** Rechecks the cached server-owned instrument plan at the cadence returned by the API. */
class InstrumentRecheckWorker(
    context: Context,
    params: WorkerParameters,
) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val container = (applicationContext as EdgeStackApp).container
        return container.syncRepo.recheckInstrument().fold(
            onSuccess = { result ->
                if (!result.recommendationStillHolds || result.betterAlternativeEmerged) {
                    AlertNotifier.notify(
                        applicationContext,
                        AlertNotifier.CHANNEL_CANONICAL,
                        120,
                        if (result.betterAlternativeEmerged) {
                            "Better timing alternative found"
                        } else {
                            "Instrument timing changed"
                        },
                        result.changes.joinToString().ifBlank {
                            "Open Analyze for the server-owned updated rating and exits."
                        },
                    )
                }
                WorkScheduler.scheduleInstrumentRecheck(
                    applicationContext,
                    result.analysis.recheckPlan.nextCheckAt,
                )
                Result.success()
            },
            onFailure = { Result.retry() },
        )
    }

    companion object {
        const val UNIQUE_NAME = "instrument-plan-recheck"
    }
}

object WorkScheduler {
    /** Enqueue a daily refresh on the next exchange session. */
    fun scheduleNext(context: Context, calendar: com.edgestack.app.domain.TradingCalendar) {
        val marketZone = ZoneId.of("America/New_York")
        val refreshTime = LocalTime.of(18, 30)
        val nowEt = ZonedDateTime.now(marketZone)
        var target = nowEt.toLocalDate()
        if (!calendar.isSession(target) || !nowEt.toLocalTime().isBefore(refreshTime)) {
            target = calendar.nextSession(target) ?: return
        }
        val fireAt = ZonedDateTime.of(target, refreshTime, marketZone)
        val delay = Duration.between(nowEt, fireAt).let {
            if (it.isNegative) Duration.ofMinutes(1) else it
        }
        val request = OneTimeWorkRequestBuilder<DailyCheckWorker>()
            .setInitialDelay(delay)
            .build()
        WorkManager.getInstance(context).enqueueUniqueWork(
            DailyCheckWorker.UNIQUE_NAME,
            ExistingWorkPolicy.REPLACE,
            request,
        )
    }

    fun scheduleInstrumentRecheck(context: Context, nextCheckAt: String?) {
        if (nextCheckAt.isNullOrBlank()) {
            WorkManager.getInstance(context).cancelUniqueWork(InstrumentRecheckWorker.UNIQUE_NAME)
            return
        }
        val target = runCatching { OffsetDateTime.parse(nextCheckAt).toInstant() }.getOrNull()
            ?: return
        val delay = Duration.between(java.time.Instant.now(), target).let {
            if (it.isNegative) Duration.ofMinutes(1) else it
        }
        val request = OneTimeWorkRequestBuilder<InstrumentRecheckWorker>()
            .setInitialDelay(delay)
            .build()
        WorkManager.getInstance(context).enqueueUniqueWork(
            InstrumentRecheckWorker.UNIQUE_NAME,
            ExistingWorkPolicy.REPLACE,
            request,
        )
    }
}
