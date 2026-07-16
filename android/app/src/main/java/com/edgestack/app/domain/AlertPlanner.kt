package com.edgestack.app.domain

import java.time.LocalDate
import java.time.LocalDateTime
import java.time.LocalTime
import java.time.YearMonth
import java.time.ZoneId
import java.time.ZonedDateTime

enum class AlertType {
    BUY_AT_CLOSE,        // generic "buy near the close" reminder
    TOM_WINDOW_START,    // penultimate session: ToM bump applies from next session
    SEPTEMBER_DERISK,    // last August session: cut to 0.5x from next session
    FEB_SNIPER,          // last January session: XLV/V buy at today's close
    NOV_WORST_DAY_FLAT,  // 6th Nov trading day: go flat at close (td7 next)
}

data class PlannedAlert(
    val type: AlertType,
    val fireAt: ZonedDateTime,   // anchored 15:45 America/New_York
    val title: String,
    val message: String,
)

data class AlertSettings(
    val buyAtClose: Boolean = true,
    val turnOfMonth: Boolean = true,
    val septemberDerisk: Boolean = true,
    val febSniper: Boolean = true,
    val novWorstDay: Boolean = true,
)

/**
 * Pure calendar-date logic: which alerts fire on [today]. Times anchor at
 * 15:45 America/New_York (15 min before the NYSE close) — NOT a fixed Berlin
 * clock, which would land after the close during the ~2-3 weeks each year when
 * US and EU DST diverge (late Oct / early Nov, exactly when ToM and the Nov
 * worst-day alerts matter).
 */
object AlertPlanner {

    val MARKET_ZONE: ZoneId = ZoneId.of("America/New_York")
    val ALERT_TIME: LocalTime = LocalTime.of(15, 45)

    fun fireAt(d: LocalDate): ZonedDateTime =
        ZonedDateTime.of(LocalDateTime.of(d, ALERT_TIME), MARKET_ZONE)

    fun alertsFor(
        calendar: TradingCalendar,
        today: LocalDate,
        settings: AlertSettings = AlertSettings(),
    ): List<PlannedAlert> {
        if (!calendar.isSession(today)) return emptyList()
        val out = mutableListOf<PlannedAlert>()
        val ym = YearMonth.from(today)
        val at = fireAt(today)

        if (settings.turnOfMonth && calendar.tradingDayFromMonthEnd(today) == 2) {
            out += PlannedAlert(
                AlertType.TOM_WINDOW_START, at,
                "Turn-of-month window starts",
                "Next session is the last of the month: overlay adds +0.5x " +
                    "through the first 3 sessions of ${ym.plusMonths(1).month}.",
            )
        }
        if (settings.septemberDerisk && today.monthValue == 8 &&
            calendar.isLastSessionOfMonth(today)
        ) {
            out += PlannedAlert(
                AlertType.SEPTEMBER_DERISK, at,
                "September de-risk",
                "Last August session: overlay cuts exposure to 0.5x for " +
                    "September (worst month, 12 of 13 instruments).",
            )
        }
        if (settings.febSniper && today.monthValue == 1 &&
            calendar.isLastSessionOfMonth(today)
        ) {
            out += PlannedAlert(
                AlertType.FEB_SNIPER, at,
                "Feb-1 sniper: buy at today's close",
                "First February session is the year's best single day " +
                    "(XLV: 89% positive years; SPY t=3.3). Buy near the close.",
            )
        }
        if (settings.novWorstDay && today.monthValue == 11 &&
            calendar.tradingDayOfMonth(today) == 6
        ) {
            out += PlannedAlert(
                AlertType.NOV_WORST_DAY_FLAT, at,
                "Nov worst-day: go flat at close",
                "Next session is the 7th November trading day - the worst " +
                    "single day of the year (SPY & QQQ agree). Overlay goes flat.",
            )
        }
        if (settings.buyAtClose && out.isEmpty() &&
            calendar.isTurnOfMonthWindow(today)
        ) {
            out += PlannedAlert(
                AlertType.BUY_AT_CLOSE, at,
                "Buy window: near the close",
                "Turn-of-month session. Scheduled buys are statistically best " +
                    "in the last hour (overnight effect).",
            )
        }
        return out
    }
}
