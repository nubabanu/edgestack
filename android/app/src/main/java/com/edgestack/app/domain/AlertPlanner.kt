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
    THANKSGIVING_WED,    // Tuesday before Thanksgiving: buy close, hold Wed
    DEC_PRE_CHRISTMAS,   // 14th Dec trading day: buy close, hold td15
    RED_CLOSE_SNIPER,    // data-driven (worker): red close in calm uptrend
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
    val thanksgiving: Boolean = true,
    val decPreChristmas: Boolean = true,
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

    /** Next [count] NAMED sniper windows on/after [from] (planning view). */
    fun upcoming(
        calendar: TradingCalendar,
        from: LocalDate,
        count: Int = 3,
        settings: AlertSettings = AlertSettings(),
    ): List<PlannedAlert> {
        val out = mutableListOf<PlannedAlert>()
        var d: LocalDate? = if (calendar.isSession(from)) from
                            else calendar.nextSession(from)
        var steps = 0
        while (d != null && out.size < count && steps < 420) {
            out += alertsFor(calendar, d, settings)
                .filter { it.type != AlertType.BUY_AT_CLOSE }
            d = calendar.nextSession(d)
            steps++
        }
        return out.take(count)
    }

    /** Session before the session before Thanksgiving (4th Thursday of Nov). */
    fun tuesdayBeforeThanksgiving(calendar: TradingCalendar, year: Int): LocalDate? {
        var d = LocalDate.of(year, 11, 1)
        var thursdays = 0
        while (true) {
            if (d.dayOfWeek.value == 4) {
                thursdays++
                if (thursdays == 4) break
            }
            d = d.plusDays(1)
        }
        val wed = calendar.sessionOnOrBefore(d.minusDays(1)) ?: return null
        return calendar.sessionOnOrBefore(wed.minusDays(1))
    }

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
        if (settings.thanksgiving && today.monthValue == 11 &&
            today == tuesdayBeforeThanksgiving(calendar, today.year)
        ) {
            out += PlannedAlert(
                AlertType.THANKSGIVING_WED, at,
                "Thanksgiving sniper: buy at today's close",
                "Tomorrow (Wed before Thanksgiving) is one of the year's most " +
                    "reliable sessions: QQQ 81% / SPY 70% positive. Sell at " +
                    "Wednesday's close.",
            )
        }
        if (settings.decPreChristmas && today.monthValue == 12 &&
            calendar.tradingDayOfMonth(today) == 14
        ) {
            out += PlannedAlert(
                AlertType.DEC_PRE_CHRISTMAS, at,
                "Pre-Christmas sniper: buy at today's close",
                "The 15th December trading day is historically strong " +
                    "(SPY +43 bps, 73% positive years). Sell at tomorrow's close.",
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
