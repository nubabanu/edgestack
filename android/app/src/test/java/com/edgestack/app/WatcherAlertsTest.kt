package com.edgestack.app

import com.edgestack.app.domain.WatcherAlerts
import com.edgestack.app.domain.model.OilEpisodeV1
import com.edgestack.app.domain.model.OilStateV1
import com.edgestack.app.domain.model.OilSurgeV1
import com.edgestack.app.domain.model.OilVerdictV1
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class WatcherAlertsTest {

    private fun surge(phase: String, dips: Int = 0, ticketed: Boolean = false) = OilSurgeV1(
        state = OilStateV1(
            phase = phase,
            episode = if (phase == "WATCHING") {
                OilEpisodeV1(shockDate = "2026-07-08", shockRet = 0.094, sessions = 4, dips = dips)
            } else {
                null
            },
        ),
        studyVerdict = OilVerdictV1(verdict = "FAIL"),
        dipTicketsEnabled = ticketed,
    )

    @Test
    fun firstSyncIsSilent() {
        assertTrue(WatcherAlerts.oilAlerts(null, surge("WATCHING", dips = 3)).isEmpty())
    }

    @Test
    fun idleToWatchingFiresShockAlert() {
        val alerts = WatcherAlerts.oilAlerts(surge("IDLE"), surge("WATCHING"))
        assertEquals(1, alerts.size)
        assertTrue(alerts.single().first.contains("dip watch armed"))
    }

    @Test
    fun watchingToWatchingWithoutNewDipIsSilent() {
        assertTrue(
            WatcherAlerts.oilAlerts(surge("WATCHING", dips = 1), surge("WATCHING", dips = 1))
                .isEmpty(),
        )
    }

    @Test
    fun newDipFiresWithDisplayOnlyWording() {
        val alerts = WatcherAlerts.oilAlerts(surge("WATCHING", dips = 0), surge("WATCHING", dips = 1))
        assertEquals(1, alerts.size)
        assertTrue(alerts.single().second.contains("DISPLAY-ONLY"))
    }

    @Test
    fun ticketedDipMentionsPaperTicketInstead() {
        val alerts = WatcherAlerts.oilAlerts(
            surge("WATCHING", dips = 0, ticketed = true),
            surge("WATCHING", dips = 1, ticketed = true),
        )
        assertTrue(alerts.single().second.contains("Paper ticket"))
    }

    @Test
    fun regimeExpiryToCooldownIsSilent() {
        assertTrue(
            WatcherAlerts.oilAlerts(surge("WATCHING", dips = 2), surge("COOLDOWN")).isEmpty(),
        )
    }
}
