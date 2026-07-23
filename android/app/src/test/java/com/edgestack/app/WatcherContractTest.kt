package com.edgestack.app

import com.edgestack.app.core.AppJson
import com.edgestack.app.domain.model.OilSurgeV1
import com.edgestack.app.domain.model.TrancheWatchV1
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Decode realistic /watchers payloads (copied from real artifacts) leniently. */
class WatcherContractTest {

    private val trancheJson = """
        {
          "run_date": "2026-07-23",
          "file_modified_at": "2026-07-23T10:00:00+00:00",
          "stale": false,
          "run_status": "ok",
          "go_alerts_enabled": false,
          "breadth": {"count": 1, "total": 6, "names": "ACN-, CTSH-, EPAM-, DXC+, IBM-, IT-"},
          "wind": {"status": "READY", "score": 1, "unknown_future_field": [1, 2]},
          "symbols": [
            {
              "symbol": "CTSH",
              "as_of": "2026-07-22",
              "close": 65.2,
              "sma50": 70.1,
              "T1": {"fired": true, "detail": "down3=False IBS=0.16 (<0.2 fires)"},
              "T2": {"fired": false, "detail": "waiting"},
              "T3": {"fired": false, "detail": "below"},
              "REL": {"fired": false, "detail": "rel-z=-0.5"},
              "CAL": null,
              "window_open": false,
              "GO": 37,
              "earnings": "2026-07-29"
            }
          ],
          "paper_book": {"trades": [
            {"symbol": "CTSH", "trigger": "T1", "eur": 300, "signal_date": "2026-07-22",
             "fill_price": null}
          ]},
          "disclaimer": "Research output only."
        }
    """.trimIndent()

    private val oilJson = """
        {
          "state": {
            "phase": "WATCHING",
            "last_session": "2026-07-23",
            "cooldown_left": 0,
            "episode": {
              "shock_date": "2026-07-08",
              "shock_kind": "DAY",
              "shock_ret": 0.0437,
              "pre_shock_close": 70.44,
              "post_shock_high": 89.96,
              "sessions": 4,
              "dip_cooldown": 0,
              "dips": 1
            },
            "intraday": {"ISHOCK:2026-07-23": 1},
            "tickets_enabled": false
          },
          "file_modified_at": "2026-07-23T10:00:00+00:00",
          "stale": false,
          "study_verdict": {"verdict": "FAIL", "rules_passed": [], "batch_id": "abc",
                            "trials": 24},
          "dip_tickets_enabled": false,
          "latest_closes": {"CL=F": {"date": "2026-07-23", "close": 89.96},
                            "BNO": {"date": "2026-07-22", "close": 51.42}},
          "paper_book": null,
          "disclaimer": "Research output only."
        }
    """.trimIndent()

    @Test
    fun trancheDecodesWithUnknownKeysAndNulls() {
        val watch = AppJson.decodeFromString(TrancheWatchV1.serializer(), trancheJson)
        assertEquals("2026-07-23", watch.runDate)
        assertFalse(watch.stale)
        assertFalse(watch.goAlertsEnabled)
        val ctsh = watch.symbols.single()
        assertTrue(ctsh.t1!!.fired)
        assertFalse(ctsh.t2!!.fired)
        assertNull(ctsh.cal)
        assertEquals(37, ctsh.go)
        assertEquals(4, ctsh.triggers().size)
        val trade = watch.paperBook!!.trades.single()
        assertNull(trade.fillPrice) // pending fill must stay null, not 0.0
        assertEquals(300.0, trade.eur, 1e-9)
    }

    @Test
    fun oilDecodesPhaseVerdictAndCloses() {
        val oil = AppJson.decodeFromString(OilSurgeV1.serializer(), oilJson)
        assertEquals("WATCHING", oil.state.phase)
        assertEquals(1, oil.state.episode!!.dips)
        assertEquals(4, oil.state.episode!!.sessions)
        assertEquals("FAIL", oil.studyVerdict!!.verdict)
        assertFalse(oil.dipTicketsEnabled)
        assertNull(oil.paperBook)
        assertEquals(89.96, oil.latestCloses.getValue("CL=F").close, 1e-9)
    }

    @Test
    fun minimalPayloadsFallBackToDefaults() {
        val oil = AppJson.decodeFromString(OilSurgeV1.serializer(), """{"state": {}}""")
        assertEquals("IDLE", oil.state.phase)
        assertNull(oil.state.episode)
        assertTrue(oil.stale)
        val watch = AppJson.decodeFromString(TrancheWatchV1.serializer(), "{}")
        assertTrue(watch.symbols.isEmpty())
    }
}
