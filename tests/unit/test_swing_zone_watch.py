"""Zone-entry transition and dedupe tests for the swing zone watcher."""

from __future__ import annotations

from scripts import swing_zone_watch as watch

ENTRY = {
    "symbol": "XYZ",
    "theta_pct": 30,
    "trough_zone": 50.0,
    "peak_zone": 120.0,
    "swings_per_year": 2.5,
    "score": 1.5,
}


def closes(price: float, session: int = 1000) -> dict:
    return {"XYZ": ("2026-07-23", price, session)}


class TestZoneTransitions:
    def test_entry_fires_once(self):
        state: dict = {}
        alerts = watch.evaluate([ENTRY], state, closes(54.0), "2026-07-23")
        assert len(alerts) == 1 and "SWING ZONE" in alerts[0]
        # still in zone next session: no re-fire
        alerts = watch.evaluate([ENTRY], state, closes(53.0, 1001), "2026-07-23")
        assert alerts == []

    def test_above_zone_never_fires(self):
        assert watch.evaluate([ENTRY], {}, closes(80.0), "?") == []
        # boundary: zone is trough * 1.10 = 55
        assert watch.evaluate([ENTRY], {}, closes(55.01), "?") == []
        assert len(watch.evaluate([ENTRY], {}, closes(54.99), "?")) == 1

    def test_exit_and_reentry_respects_cooldown(self):
        state: dict = {}
        assert len(watch.evaluate([ENTRY], state, closes(54.0, 1000), "?")) == 1
        assert watch.evaluate([ENTRY], state, closes(60.0, 1003), "?") == []  # exit
        # re-entry within cooldown window: suppressed
        assert watch.evaluate([ENTRY], state, closes(54.0, 1005), "?") == []
        # exit again, then re-entry after cooldown: fires
        assert watch.evaluate([ENTRY], state, closes(60.0, 1008), "?") == []
        assert len(watch.evaluate([ENTRY], state, closes(54.0, 1015), "?")) == 1

    def test_missing_symbol_data_is_graceful(self):
        assert watch.evaluate([ENTRY], {}, {}, "?") == []


class TestAlertText:
    def test_ascii_and_display_only(self):
        text = watch.zone_alert_text(ENTRY, 52.0, "2026-07-23")
        assert all(ord(ch) < 128 for ch in text)
        assert "DISPLAY-ONLY" in text
        assert "ZERO floor-bounce edge" in text
        assert "+131%" in text  # upside to peak zone from 52
