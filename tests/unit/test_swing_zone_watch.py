"""Buy-window transition and playbook-alert tests for the swing zone watcher."""

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
# tested window: [50*0.85, 50*1.05] = [42.5, 52.5]


def closes(price: float, session: int = 1000) -> dict:
    return {"XYZ": ("2026-07-23", price, session)}


class TestBuyWindowTransitions:
    def test_entry_fires_once(self):
        state: dict = {}
        alerts = watch.evaluate([ENTRY], state, closes(52.0), "2026-07-23")
        assert len(alerts) == 1 and "BUY WINDOW" in alerts[0]
        # still inside window next session: no re-fire
        assert watch.evaluate([ENTRY], state, closes(51.0, 1001), "2026-07-23") == []

    def test_window_boundaries(self):
        assert watch.evaluate([ENTRY], {}, closes(52.51), "?") == []  # above buy level
        assert len(watch.evaluate([ENTRY], {}, closes(52.49), "?")) == 1
        assert watch.evaluate([ENTRY], {}, closes(42.0), "?") == []  # below stop: broken

    def test_exit_and_reentry_respects_cooldown(self):
        state: dict = {}
        assert len(watch.evaluate([ENTRY], state, closes(52.0, 1000), "?")) == 1
        assert watch.evaluate([ENTRY], state, closes(60.0, 1003), "?") == []  # exit
        assert watch.evaluate([ENTRY], state, closes(52.0, 1005), "?") == []  # cooldown
        assert watch.evaluate([ENTRY], state, closes(60.0, 1008), "?") == []
        assert len(watch.evaluate([ENTRY], state, closes(52.0, 1015), "?")) == 1

    def test_missing_symbol_data_is_graceful(self):
        assert watch.evaluate([ENTRY], {}, {}, "?") == []


class TestPlaybookAlert:
    def test_alert_contains_full_playbook_and_is_ascii(self):
        text = watch.zone_alert_text(ENTRY, 52.0, "2026-07-23")
        assert all(ord(ch) < 128 for ch in text)
        assert "BUY WINDOW" in text
        assert "buy at <= 52.50" in text
        assert "SELL TARGET ~114.00" in text  # 120 * 0.95
        assert "below 42.50" in text  # hard exit
        assert "TIME EXIT" in text
        assert "no guarantees" in text
        assert "+119%" in text  # upside from 52 to 114
