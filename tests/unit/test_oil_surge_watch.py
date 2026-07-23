"""State-machine and gating tests for the oil surge watcher."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from scripts import oil_surge_watch as osw


def series(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2026-01-05", periods=len(values))
    return pd.Series(values, index=idx)


def fresh_state() -> dict:
    return json.loads(json.dumps(osw.DEFAULT_STATE))


def run_days(state: dict, closes: pd.Series, start: int = 1) -> list[dict]:
    events = []
    for i in range(start, len(closes)):
        upto = closes.iloc[: i + 1]
        events += osw.step_session(state, str(closes.index[i].date()), upto)
    return events


class TestDetectShock:
    def test_four_percent_day_fires(self):
        shock = osw.detect_shock(series([100.0, 104.0]))
        assert shock and shock["kind"] == "DAY" and shock["pre_close"] == 100.0

    def test_just_below_threshold_does_not_fire(self):
        assert osw.detect_shock(series([100.0, 103.9])) is None

    def test_five_session_window_fires_without_any_big_day(self):
        closes = series([100.0, 101.7, 103.4, 105.1, 106.9, 108.7])  # ~1.7%/day
        shock = osw.detect_shock(closes)
        assert shock and shock["kind"] == "WINDOW" and shock["pre_close"] == 100.0

    def test_short_series_returns_none(self):
        assert osw.detect_shock(series([100.0])) is None


class TestStateMachine:
    def test_shock_arms_watching(self):
        state = fresh_state()
        events = run_days(state, series([100.0, 100.5, 105.0]))
        assert [e["type"] for e in events] == ["SHOCK"]
        assert state["phase"] == "WATCHING"
        assert state["episode"]["pre_shock_close"] == 100.5

    def test_dip_fires_once_then_cooldown_suppresses(self):
        # shock to 105, then two consecutive sessions ~2% below the high
        state = fresh_state()
        events = run_days(state, series([100.0, 105.0, 102.9, 102.8]))
        dips = [e for e in events if e["type"] == "DIP"]
        assert len(dips) == 1 and dips[0]["pullback"] == pytest.approx(-0.02, abs=1e-3)

    def test_new_high_then_new_dip_after_cooldown(self):
        state = fresh_state()
        closes = series([100.0, 105.0, 102.9, 104.0, 106.0, 107.0, 105.2])
        events = run_days(state, closes)
        dips = [e for e in events if e["type"] == "DIP"]
        assert len(dips) == 2  # first at 102.9, second from the 107 high

    def test_regime_expires_after_max_sessions(self):
        state = fresh_state()
        flat = [100.0, 105.0] + [105.0] * (osw.REGIME_SESSIONS + 1)
        events = run_days(state, series(flat))
        expired = [e for e in events if e["type"] == "EXPIRED"]
        assert expired and expired[0]["reason"] == "regime elapsed"
        assert state["phase"] in ("COOLDOWN", "IDLE")

    def test_round_trip_expires_early(self):
        state = fresh_state()
        events = run_days(state, series([100.0, 105.0, 99.0]))
        expired = [e for e in events if e["type"] == "EXPIRED"]
        assert expired and expired[0]["reason"] == "round-trip"
        assert state["phase"] == "COOLDOWN"

    def test_cooldown_suppresses_immediate_re_arm_then_re_arms(self):
        state = fresh_state()
        closes = [100.0, 105.0, 99.0]  # shock then round-trip -> cooldown
        closes += [103.5] * osw.POST_REGIME_COOLDOWN  # +4.5% day inside cooldown: ignored
        run_days(state, series(closes))
        assert state["phase"] == "IDLE"
        # next shock after cooldown re-arms
        closes += [108.0]
        state2 = fresh_state()
        events = run_days(state2, series(closes))
        assert [e["type"] for e in events].count("SHOCK") == 2

    def test_reshock_extends_regime(self):
        state = fresh_state()
        flat = [100.0, 105.0] + [105.0] * (osw.REGIME_SESSIONS - 2) + [110.0]
        events = run_days(state, series(flat))
        assert any(e["type"] == "RESHOCK" for e in events)
        assert state["phase"] == "WATCHING"
        assert state["episode"]["sessions"] < osw.REGIME_SESSIONS


class TestTicketGating:
    def _verdict(self, tmp_path, verdict: str, rules: list[str]):
        path = tmp_path / "verdict.json"
        path.write_text(json.dumps({"verdict": verdict, "rules_passed": rules}))
        return path

    def test_constant_off_means_no_ticket_even_with_pass(self, tmp_path, monkeypatch):
        monkeypatch.setattr(osw, "DIP_TICKETS_ENABLED", False)
        monkeypatch.setattr(osw, "VERDICT_PATH", self._verdict(tmp_path, "PASS", ["E2/R3@10d"]))
        assert osw.tickets_enabled() is False

    def test_constant_on_without_pass_file_means_no_ticket(self, tmp_path, monkeypatch):
        monkeypatch.setattr(osw, "DIP_TICKETS_ENABLED", True)
        monkeypatch.setattr(osw, "VERDICT_PATH", tmp_path / "missing.json")
        assert osw.tickets_enabled() is False

    def test_pass_for_other_rule_does_not_enable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(osw, "DIP_TICKETS_ENABLED", True)
        monkeypatch.setattr(osw, "VERDICT_PATH", self._verdict(tmp_path, "PASS", ["E2/R1@20d"]))
        assert osw.tickets_enabled() is False

    def test_both_gates_open_enables(self, tmp_path, monkeypatch):
        monkeypatch.setattr(osw, "DIP_TICKETS_ENABLED", True)
        monkeypatch.setattr(osw, "VERDICT_PATH", self._verdict(tmp_path, "PASS", ["E2/R3@10d"]))
        assert osw.tickets_enabled() is True

    def test_fail_verdict_never_enables(self, tmp_path, monkeypatch):
        monkeypatch.setattr(osw, "DIP_TICKETS_ENABLED", True)
        monkeypatch.setattr(osw, "VERDICT_PATH", self._verdict(tmp_path, "FAIL", ["E2/R3@10d"]))
        assert osw.tickets_enabled() is False


class TestAlertText:
    def test_alert_strings_are_ascii(self, monkeypatch, tmp_path):
        monkeypatch.setattr(osw, "VERDICT_PATH", tmp_path / "missing.json")
        shock = osw.shock_alert_text(
            {"kind": "DAY", "ret": 0.068}, " (BNO +6.2%, OVX 52)", ["Tanker attack"], True
        )
        dip = osw.dip_alert_text(
            {"pullback": -0.021, "session": 3, "shock_date": "2026-07-08", "shock_ret": 0.094},
            False,
        )
        for text in (shock, dip):
            assert all(ord(ch) < 128 for ch in text), text

    def test_unvalidated_dip_alert_is_display_only(self, monkeypatch, tmp_path):
        monkeypatch.setattr(osw, "VERDICT_PATH", tmp_path / "missing.json")
        dip = osw.dip_alert_text(
            {"pullback": -0.021, "session": 3, "shock_date": "2026-07-08", "shock_ret": 0.094},
            False,
        )
        assert "DISPLAY-ONLY" in dip and "ORDER" not in dip
