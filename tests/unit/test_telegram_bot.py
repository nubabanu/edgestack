"""Auth filter, router, and formatter tests for the read-only Telegram bot."""

from __future__ import annotations

import json

from scripts import telegram_bot as bot


def _update(chat_id: int, text: str, update_id: int = 1) -> dict:
    return {"update_id": update_id, "message": {"chat": {"id": chat_id}, "text": text}}


class TestAuthFilter:
    def test_unknown_chat_is_silently_ignored(self, monkeypatch):
        monkeypatch.setattr(bot, "fmt_help", lambda: "help")
        assert bot.handle_update(_update(999, "/help"), frozenset({-100123})) is None

    def test_configured_group_and_list_are_accepted(self, monkeypatch):
        monkeypatch.setattr(bot, "fmt_help", lambda: "help")
        allowed = bot.allowed_chat_ids({"chat_ids": [-100123, 42]})
        assert bot.handle_update(_update(-100123, "/help"), allowed) == (-100123, "help")
        assert bot.handle_update(_update(42, "/help"), allowed) == (42, "help")

    def test_single_chat_id_config(self):
        assert bot.allowed_chat_ids({"chat_id": "-100777"}) == frozenset({-100777})

    def test_non_text_update_is_ignored(self):
        update = {"update_id": 5, "message": {"chat": {"id": 42}, "photo": []}}
        assert bot.handle_update(update, frozenset({42})) is None


class TestRouter:
    def test_commands_map_and_bot_suffix_is_stripped(self, monkeypatch):
        for name in ("fmt_status", "fmt_watch", "fmt_oil", "fmt_paper"):
            monkeypatch.setattr(bot, name, lambda name=name: name)
        assert bot.route("/watch") == "fmt_watch"
        assert bot.route("/watch@EdgeStackBot") == "fmt_watch"
        assert bot.route("/OIL") == "fmt_oil"
        assert bot.route("/status extra words") == "fmt_status"

    def test_unknown_command_gets_help_and_chatter_gets_silence(self):
        assert "read-only" in (bot.route("/foo") or "")
        assert bot.route("hello bot") is None
        assert bot.route("") is None

    def test_handler_exception_becomes_warn_reply(self, monkeypatch):
        def boom():
            raise ValueError("bad artifact")

        monkeypatch.setattr(bot, "fmt_watch", boom)
        reply = bot.route("/watch")
        assert reply is not None and "WARN" in reply and "bad artifact" in reply


class TestFormatters:
    def test_fmt_watch_reads_artifact_and_shows_fired_triggers(self, tmp_path, monkeypatch):
        payload = {
            "run_date": "2026-07-23",
            "breadth": {"count": 1, "total": 6, "names": "ACN-"},
            "symbols": [
                {
                    "symbol": "CTSH",
                    "close": 65.2,
                    "GO": 37,
                    "T1": {"fired": True, "detail": "IBS=0.1 (<0.2 fires)"},
                    "T2": {"fired": False, "detail": "waiting"},
                }
            ],
        }
        path = tmp_path / "tranche_watch.json"
        path.write_text(json.dumps(payload))
        monkeypatch.setattr(bot, "WATCH_PATH", path)
        text = bot.fmt_watch()
        assert "CTSH" in text and "T1 [IBS=0.1 (<0.2 fires)]" in text
        assert "display-only" in text  # GO score labeled honestly
        assert "T2" not in text.split("CTSH")[1].split("|")[0]  # unfired not listed

    def test_fmt_watch_missing_artifact(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bot, "WATCH_PATH", tmp_path / "missing.json")
        assert "no tranche watcher status" in bot.fmt_watch()

    def test_fmt_oil_absent_then_populated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bot, "OIL_STATE_PATH", tmp_path / "missing.json")
        assert "no oil surge state" in bot.fmt_oil()
        state = {
            "phase": "WATCHING",
            "tickets_enabled": False,
            "episode": {
                "shock_date": "2026-07-08",
                "shock_ret": 0.094,
                "sessions": 4,
                "post_shock_high": 89.96,
                "dips": 1,
            },
        }
        state_path = tmp_path / "oil_surge_state.json"
        state_path.write_text(json.dumps(state))
        verdict_path = tmp_path / "verdict.json"
        verdict_path.write_text(json.dumps({"verdict": "FAIL"}))
        monkeypatch.setattr(bot, "OIL_STATE_PATH", state_path)
        monkeypatch.setattr(bot, "OIL_VERDICT_PATH", verdict_path)
        monkeypatch.setattr(bot, "PRICES_DIR", tmp_path)  # no parquets -> closes skipped
        text = bot.fmt_oil()
        assert "WATCHING" in text and "session 4/10" in text
        assert "FAIL" in text and "DISPLAY-ONLY" in text

    def test_fmt_paper_pending_fill_and_missing_oil_book(self, tmp_path, monkeypatch):
        tranche = {"trades": [{"symbol": "CTSH", "trigger": "T1", "eur": 300, "fill_price": None}]}
        path = tmp_path / "paper_tranche.json"
        path.write_text(json.dumps(tranche))
        monkeypatch.setattr(bot, "PAPER_TRANCHE_PATH", path)
        monkeypatch.setattr(bot, "PAPER_OIL_PATH", tmp_path / "missing.json")
        monkeypatch.setattr(bot, "PRICES_DIR", tmp_path)
        text = bot.fmt_paper()
        assert "pending fill" in text and "OIL BOOK: no paper trades" in text
        assert "no live orders" in text

    def test_replies_are_ascii_and_bounded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bot, "WATCH_PATH", tmp_path / "m.json")
        monkeypatch.setattr(bot, "OIL_STATE_PATH", tmp_path / "m.json")
        monkeypatch.setattr(bot, "PAPER_TRANCHE_PATH", tmp_path / "m.json")
        monkeypatch.setattr(bot, "PAPER_OIL_PATH", tmp_path / "m.json")
        for text in (bot.fmt_watch(), bot.fmt_oil(), bot.fmt_paper(), bot.fmt_help()):
            assert all(ord(ch) < 128 for ch in text)
            assert len(text) <= bot.MAX_REPLY_CHARS

    def test_ascii_helper_replaces_and_truncates(self):
        assert bot._ascii("café") == "caf?"
        assert len(bot._ascii("x" * 9999)) == bot.MAX_REPLY_CHARS


class TestOffsetAndPolling:
    def test_offset_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bot, "STATE_PATH", tmp_path / "state.json")
        assert bot.load_offset() == 0
        bot.save_offset(1234)
        assert bot.load_offset() == 1234

    def test_409_conflict_signals_backoff(self, monkeypatch):
        class Resp:
            status_code = 409

        monkeypatch.setattr(bot.requests, "get", lambda *a, **k: Resp())
        updates, conflict = bot.fetch_updates({"bot_token": "t"}, 0)
        assert conflict is True and updates == []

    def test_poll_once_advances_offset_past_ignored_updates(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bot, "STATE_PATH", tmp_path / "state.json")
        stranger = _update(999, "/status", update_id=7)
        monkeypatch.setattr(bot, "fetch_updates", lambda cfg, off: ([stranger], False))
        sent: list = []
        monkeypatch.setattr(bot, "send_reply", lambda cfg, cid, text: sent.append(cid))
        conflict = bot.poll_once({"bot_token": "t"}, frozenset({42}))
        assert conflict is False and sent == []
        assert bot.load_offset() == 8  # stranger's message consumed, never replayed
