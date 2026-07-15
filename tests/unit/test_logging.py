"""Logging redaction tests."""

from __future__ import annotations

from edgestack.logging import redact


def test_redacts_api_keys() -> None:
    assert "hunter2" not in redact("connecting with api_key=hunter2 to provider")
    assert "***REDACTED***" in redact("api-key: abc123")


def test_redacts_tokens_and_passwords() -> None:
    assert "tok_123" not in redact("token=tok_123")
    assert "pw" not in redact("password: pw").split("password: ")[1]


def test_leaves_normal_text_alone() -> None:
    msg = "downloaded 2520 rows for AAPL"
    assert redact(msg) == msg
