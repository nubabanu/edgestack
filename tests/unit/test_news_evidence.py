"""News-context feeder tests: parsing, scoring, caching, and graceful outage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import requests

from edgestack.exceptions import DataError
from edgestack.recommendation.news import (
    RawNewsItem,
    build_news_evidence,
    fetch_symbol_news,
    gather_news_evidence,
    load_cached_news,
    parse_news_rss,
    score_headline,
)

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item>
    <title>Acme beats estimates as profit surges</title>
    <link>https://example.com/a</link>
    <guid>acme-1</guid>
    <pubDate>Fri, 17 Jul 2026 09:00:00 +0000</pubDate>
    <description>Quarterly results.</description>
  </item>
  <item>
    <title>Regulator opens probe into Acme</title>
    <link>https://example.com/b</link>
    <guid>acme-2</guid>
    <pubDate>Thu, 16 Jul 2026 12:00:00 +0000</pubDate>
  </item>
  <item>
    <title>No date item is skipped</title>
    <link>https://example.com/c</link>
  </item>
</channel></rss>
"""

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _Session:
    def __init__(self, text: str | None = None, error: Exception | None = None) -> None:
        self._text = text
        self._error = error
        self.calls = 0
        self.headers: dict[str, str] = {}

    def get(self, *args: object, **kwargs: object) -> _Response:
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._text is not None
        return _Response(self._text)


def test_parse_extracts_items_and_skips_missing_pubdate() -> None:
    items = parse_news_rss("ACME", RSS)
    assert [item.news_id for item in items] == ["acme-1", "acme-2"]
    assert items[0].published_at == "2026-07-17T09:00:00+00:00"
    assert items[0].url == "https://example.com/a"
    assert items[0].summary == "Quarterly results."


def test_score_headline_deterministic() -> None:
    assert score_headline("Shares SURGE after record beat") == "POSITIVE"
    assert score_headline("Lawsuit and probe hit the company") == "NEGATIVE"
    assert score_headline("Profit surges but probe looms") == "MIXED"
    assert score_headline("Company announces annual meeting") == "UNSCORED"


def test_build_news_evidence_freshness_boundary() -> None:
    fresh = RawNewsItem("a", "h", (NOW - timedelta(hours=47)).isoformat(), None, "")
    stale = RawNewsItem("b", "h", (NOW - timedelta(hours=49)).isoformat(), None, "")
    fresh_out, stale_out = build_news_evidence("SPY", (fresh, stale), now=NOW)
    assert fresh_out.is_fresh and not stale_out.is_fresh
    assert fresh_out.age_hours == pytest.approx(47.0)
    assert fresh_out.actionable_contribution == 0.0


def test_cache_hit_avoids_network(tmp_path: Path) -> None:
    session = _Session(text=RSS)
    cache = tmp_path / "ACME.json"
    first = fetch_symbol_news("ACME", cache, now=NOW, timeout_seconds=5, session=session)
    second = fetch_symbol_news("ACME", cache, now=NOW, timeout_seconds=5, session=session)
    assert session.calls == 1
    assert first == second
    cached = load_cached_news(cache)
    assert cached is not None and len(cached[1]) == 2


def test_fetch_falls_back_to_stale_cache(tmp_path: Path) -> None:
    cache = tmp_path / "ACME.json"
    fetch_symbol_news(
        "ACME", cache, now=NOW - timedelta(days=1), timeout_seconds=5, session=_Session(text=RSS)
    )
    failing = _Session(error=requests.ConnectionError("down"))
    items = fetch_symbol_news("ACME", cache, now=NOW, timeout_seconds=5, session=failing)
    assert len(items) == 2


def test_corrupt_cache_raises_data_error(tmp_path: Path) -> None:
    cache = tmp_path / "ACME.json"
    cache.write_text("[]", encoding="utf-8")
    with pytest.raises(DataError):
        load_cached_news(cache)


def test_gather_never_raises_on_total_outage(tmp_path: Path) -> None:
    failing = _Session(error=requests.ConnectionError("down"))
    result = gather_news_evidence(("SPY", "GLD"), tmp_path, now=NOW, session=failing)
    assert result == ()


def test_gather_keeps_healthy_symbols(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from edgestack.recommendation import news as news_module

    def fake_fetch(symbol: str, cache_path: Path, **kwargs: object) -> tuple[RawNewsItem, ...]:
        if symbol == "BAD":
            raise news_module.ProviderError("boom")
        return (RawNewsItem("x", "Profit surges", NOW.isoformat(), None, ""),)

    monkeypatch.setattr(news_module, "fetch_symbol_news", fake_fetch)
    monkeypatch.setattr(news_module.time, "sleep", lambda _s: None)
    result = gather_news_evidence(("BAD", "SPY"), tmp_path, now=NOW)
    assert [item.symbol for item in result] == ["SPY"]
    assert result[0].sentiment_label == "POSITIVE"
