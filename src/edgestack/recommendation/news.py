"""Free per-symbol news headlines for the published news-context slot.

Fetches the Yahoo Finance RSS 2.0 headline feed with a daily local cache,
scores headlines with a deterministic keyword rule, and emits
:class:`NewsEvidenceV2` items. News is publication context only: the schema
pins ``actionable_contribution`` to zero, and a total feed outage yields an
empty tuple rather than blocking the nightly publication.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree

import requests

from edgestack.exceptions import DataError, ProviderError
from edgestack.recommendation.hashing import stable_hash
from edgestack.recommendation.instrument_schemas import NewsEvidenceV2

YAHOO_NEWS_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
NEWS_SOURCE = "yahoo_finance_rss"
FRESH_HOURS = 48.0
MAX_ITEMS_PER_SYMBOL = 5
_POLITE_DELAY_SECONDS = 0.3
_USER_AGENT = "Mozilla/5.0 (research; edgestack/0.1)"

POSITIVE_WORDS = frozenset(
    [
        "beat",
        "beats",
        "surge",
        "surges",
        "soar",
        "soars",
        "upgrade",
        "upgraded",
        "record",
        "rally",
        "rallies",
        "raises",
        "profit",
        "profits",
        "growth",
        "wins",
        "win",
        "approval",
        "approves",
        "expands",
        "expansion",
        "strong",
        "outperform",
        "breakthrough",
        "gains",
        "gain",
        "jumps",
        "jump",
    ]
)
NEGATIVE_WORDS = frozenset(
    [
        "miss",
        "misses",
        "plunge",
        "plunges",
        "sink",
        "sinks",
        "downgrade",
        "downgraded",
        "lawsuit",
        "lawsuits",
        "recall",
        "recalls",
        "fraud",
        "probe",
        "probes",
        "layoff",
        "layoffs",
        "cuts",
        "bankruptcy",
        "halts",
        "halt",
        "warns",
        "warning",
        "falls",
        "fall",
        "drops",
        "drop",
        "weak",
        "underperform",
        "losses",
        "loss",
        "slump",
    ]
)

SentimentLabel = Literal["POSITIVE", "NEGATIVE", "MIXED", "NEUTRAL", "UNSCORED"]


@dataclass(frozen=True)
class RawNewsItem:
    news_id: str
    headline: str
    published_at: str  # ISO-8601 UTC
    url: str | None
    summary: str


def score_headline(text: str) -> SentimentLabel:
    """Deterministic keyword sentiment; zero hits stays UNSCORED, never NEUTRAL."""
    tokens = set(re.findall(r"[a-z']+", text.lower()))
    positive = len(tokens & POSITIVE_WORDS)
    negative = len(tokens & NEGATIVE_WORDS)
    if positive and negative:
        return "MIXED"
    if positive:
        return "POSITIVE"
    if negative:
        return "NEGATIVE"
    return "UNSCORED"


def parse_news_rss(symbol: str, xml_text: str) -> tuple[RawNewsItem, ...]:
    """Parse an RSS 2.0 document; items without a parseable pubDate are skipped."""
    root = ElementTree.fromstring(xml_text)
    items: list[RawNewsItem] = []
    for node in root.iter("item"):
        headline = (node.findtext("title") or "").strip()
        if not headline:
            continue
        raw_date = (node.findtext("pubDate") or "").strip()
        try:
            published = parsedate_to_datetime(raw_date)
        except (TypeError, ValueError):
            continue
        if published.tzinfo is None:
            continue  # timestamps must be zone-aware; never guess an offset
        published_iso = published.astimezone(UTC).isoformat(timespec="seconds")
        url = (node.findtext("link") or "").strip() or None
        guid = (node.findtext("guid") or "").strip()
        news_id = guid or stable_hash(
            {"symbol": symbol, "headline": headline, "published_at": published_iso}
        )
        items.append(
            RawNewsItem(
                news_id=news_id,
                headline=headline,
                published_at=published_iso,
                url=url,
                summary=(node.findtext("description") or "").strip(),
            )
        )
    return tuple(items)


def build_news_evidence(
    symbol: str,
    items: tuple[RawNewsItem, ...],
    *,
    now: datetime,
    fresh_hours: float = FRESH_HOURS,
) -> tuple[NewsEvidenceV2, ...]:
    """Freshness and sentiment are computed at gather time, not cache time."""
    evidence = []
    for item in items[:MAX_ITEMS_PER_SYMBOL]:
        published = datetime.fromisoformat(item.published_at)
        age_hours = max((now - published).total_seconds() / 3600.0, 0.0)
        evidence.append(
            NewsEvidenceV2(
                news_id=item.news_id,
                symbol=symbol,
                headline=item.headline,
                source=NEWS_SOURCE,
                published_at=published,
                url=item.url,
                summary=item.summary,
                sentiment_label=score_headline(item.headline),
                relevance=None,
                is_fresh=age_hours <= fresh_hours,
                age_hours=age_hours,
            )
        )
    return tuple(evidence)


def load_cached_news(path: Path) -> tuple[datetime, tuple[RawNewsItem, ...]] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("cache payload is not an object")
        fetched_at = datetime.fromisoformat(str(payload["fetched_at"]))
        items = tuple(RawNewsItem(**item) for item in payload["items"])
        return fetched_at, items
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError(f"invalid news cache {path}: {exc}") from exc


def fetch_symbol_news(
    symbol: str,
    cache_path: Path,
    *,
    now: datetime,
    timeout_seconds: float,
    session: requests.Session,
) -> tuple[RawNewsItem, ...]:
    cached = load_cached_news(cache_path)
    if cached is not None and cached[0].date() == now.date():
        return cached[1]
    try:
        response = session.get(
            YAHOO_NEWS_RSS_URL,
            params={"s": symbol, "region": "US", "lang": "en-US"},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        items = parse_news_rss(symbol, response.text)
    except (requests.RequestException, ElementTree.ParseError) as exc:
        if cached is not None:
            return cached[1]
        raise ProviderError(f"could not fetch news for {symbol}: {exc}") from exc
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "fetched_at": now.isoformat(timespec="seconds"),
                "items": [asdict(item) for item in items],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    temporary.replace(cache_path)
    return items


def gather_news_evidence(
    symbols: tuple[str, ...],
    cache_dir: Path,
    *,
    now: datetime | None = None,
    timeout_seconds: float = 10.0,
    session: requests.Session | None = None,
) -> tuple[NewsEvidenceV2, ...]:
    """Best-effort gather; a symbol failure or total outage never raises."""
    current = now or datetime.now(UTC)
    client = session or requests.Session()
    # requests ships a default python-requests UA, which the feed rejects —
    # setdefault would keep it, so assign outright.
    client.headers["User-Agent"] = _USER_AGENT
    evidence: list[NewsEvidenceV2] = []
    for index, symbol in enumerate(sorted(set(symbols))):
        if index:
            time.sleep(_POLITE_DELAY_SECONDS)
        try:
            items = fetch_symbol_news(
                symbol,
                cache_dir / f"{symbol}.json",
                now=current,
                timeout_seconds=timeout_seconds,
                session=client,
            )
        except (ProviderError, DataError):
            continue
        evidence.extend(build_news_evidence(symbol, items, now=current))
    return tuple(evidence)
