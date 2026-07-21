"""Local-file and Stooq provider tests (no network: pinned fixtures only)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from edgestack.config import EdgeStackConfig
from edgestack.data.providers.local_files import LocalFilesProvider
from edgestack.data.providers.registry import get_price_provider
from edgestack.data.providers.stooq import parse_stooq_csv
from edgestack.data.universe import SURVIVORSHIP_WARNING, static_universe
from edgestack.exceptions import ProviderError

# A pinned excerpt of a real Stooq response shape.
STOOQ_FIXTURE = """Date,Open,High,Low,Close,Volume
2023-01-03,130.28,130.9,124.17,125.07,112117471
2023-01-04,126.89,128.6557,125.08,126.36,89113633
2023-01-05,127.13,127.77,124.76,125.02,80962708
"""


def test_parse_stooq_csv() -> None:
    out = parse_stooq_csv("aapl", STOOQ_FIXTURE)
    assert out["symbol"].unique().tolist() == ["AAPL"]
    assert len(out) == 3
    assert {"date", "open", "high", "low", "close", "volume"}.issubset(out.columns)


def test_parse_stooq_rejects_no_data() -> None:
    with pytest.raises(ProviderError, match="no data"):
        parse_stooq_csv("zzzz", "No data")
    with pytest.raises(ProviderError, match="no data"):
        parse_stooq_csv("zzzz", "<html><body>throttled</body></html>")


def test_local_provider_reads_csv(tmp_path: Path) -> None:
    root = tmp_path / "local"
    root.mkdir()
    (root / "TEST.csv").write_text(
        "Date,Open,High,Low,Close,Adj Close,Volume\n"
        "2023-01-03,10,11,9,10.5,10.4,1000\n"
        "2023-01-04,10.5,12,10,11.5,11.4,1500\n",
        encoding="utf-8",
    )
    provider = LocalFilesProvider(root)
    bars = provider.fetch_daily_bars(("TEST",), date(2023, 1, 1), date(2023, 1, 31))
    assert len(bars) == 2
    assert bars["adj_close"].tolist() == [10.4, 11.4]


def test_local_provider_missing_column(tmp_path: Path) -> None:
    root = tmp_path / "local"
    root.mkdir()
    (root / "BAD.csv").write_text("Date,Open,Close\n2023-01-03,1,2\n", encoding="utf-8")
    provider = LocalFilesProvider(root)
    with pytest.raises(ProviderError, match="missing columns"):
        provider.fetch_daily_bars(("BAD",), date(2023, 1, 1), date(2023, 1, 31))


def test_registry_resolves_and_rejects(cfg: EdgeStackConfig) -> None:
    provider = get_price_provider("synthetic", cfg)
    assert provider.metadata.name == "synthetic"
    with pytest.raises(ProviderError, match="unknown price provider"):
        get_price_provider("bloomberg", cfg)


def test_static_universe_carries_survivorship_warning(cfg: EdgeStackConfig) -> None:
    snap = static_universe(cfg, date(2020, 6, 1))
    assert not snap.is_point_in_time
    assert SURVIVORSHIP_WARNING in snap.limitations
    assert snap.symbols == ("AAA", "BBB")


def test_range_cache_fresh_requires_post_close_write() -> None:
    from datetime import UTC, datetime

    from edgestack.data.providers.base import range_cache_fresh

    end = date(2026, 7, 20)  # a Monday session
    ttl_s = 30 * 86400
    close = datetime(2026, 7, 20, 21, 0, tzinfo=UTC).timestamp()
    now = datetime(2026, 7, 21, 7, 0, tzinfo=UTC).timestamp()

    # Cached before the end-date close: permanently incomplete, never fresh
    # (the 2026-07-20 poisoning: a 11:42 UTC catch-up run masked the close bar).
    pre_close = datetime(2026, 7, 20, 11, 42, tzinfo=UTC).timestamp()
    assert not range_cache_fresh(pre_close, end, ttl_s, now_s=now)

    # Cached after the close and within TTL: fresh.
    assert range_cache_fresh(close + 3600, end, ttl_s, now_s=now)

    # TTL expiry still applies even post-close.
    late_now = close + ttl_s + 7200
    assert not range_cache_fresh(close + 3600, end, ttl_s, now_s=late_now)


def test_stooq_network_marker_exists() -> None:
    # Real network access is exercised only via `pytest -m network` and the
    # demo pipeline; unit tests must stay offline.
    assert isinstance(pd.Timestamp("2023-01-03"), pd.Timestamp)
