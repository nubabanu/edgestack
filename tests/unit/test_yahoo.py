"""Yahoo chart-payload parsing (pinned fixture; live fetch behind -m network)."""

from __future__ import annotations

from datetime import date

import pytest

from edgestack.data.providers.yahoo import parse_chart_payload
from edgestack.exceptions import ProviderError

FIXTURE = {
    "chart": {
        "error": None,
        "result": [
            {
                "meta": {"exchangeTimezoneName": "America/New_York"},
                "timestamp": [1704292200, 1704378600, 1704465000],
                "indicators": {
                    "quote": [
                        {
                            "open": [184.22, 182.15, None],
                            "high": [185.88, 183.09, 182.76],
                            "low": [183.43, 180.88, 180.17],
                            "close": [184.25, 181.91, 181.18],
                            "volume": [58414500, 71983600, None],
                        }
                    ],
                    "adjclose": [{"adjclose": [183.20, 180.87, 180.15]}],
                },
            }
        ],
    }
}


def test_parse_chart_payload() -> None:
    df = parse_chart_payload("aapl", FIXTURE)
    # Third bar has a null open and is dropped.
    assert len(df) == 2
    assert df["symbol"].unique().tolist() == ["AAPL"]
    assert str(df["date"].iloc[0].date()) == "2024-01-03"
    assert df["adj_close"].iloc[0] == pytest.approx(183.20)


def test_parse_rejects_error_and_empty() -> None:
    with pytest.raises(ProviderError, match="yahoo error"):
        parse_chart_payload("x", {"chart": {"error": {"code": "Not Found"}}})
    with pytest.raises(ProviderError, match="no data"):
        parse_chart_payload("x", {"chart": {"result": []}})


@pytest.mark.network
def test_live_fetch_small_window(tmp_path) -> None:
    from edgestack.data.providers.yahoo import YahooProvider

    provider = YahooProvider(tmp_path, timeout=30, max_retries=1, cache_ttl_days=1)
    bars = provider.fetch_daily_bars(("AAPL",), date(2024, 1, 2), date(2024, 1, 31))
    assert len(bars) >= 15
    assert bars["adj_close"].notna().all()
