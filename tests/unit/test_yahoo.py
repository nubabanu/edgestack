"""Yahoo chart-payload parsing (pinned fixture; live fetch behind -m network)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from edgestack.data.providers.yahoo import (
    YahooProvider,
    parse_chart_payload,
    parse_corporate_actions,
    parse_intraday_chart_payload,
)
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
                "events": {
                    "dividends": {"1704292200": {"date": 1704292200, "amount": 0.24}},
                    "splits": {
                        "1704378600": {
                            "date": 1704378600,
                            "numerator": 4.0,
                            "denominator": 1.0,
                            "splitRatio": "4:1",
                        }
                    },
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


def test_parse_corporate_actions_preserves_dividends_and_split_ratios() -> None:
    actions = parse_corporate_actions("aapl", FIXTURE)

    assert actions[["action_type", "value"]].to_dict(orient="records") == [
        {"action_type": "dividend", "value": 0.24},
        {"action_type": "split", "value": 4.0},
    ]
    assert actions["symbol"].tolist() == ["AAPL", "AAPL"]


def test_parse_intraday_payload_preserves_timezone_aware_timestamp() -> None:
    bars = parse_intraday_chart_payload("aapl", FIXTURE)

    assert len(bars) == 2
    assert str(bars["timestamp"].dt.tz) == "UTC"
    assert bars["interval_minutes"].unique().tolist() == [60]
    assert bars["symbol"].unique().tolist() == ["AAPL"]
    bars_15 = parse_intraday_chart_payload("aapl", FIXTURE, interval_minutes=15)
    assert bars_15["interval_minutes"].unique().tolist() == [15]


@pytest.mark.parametrize(("interval", "minutes"), [("1m", 1), ("5m", 5)])
def test_intraday_fetch_supports_fine_bars_and_extended_hours(
    tmp_path, monkeypatch: pytest.MonkeyPatch, interval: str, minutes: int
) -> None:
    provider = YahooProvider(tmp_path, timeout=1, max_retries=0, cache_ttl_days=1)
    seen: dict[str, str] = {}

    def fake_request(symbol: str, params: dict[str, str]) -> dict[str, Any]:
        assert symbol == "SPY"
        seen.update(params)
        return FIXTURE

    monkeypatch.setattr(provider, "_request_with_retries", fake_request)
    bars = provider.fetch_intraday_bars(
        ("SPY",),
        date(2024, 1, 2),
        date(2024, 1, 3),
        interval=interval,
        include_prepost=True,
    )

    assert bars["interval_minutes"].unique().tolist() == [minutes]
    assert seen["interval"] == interval
    assert seen["includePrePost"] == "true"


def test_intraday_fetch_rejects_unsupported_interval(tmp_path) -> None:
    provider = YahooProvider(tmp_path, timeout=1, max_retries=0, cache_ttl_days=1)

    with pytest.raises(ProviderError, match="1m, 5m, 15m, or 60m"):
        provider.fetch_intraday_bars(("SPY",), date(2024, 1, 2), date(2024, 1, 3), interval="30m")


def test_one_minute_fetch_limit_counts_both_endpoint_dates(tmp_path) -> None:
    provider = YahooProvider(tmp_path, timeout=1, max_retries=0, cache_ttl_days=1)

    with pytest.raises(ProviderError, match="at most 7 calendar days"):
        provider.fetch_intraday_bars(("SPY",), date(2024, 1, 1), date(2024, 1, 8), interval="1m")


@pytest.mark.network
def test_live_fetch_small_window(tmp_path) -> None:
    provider = YahooProvider(tmp_path, timeout=30, max_retries=1, cache_ttl_days=1)
    bars = provider.fetch_daily_bars(("AAPL",), date(2024, 1, 2), date(2024, 1, 31))
    assert len(bars) >= 15
    assert bars["adj_close"].notna().all()
