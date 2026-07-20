from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from pandas.testing import assert_frame_equal

from edgestack.data.intraday import resample_intraday_bars
from edgestack.data.provider_credentials import ResearchProviderCredentials
from edgestack.data.providers.alpaca import AlpacaHistoricalProvider
from edgestack.exceptions import ProviderError


class _Response:
    def __init__(self, status_code: int, payload: dict[str, Any]):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _Session:
    def __init__(self, responses: list[_Response]):
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def get(self, _url: str, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _bar(timestamp: str, close: float) -> dict[str, Any]:
    return {
        "t": timestamp,
        "o": close - 1,
        "h": close + 1,
        "l": close - 2,
        "c": close,
        "v": 1_000,
    }


def test_alpaca_paginates_clamps_delay_and_hashes_raw_pages(tmp_path: Path) -> None:
    session = _Session(
        [
            _Response(
                200,
                {
                    "bars": {"SPY": [_bar("2026-07-17T13:30:00Z", 620.0)]},
                    "next_page_token": "next",
                },
            ),
            _Response(
                200,
                {"bars": {"SPY": [_bar("2026-07-17T13:31:00Z", 621.0)]}},
            ),
        ]
    )
    provider = AlpacaHistoricalProvider(
        "key-id",
        "secret-value",
        raw_dir=tmp_path,
        max_retries=0,
        session=session,  # type: ignore[arg-type]
        now=lambda: datetime(2026, 7, 20, 14, 16, tzinfo=UTC),
    )
    bars = provider.fetch_intraday_bars(
        ("spy",),
        date(2026, 7, 17),
        date(2026, 7, 20),
        interval="1m",
        include_prepost=True,
    )

    assert len(bars) == 2
    assert session.calls[1]["params"]["page_token"] == "next"
    params = session.calls[0]["params"]
    assert params["feed"] == "sip"
    assert params["adjustment"] == "split"
    assert params["asof"] == "2026-07-20"
    assert params["end"] == "2026-07-20T14:00:00Z"
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert len(provider.last_raw_hashes) == 2
    assert all((tmp_path / f"{digest}.json").exists() for digest in provider.last_raw_hashes)
    assert all("secret-value" not in path.read_text() for path in tmp_path.glob("*.json"))


def test_alpaca_rate_limit_uses_bounded_backoff(tmp_path: Path) -> None:
    session = _Session(
        [
            _Response(429, {}),
            _Response(200, {"bars": {"SPY": [_bar("2024-01-03T14:30:00Z", 470.0)]}}),
        ]
    )
    sleeps: list[float] = []
    provider = AlpacaHistoricalProvider(
        "key",
        "secret",
        raw_dir=tmp_path,
        max_retries=1,
        session=session,  # type: ignore[arg-type]
        now=lambda: datetime(2024, 1, 5, tzinfo=UTC),
        sleep=sleeps.append,
    )
    result = provider.fetch_daily_bars(("SPY",), date(2024, 1, 3), date(2024, 1, 3))
    assert len(result) == 1
    assert sleeps == [1.0]


def test_alpaca_requires_auth_and_refuses_unsafe_range(tmp_path: Path) -> None:
    unsafe = AlpacaHistoricalProvider(
        "key",
        "secret",
        raw_dir=tmp_path,
        max_retries=0,
        now=lambda: datetime(2026, 7, 20, 4, 10, tzinfo=UTC),
    )
    with pytest.raises(ProviderError, match="safety delay"):
        unsafe.fetch_daily_bars(("SPY",), date(2026, 7, 20), date(2026, 7, 20))

    provider = AlpacaHistoricalProvider(
        None,
        None,
        raw_dir=tmp_path,
        max_retries=0,
        now=lambda: datetime(2026, 7, 20, 14, 0, tzinfo=UTC),
    )
    with pytest.raises(ProviderError, match="unconfigured") as failure:
        provider.fetch_daily_bars(("SPY",), date(2026, 7, 17), date(2026, 7, 17))
    assert "None" not in str(failure.value)
    assert not list(tmp_path.glob("*.json"))


def test_raw_pages_are_valid_json(tmp_path: Path) -> None:
    session = _Session([_Response(200, {"bars": {"SPY": [_bar("2024-01-03T14:30:00Z", 470.0)]}})])
    provider = AlpacaHistoricalProvider(
        "key",
        "secret",
        raw_dir=tmp_path,
        max_retries=0,
        session=session,  # type: ignore[arg-type]
        now=lambda: datetime(2024, 1, 5, tzinfo=UTC),
    )
    provider.fetch_daily_bars(("SPY",), date(2024, 1, 3), date(2024, 1, 3))
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["bars"]["SPY"]


def test_intraday_session_filter_resampling_and_raw_writes_are_idempotent(
    tmp_path: Path,
) -> None:
    payload = {
        "bars": {
            "SPY": [
                _bar("2024-01-03T14:29:00Z", 469.0),
                _bar("2024-01-03T14:30:00Z", 470.0),
                _bar("2024-01-03T14:31:00Z", 471.0),
                _bar("2024-01-03T21:00:00Z", 472.0),
            ]
        }
    }
    session = _Session([_Response(200, payload), _Response(200, payload)])
    provider = AlpacaHistoricalProvider(
        "key",
        "secret",
        raw_dir=tmp_path,
        max_retries=0,
        session=session,  # type: ignore[arg-type]
        now=lambda: datetime(2024, 1, 5, tzinfo=UTC),
    )

    first = provider.fetch_intraday_bars(
        ("SPY",), date(2024, 1, 3), date(2024, 1, 3), interval="1m"
    )
    second = provider.fetch_intraday_bars(
        ("SPY",), date(2024, 1, 3), date(2024, 1, 3), interval="1m"
    )
    assert_frame_equal(first, second)
    assert first["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%H:%M").tolist() == [
        "09:30",
        "09:31",
    ]
    assert len(list(tmp_path.glob("*.json"))) == 1

    fifteen_a = resample_intraday_bars(first, 15)
    fifteen_b = resample_intraday_bars(second, 15)
    assert_frame_equal(fifteen_a, fifteen_b)
    assert fifteen_a.iloc[0].to_dict() == {
        "symbol": "SPY",
        "timestamp": first.iloc[0]["timestamp"],
        "interval_minutes": 15,
        "open": 469.0,
        "high": 472.0,
        "low": 468.0,
        "close": 471.0,
        "volume": 2_000,
    }


def test_research_credentials_load_dotenv_without_exposing_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "EDGESTACK_ALPACA_API_KEY_ID",
        "EDGESTACK_ALPACA_API_SECRET_KEY",
        "EDGESTACK_FRED_API_KEY",
        "EDGESTACK_SEC_USER_AGENT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "EDGESTACK_ALPACA_API_KEY_ID=dotenv-key\n"
        "EDGESTACK_ALPACA_API_SECRET_KEY=dotenv-secret\n"
        "EDGESTACK_FRED_API_KEY=fred-secret\n"
        "EDGESTACK_SEC_USER_AGENT=EdgeStack test@example.com\n",
        encoding="utf-8",
    )

    credentials = ResearchProviderCredentials()

    assert credentials.alpaca_key_id == "dotenv-key"
    assert credentials.alpaca_secret_key == "dotenv-secret"
    assert credentials.fred_key == "fred-secret"
    assert credentials.sec_agent == "EdgeStack test@example.com"
    assert "dotenv-secret" not in repr(credentials)
