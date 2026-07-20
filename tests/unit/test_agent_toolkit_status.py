"""Operational freshness reporting for the JSON agent toolkit."""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from scripts import agent_toolkit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_parquet(path: Path, column: str, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({column: [value]}).to_parquet(path, index=False)


def _healthy_root(tmp_path: Path, now: datetime) -> Path:
    policy = tmp_path / "configs" / "policies" / "baseline-diversified-v1.yaml"
    policy.parent.mkdir(parents=True)
    policy.write_text(
        """weights:
  - {symbol: SPY, asset_kind: ETF}
  - {symbol: TLT, asset_kind: ETF}
  - {symbol: SHY, asset_kind: ETF}
  - {symbol: GLD, asset_kind: ETF}
""",
        encoding="utf-8",
    )

    for symbol in ("SPY", "TLT", "SHY", "GLD", "ACN"):
        _write_parquet(
            tmp_path / "data" / "curated" / "prices" / f"{symbol}.parquet",
            "date",
            pd.Timestamp("2026-07-17"),
        )

    run_id = "test-run"
    _write_json(
        tmp_path / "artifacts" / "recommendations" / "current.json",
        {"run_id": run_id, "session": "2026-07-17"},
    )
    _write_json(
        tmp_path / "artifacts" / "recommendations" / "runs" / run_id / "publication.json",
        {"published_at": "2026-07-19T12:00:00Z"},
    )
    _write_json(
        tmp_path / "artifacts" / "tranche_watch.json",
        {
            "run_status": "ok",
            "run_date": "2026-07-20",
            "go_alerts_enabled": False,
            "breadth": {"count": 4, "total": 6},
            "events": [
                {
                    "symbol": "ACN",
                    "trigger": "REVIEW",
                    "as_of": "2026-07-20",
                    "detail": "test review",
                }
            ],
            "symbols": [
                {
                    "symbol": "ACN",
                    "T1": {"fired": False},
                    "T2": {"fired": False},
                    "T3": {"fired": False},
                    "REL": {"fired": True},
                    "CAL": "seasonal window",
                    "window_open": True,
                    "GO": 72,
                }
            ],
        },
    )
    _write_json(
        tmp_path / "data" / "curated" / "universe_snapshots" / "2026-07-20.json",
        {
            "date": "2026-07-20",
            "membership_source": "sp500_change_log",
            "sp500_members": ["AAPL", "MSFT"],
            "catalog_active": ["AAPL", "MSFT", "SPY"],
        },
    )

    for interval in ("15m", "60m"):
        for symbol in agent_toolkit.INTRADAY_COLLECTOR_SYMBOLS:
            _write_parquet(
                tmp_path / "data" / "curated" / "intraday" / interval / f"{symbol}.parquet",
                "timestamp",
                pd.Timestamp("2026-07-17T20:00:00Z"),
            )

    refresh_timestamp = now.timestamp()
    for symbol in agent_toolkit.EARNINGS_COLLECTOR_SYMBOLS:
        path = tmp_path / "data" / "curated" / "events" / f"{symbol}.parquet"
        _write_parquet(path, "accepted_at", pd.Timestamp("2026-06-01T12:00:00Z"))
        os.utime(path, (refresh_timestamp, refresh_timestamp))
    return tmp_path


def test_status_resolves_publication_and_reports_all_health_sections(tmp_path: Path) -> None:
    now = datetime(2026, 7, 20, 14, tzinfo=UTC)
    root = _healthy_root(tmp_path, now)

    report = agent_toolkit.build_status_report(root, today=date(2026, 7, 20), now=now)

    assert report["health"] == "healthy"
    assert report["publication"] == {
        "run_id": "test-run",
        "session": "2026-07-17",
        "published_at": "2026-07-19T12:00:00Z",
        "age_hours": 26.0,
        "session_lag": 1,
        "stale": False,
    }
    assert report["market_data"]["required_etfs"] == ["SPY", "TLT", "SHY", "GLD"]
    assert set(report["last_bar"]) == {"SPY", "TLT", "SHY", "GLD", "ACN"}
    assert report["tranche_watch"]["fired"]["ACN"] == ["REL", "CAL", "WINDOW"]
    assert report["tranche_watch"]["run_status"] == "ok"
    assert report["tranche_watch"]["global_active"] == ["SECTOR"]
    assert report["tranche_watch"]["events"][0]["trigger"] == "REVIEW"
    assert report["tranche_watch"]["trigger_types"] == list(agent_toolkit.WATCHER_TRIGGER_TYPES)
    assert report["collectors"]["universe_snapshot"]["status"] == "healthy"
    assert report["collectors"]["intraday"]["status"] == "healthy"
    assert report["collectors"]["earnings"]["status"] == "healthy"
    assert report["issues"] == []


def test_status_marks_session_and_daily_job_lag_with_actionable_issues(tmp_path: Path) -> None:
    initial_now = datetime(2026, 7, 20, 14, tzinfo=UTC)
    root = _healthy_root(tmp_path, initial_now)
    stale_now = datetime(2026, 7, 22, 14, tzinfo=UTC)

    report = agent_toolkit.build_status_report(
        root,
        today=date(2026, 7, 22),
        now=stale_now,
    )

    assert report["health"] == "unhealthy"
    assert report["publication"]["session_lag"] == 3
    assert report["publication"]["stale"] is True
    assert report["tranche_watch"]["age_days"] == 2
    assert report["tranche_watch"]["stale"] is True
    assert report["market_data"]["stale_required_etfs"] == ["SPY", "TLT", "SHY", "GLD"]
    assert report["collectors"]["universe_snapshot"]["stale"] is True
    assert report["collectors"]["intraday"]["status"] == "stale"
    assert {issue["code"] for issue in report["issues"]} >= {
        "required_daily_bars_stale",
        "publication_stale",
        "watcher_stale",
        "universe_snapshot_stale",
        "intraday_collector_stale",
    }


def test_status_does_not_count_weekends_as_missing_sessions(tmp_path: Path) -> None:
    now = datetime(2026, 7, 19, 14, tzinfo=UTC)
    root = _healthy_root(tmp_path, now)

    report = agent_toolkit.build_status_report(root, today=date(2026, 7, 19), now=now)

    assert report["market_data"]["symbols"]["SPY"]["session_lag"] == 0
    assert report["collectors"]["intraday"]["intervals"]["15m"]["stale_symbols"] == []


def test_status_reports_missing_outputs_instead_of_raising(tmp_path: Path) -> None:
    policy = tmp_path / "configs" / "policies" / "baseline-diversified-v1.yaml"
    policy.parent.mkdir(parents=True)
    policy.write_text(
        "weights:\n  - {symbol: SPY, asset_kind: ETF}\n",
        encoding="utf-8",
    )
    now = datetime(2026, 7, 20, 14, tzinfo=UTC)

    report = agent_toolkit.build_status_report(
        tmp_path,
        today=date(2026, 7, 20),
        now=now,
    )

    assert report["health"] == "unhealthy"
    assert report["publication"]["stale"] is True
    assert report["tranche_watch"]["status"] == "missing"
    assert report["collectors"]["universe_snapshot"]["status"] == "missing"
    assert report["collectors"]["intraday"]["status"] == "stale"
    assert report["collectors"]["earnings"]["status"] == "stale"
    assert {issue["code"] for issue in report["issues"]} >= {
        "required_daily_bars_stale",
        "publication_missing",
        "watcher_status_missing",
        "universe_snapshot_missing",
        "intraday_collector_stale",
        "earnings_collector_stale",
    }
