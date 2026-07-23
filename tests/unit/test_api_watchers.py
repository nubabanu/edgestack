"""Watcher artifact endpoints: tranche + oil surge, read-only and absent-safe."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from edgestack.api.app import create_app
from edgestack.config import EdgeStackConfig


@pytest.fixture()
def client(cfg: EdgeStackConfig) -> TestClient:
    return TestClient(create_app(cfg))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _tranche_payload(run_date: str) -> dict:
    return {
        "run_status": "ok",
        "run_date": run_date,
        "go_alerts_enabled": False,
        "breadth": {"count": 4, "total": 6, "names": "ACN+, CTSH+"},
        "wind": {"status": "READY", "score": 1},
        "symbols": [
            {
                "symbol": "CTSH",
                "as_of": run_date,
                "close": 65.2,
                "T1": {"fired": True, "detail": "down3=True IBS=0.1 (<0.2 fires)"},
                "T2": {"fired": False, "detail": "waiting"},
            }
        ],
        "events": [],
    }


class TestTrancheEndpoint:
    def test_404_before_watcher_ran(self, client: TestClient):
        resp = client.get("/watchers/tranche")
        assert resp.status_code == 404
        assert "not available" in resp.json()["detail"]

    def test_payload_round_trip_with_paper_book(self, client: TestClient, cfg: EdgeStackConfig):
        artifacts = Path(cfg.paths.artifacts_dir)
        today = date.today().isoformat()
        _write(artifacts / "tranche_watch.json", _tranche_payload(today))
        _write(artifacts / "paper_tranche.json", {"trades": [{"symbol": "CTSH", "eur": 300}]})
        body = client.get("/watchers/tranche").json()
        assert body["run_date"] == today
        assert body["stale"] is False
        assert body["symbols"][0]["T1"]["fired"] is True
        assert body["paper_book"]["trades"][0]["eur"] == 300
        assert body["go_alerts_enabled"] is False

    def test_missing_paper_book_is_null_and_old_run_is_stale(
        self, client: TestClient, cfg: EdgeStackConfig
    ):
        artifacts = Path(cfg.paths.artifacts_dir)
        old = (date.today() - timedelta(days=10)).isoformat()
        _write(artifacts / "tranche_watch.json", _tranche_payload(old))
        body = client.get("/watchers/tranche").json()
        assert body["paper_book"] is None
        assert body["stale"] is True


class TestOilSurgeEndpoint:
    def test_404_before_watcher_ran(self, client: TestClient):
        resp = client.get("/watchers/oil-surge")
        assert resp.status_code == 404

    def test_payload_includes_verdict_and_gate(self, client: TestClient, cfg: EdgeStackConfig):
        artifacts = Path(cfg.paths.artifacts_dir)
        _write(
            artifacts / "oil_surge_state.json",
            {
                "phase": "WATCHING",
                "last_session": date.today().isoformat(),
                "episode": {"shock_date": "2026-07-08", "sessions": 4, "dips": 1},
                "tickets_enabled": False,
            },
        )
        _write(
            artifacts / "oil_shock_study_verdict.json",
            {"verdict": "FAIL", "rules_passed": []},
        )
        body = client.get("/watchers/oil-surge").json()
        assert body["state"]["phase"] == "WATCHING"
        assert body["study_verdict"]["verdict"] == "FAIL"
        assert body["dip_tickets_enabled"] is False
        assert body["paper_book"] is None
        assert body["stale"] is False

    def test_latest_closes_from_parquet_and_omitted_when_absent(
        self, client: TestClient, cfg: EdgeStackConfig
    ):
        artifacts = Path(cfg.paths.artifacts_dir)
        _write(artifacts / "oil_surge_state.json", {"phase": "IDLE", "last_session": None})
        prices = Path(cfg.paths.data_dir) / "curated" / "prices"
        prices.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"date": [pd.Timestamp("2026-07-22")], "close": [51.42], "adj_close": [51.42]}
        ).to_parquet(prices / "BNO.parquet", index=False)
        body = client.get("/watchers/oil-surge").json()
        assert body["latest_closes"]["BNO"] == {"date": "2026-07-22", "close": 51.42}
        assert "CL=F" not in body["latest_closes"]
        assert body["stale"] is True  # no last_session -> conservatively stale

    def test_endpoints_do_not_mutate_artifacts(self, client: TestClient, cfg: EdgeStackConfig):
        artifacts = Path(cfg.paths.artifacts_dir)
        _write(artifacts / "oil_surge_state.json", {"phase": "IDLE", "last_session": None})
        _write(artifacts / "tranche_watch.json", _tranche_payload(date.today().isoformat()))
        before = {p.name: p.read_bytes() for p in artifacts.glob("*.json")}
        client.get("/watchers/oil-surge")
        client.get("/watchers/tranche")
        after = {p.name: p.read_bytes() for p in artifacts.glob("*.json")}
        assert before == after
