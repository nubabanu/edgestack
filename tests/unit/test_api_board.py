"""GET /board serves artifacts/live_board.json; 404 when not generated."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from edgestack.api.app import create_app


@pytest.fixture()
def client(cfg, tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.chdir(tmp_path)  # /board resolves artifacts/ relative to cwd
    return TestClient(create_app(cfg))


def test_board_missing_is_404(client: TestClient) -> None:
    resp = client.get("/board")
    assert resp.status_code == 404
    assert "live board not generated" in resp.json()["detail"]


def test_paper_missing_is_404(client: TestClient) -> None:
    resp = client.get("/paper")
    assert resp.status_code == 404
    assert "no paper state" in resp.json()["detail"]


def test_master_missing_is_404(client: TestClient) -> None:
    resp = client.get("/master")
    assert resp.status_code == 404
    assert "master signal not generated" in resp.json()["detail"]


def test_master_served(client: TestClient, tmp_path: Path) -> None:
    art = tmp_path / "artifacts"
    art.mkdir(exist_ok=True)
    payload = {"schema_version": 1, "instruments": {"SPY": {
        "ensemble_exposure_next_session": 0.85}}}
    (art / "master_signal.json").write_text(json.dumps(payload), encoding="utf-8")
    resp = client.get("/master")
    assert resp.status_code == 200
    assert resp.json() == payload


def test_board_served_verbatim(client: TestClient, tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "as_of": "2026-07-15",
        "disclaimer": "Research output only. Not investment advice.",
        "regime": {"trend": "UP", "vol": "MEDIUM"},
        "rows": [{"symbol": "NRG", "close": 137.86, "conviction": 33.0,
                  "e_net_10d": 0.0065, "hit": 0.54, "n_edges": 9,
                  "families": 5, "stop": 126.75, "target": 151.75}],
    }
    art = tmp_path / "artifacts"
    art.mkdir(exist_ok=True)
    (art / "live_board.json").write_text(json.dumps(payload), encoding="utf-8")
    resp = client.get("/board")
    assert resp.status_code == 200
    assert resp.json() == payload
