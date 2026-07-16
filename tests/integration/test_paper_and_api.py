"""Paper-trading sessions and read-only API over the prepared environment."""

from __future__ import annotations

import pandas as pd
import pytest

from edgestack.config import EdgeStackConfig
from edgestack.exceptions import DataError
from edgestack.paper.broker import SimulatedBroker, get_broker
from edgestack.paper.session import run_session
from edgestack.pipelines import load_feature_frame, run_signals_generate


def _last_thursday(cfg: EdgeStackConfig) -> pd.Timestamp:
    features = load_feature_frame(cfg)
    dates = pd.to_datetime(features["date"]).drop_duplicates().sort_values()
    return dates[dates.dt.dayofweek == 3].iloc[-2]  # not the very last session


def test_broker_is_always_simulated(prepared: EdgeStackConfig) -> None:
    broker = get_broker(prepared)
    assert isinstance(broker, SimulatedBroker)
    assert broker.is_live is False


def test_legacy_signal_report_cannot_drive_paper_orders(prepared: EdgeStackConfig, capsys) -> None:
    cfg = prepared
    signal_day = _last_thursday(cfg)
    run_signals_generate(cfg, as_of=signal_day.date())
    capsys.readouterr()

    with pytest.raises(DataError, match="canonical recommendation has not been published"):
        run_session(cfg, signal_day.date())


def test_api_read_only_surface(prepared: EdgeStackConfig) -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from edgestack.api.app import create_app

    client = fastapi_testclient.TestClient(create_app(prepared))

    assert client.get("/health").json() == {"status": "ok"}
    version = client.get("/version").json()
    assert "Not investment advice" in version["disclaimer"]

    edges = client.get("/edges").json()
    assert edges and any(e["status"] in ("VALIDATED", "ACTIVE") for e in edges)
    detail = client.get(f"/edges/{edges[0]['edge_id']}").json()
    assert detail["identity"]["edge_id"] == edges[0]["edge_id"]
    assert client.get("/edges/nope").status_code == 404

    # Legacy candidates fail closed until an atomic V2 bundle is published;
    # the old signal report is no longer an actionable API fallback.
    assert client.get("/recommendations/latest").status_code == 404
    assert client.get("/signals/latest").status_code == 404
    assert client.get("/candidates/long").status_code == 404
    assert client.get("/candidates/short").status_code == 404

    runs = client.get("/backtests").json()
    assert isinstance(runs, list)
    if runs:
        payload = client.get(f"/backtests/{runs[0]['run_id']}").json()
        assert "metrics" in payload
    assert client.get("/monitoring/edges").status_code == 404
