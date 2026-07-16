"""Paper-trading sessions and read-only API over the prepared environment."""

from __future__ import annotations

import pandas as pd
import pytest

from edgestack.config import EdgeStackConfig
from edgestack.data.calendar import TradingCalendar
from edgestack.data.catalog import DataCatalog
from edgestack.exceptions import DataError
from edgestack.paper.broker import SimulatedBroker, get_broker
from edgestack.paper.session import load_state, run_session
from edgestack.pipelines import load_feature_frame, run_signals_generate


def _last_thursday(cfg: EdgeStackConfig) -> pd.Timestamp:
    features = load_feature_frame(cfg)
    dates = pd.to_datetime(features["date"]).drop_duplicates().sort_values()
    return dates[dates.dt.dayofweek == 3].iloc[-2]  # not the very last session


def test_broker_is_always_simulated(prepared: EdgeStackConfig) -> None:
    broker = get_broker(prepared)
    assert isinstance(broker, SimulatedBroker)
    assert broker.is_live is False


def test_paper_session_full_cycle(prepared: EdgeStackConfig, capsys) -> None:
    cfg = prepared
    catalog = DataCatalog(cfg)
    calendar = TradingCalendar(cfg.data.calendar)

    signal_day = _last_thursday(cfg)
    run_signals_generate(cfg, as_of=signal_day.date())
    capsys.readouterr()

    entry_day = calendar.next_session(signal_day.date())
    report = run_session(cfg, entry_day.date())
    assert "PAPER session" in report
    assert "not advice" in report
    assert "opened" in report  # Thursday signals -> Friday entries

    state = load_state(catalog, cfg)
    assert state.positions, "expected an open paper position"
    position = state.positions[0]
    assert position.side.value == "LONG"
    assert position.quantity >= 1
    assert position.predicted_probability > 0
    assert state.cash < cfg.paper.initial_cash  # cash consumed by the entry

    # A 1-session horizon exits at the next session's open.
    exit_day = calendar.next_session(entry_day.date())
    report2 = run_session(cfg, exit_day.date())
    assert "closed" in report2
    state2 = load_state(catalog, cfg)
    assert state2.trades, "expected a completed paper trade"
    trade = state2.trades[-1]
    assert trade.exit_reason in ("time_exit", "stop_loss", "target")
    assert trade.predicted_net_return != 0

    # Sessions cannot be double-processed.
    with pytest.raises(DataError, match="already processed"):
        run_session(cfg, exit_day.date())

    # Paper sessions are audited.
    assert len(catalog.audit_events("paper_session")) == 2


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
    statuses = client.get("/monitoring/edges").json()
    assert isinstance(statuses, list) and statuses
