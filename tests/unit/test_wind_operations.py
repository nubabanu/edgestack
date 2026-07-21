from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edgestack.data.calendar import TradingCalendar
from edgestack.research.market_wind import COMPONENTS, WindSnapshotV2
from edgestack.research.wind_execution_shadow import advance_shadow, new_shadow_state


def _load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _ready_snapshot(*, score: int, as_of: date, target: date) -> WindSnapshotV2:
    votes = dict.fromkeys(COMPONENTS, 0)
    votes["trend_vol_regime"] = score
    return WindSnapshotV2(
        status="READY",
        as_of=as_of,
        for_session=target,
        score=score,
        votes=votes,
    )


def test_execution_shadow_defers_exactly_one_exchange_session_and_is_idempotent() -> None:
    state = new_shadow_state(registered_on=date(2023, 4, 5))
    snapshot = _ready_snapshot(score=-1, as_of=date(2023, 4, 5), target=date(2023, 4, 6))
    first_bars = pd.DataFrame(
        {"open": [100.0]}, index=pd.DatetimeIndex([pd.Timestamp("2023-04-06")])
    )

    pending = advance_shadow(state, first_bars, snapshot)
    replay = advance_shadow(pending, first_bars, snapshot)

    assert len(pending["events"]) == 1
    assert replay == pending
    assert pending["events"][0]["delayed_session"] == "2023-04-10"
    assert pending["events"][0]["status"] == "PENDING"

    completed_bars = pd.DataFrame(
        {"open": [100.0, 98.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2023-04-06"), pd.Timestamp("2023-04-10")]),
    )
    completed = advance_shadow(pending, completed_bars, snapshot)

    assert completed["events"][0]["status"] == "COMPLETED"
    assert completed["events"][0]["fill_improvement_bps"] > 0
    assert completed["metrics"]["completed_events"] == 1
    assert completed["action"] == "NO_ACTION"


def test_execution_shadow_observes_nonnegative_sessions_without_creating_events() -> None:
    state = new_shadow_state(registered_on=date(2023, 4, 5))
    snapshot = _ready_snapshot(score=2, as_of=date(2023, 4, 5), target=date(2023, 4, 6))
    bars = pd.DataFrame({"open": [100.0]}, index=pd.DatetimeIndex([pd.Timestamp("2023-04-06")]))

    updated = advance_shadow(state, bars, snapshot)

    assert updated["observed_sessions"] == ["2023-04-06"]
    assert updated["events"] == []


def test_execution_shadow_rejects_manifest_tampering() -> None:
    state = new_shadow_state(registered_on=date(2023, 4, 5))
    state["manifest"]["score_condition"] = "score < 2"
    snapshot = _ready_snapshot(score=0, as_of=date(2023, 4, 5), target=date(2023, 4, 6))
    bars = pd.DataFrame({"open": [100.0]}, index=pd.DatetimeIndex([pd.Timestamp("2023-04-06")]))

    with pytest.raises(ValueError, match="manifest mismatch"):
        advance_shadow(state, bars, snapshot)


def test_execution_shadow_review_gate_requires_sessions_and_events() -> None:
    state = new_shadow_state(registered_on=date(2023, 1, 1))
    state["observed_sessions"] = [f"prior-{index:03d}" for index in range(251)]
    state["events"] = [
        {
            "event_id": f"event-{index:02d}",
            "status": "COMPLETED",
            "immediate_session": f"2022-01-{index + 1:02d}",
            "fill_improvement_bps": 2.0,
        }
        for index in range(30)
    ]
    snapshot = _ready_snapshot(score=0, as_of=date(2023, 4, 5), target=date(2023, 4, 6))
    bars = pd.DataFrame({"open": [100.0]}, index=pd.DatetimeIndex([pd.Timestamp("2023-04-06")]))

    updated = advance_shadow(state, bars, snapshot)

    assert updated["metrics"]["prospective_sessions"] == 252
    assert updated["metrics"]["completed_events"] == 30
    assert updated["metrics"]["review_eligible"] is True
    assert updated["metrics"]["result"] == "POSITIVE_DIAGNOSTIC"
    assert updated["metrics"]["action"] == "NO_ACTION"


def test_watcher_payload_is_structured_non_actionable_and_exchange_aware(monkeypatch) -> None:
    watcher = _load_script("tranche_watch_test", "scripts/tranche_watch.py")
    sessions = TradingCalendar().sessions(date(2022, 1, 3), date(2023, 4, 6))
    close = pd.Series(np.linspace(100.0, 140.0, len(sessions)), index=sessions)
    bars = pd.DataFrame({"close": close, "adj_close": close})
    monkeypatch.setattr(watcher, "load", lambda _symbol: bars)

    payload = watcher.market_wind_score(evaluated_on=date(2023, 4, 6))

    assert payload["schema_version"] == 2
    assert payload["status"] == "READY"
    assert payload["for_session"] == "2023-04-10"
    assert payload["label"] == "DESCRIPTIVE_ONLY"
    assert payload["evidence_status"] == "HISTORICAL_DIAGNOSTIC_ONLY"
    assert payload["action"] == "NO_ACTION"
    assert "WIND" not in watcher.PUBLIC_TRIGGER_TYPES


def test_watcher_failure_returns_unavailable_payload(monkeypatch) -> None:
    watcher = _load_script("tranche_watch_failure_test", "scripts/tranche_watch.py")

    def fail(_symbol: str):
        raise OSError("missing")

    monkeypatch.setattr(watcher, "load", fail)
    payload = watcher.market_wind_score(evaluated_on=date(2023, 4, 6))

    assert payload["status"] == "UNAVAILABLE"
    assert payload["score"] is None
    assert payload["action"] == "NO_ACTION"
    assert payload["reasons"] == ["WATCHER_FAILED:missing"]


def test_leveraged_retirement_settles_pending_once_and_archives(
    monkeypatch, tmp_path: Path
) -> None:
    retirement = _load_script("wind_paper_retirement_test", "scripts/wind_paper_book.py")
    state_path = tmp_path / "wind_paper_book.json"
    archive_dir = tmp_path / "archive"
    state_path.write_text(
        json.dumps(
            {
                "registered": "2026-07-21",
                "sailor_3x": {"equity": 10_000.0, "pending": None, "trades": []},
                "barrier_20x": {
                    "equity": 5_000.0,
                    "status": "active",
                    "pending": {"decided_at": "2026-07-20", "score": 2},
                    "trades": [],
                },
            }
        ),
        encoding="utf-8",
    )
    bars = pd.DataFrame(
        {
            "open": [100.0, 99.0],
            "low": [99.0, 98.0],
            "close": [100.0, 101.0],
            "adj_close": [100.0, 101.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-20"), pd.Timestamp("2026-07-21")]),
    )
    monkeypatch.setattr(retirement, "STATE_PATH", state_path)
    monkeypatch.setattr(retirement, "ARCHIVE_DIR", archive_dir)
    monkeypatch.setattr(retirement, "_bars", lambda: bars)

    assert retirement.main() == 0
    first = json.loads(state_path.read_text(encoding="utf-8"))
    assert retirement.main() == 0
    second = json.loads(state_path.read_text(encoding="utf-8"))

    assert first == second
    assert second["status"] == "RETIRED"
    assert second["sailor_3x"]["status"] == "RETIRED"
    assert second["barrier_20x"]["status"] == "RETIRED"
    assert second["barrier_20x"]["pending"] is None
    assert len(second["barrier_20x"]["trades"]) == 1
    assert len(list(archive_dir.glob("*.json"))) == 1


def test_legacy_cost_accounting_charges_each_exposure_change() -> None:
    confluence = _load_script("confluence_cost_test", "scripts/confluence_timing.py")
    hold = pd.Series([0.0, 1.0, 1.0, 0.0])
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])

    result = confluence.net_returns(hold, returns, cost=0.0002)

    assert result.tolist() == [0.0, 0.0098, 0.01, -0.0002]


def test_legacy_reproduction_preserves_and_reports_dataset_edge_bug() -> None:
    confluence = _load_script("confluence_legacy_test", "scripts/confluence_timing.py")
    sessions = TradingCalendar().sessions(date(2023, 3, 10), date(2023, 3, 20))
    close = pd.Series(np.linspace(100.0, 105.0, len(sessions)), index=sessions)
    bars = pd.DataFrame({"close": close, "adj": close})

    legacy = confluence.legacy_component_votes(bars)
    corrected = confluence.component_votes(bars)

    assert legacy.iloc[0]["turn_of_month"] == 1
    assert legacy.iloc[-1]["turn_of_month"] == 1
    assert corrected["turn_of_month"].eq(0).all()
