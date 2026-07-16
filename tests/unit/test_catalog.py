"""Catalog, TestPeriodGuard and audit-log tests."""

from __future__ import annotations

import pandas as pd
import pytest

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog, atomic_write_bytes, safe_child_path
from edgestack.exceptions import DataError, TestPeriodLockedError


@pytest.fixture()
def catalog(cfg: EdgeStackConfig, synthetic_panel: pd.DataFrame) -> DataCatalog:
    cat = DataCatalog(cfg)
    cat.write_bars(synthetic_panel, provider="synthetic")
    return cat


def test_round_trip_and_merge(catalog: DataCatalog, synthetic_panel: pd.DataFrame) -> None:
    assert catalog.list_symbols() == ("AAA", "BBB")
    # Re-writing the same data must not duplicate rows.
    catalog.write_bars(synthetic_panel, provider="synthetic")
    with catalog.guard.unlock(reason="test") as key:
        panel = catalog.load_panel(unlock_key=key)
    assert len(panel) == len(synthetic_panel)


def test_guard_truncates_by_default(catalog: DataCatalog) -> None:
    panel = catalog.load_panel()
    assert panel["date"].max() < pd.Timestamp("2020-01-01")


def test_guard_unlock_serves_test_period_and_audits(catalog: DataCatalog) -> None:
    with catalog.guard.unlock(reason="final evaluation") as key:
        panel = catalog.load_panel(unlock_key=key)
    assert panel["date"].max() >= pd.Timestamp("2020-01-01")
    events = catalog.audit_events("test_set_accessed")
    assert len(events) == 1
    assert events.iloc[0]["reason"] == "final evaluation"


def test_guard_key_is_single_use(catalog: DataCatalog) -> None:
    with catalog.guard.unlock(reason="once") as key:
        catalog.load_panel(unlock_key=key)
    with pytest.raises(TestPeriodLockedError):
        catalog.load_panel(unlock_key=key)  # key expired with the context


def test_guard_rejects_made_up_key(catalog: DataCatalog) -> None:
    with pytest.raises(TestPeriodLockedError):
        catalog.load_panel(unlock_key="deadbeef")


def test_guard_requires_reason(catalog: DataCatalog) -> None:
    with pytest.raises(TestPeriodLockedError), catalog.guard.unlock(reason="  "):
        pass


def test_no_data_raises(cfg: EdgeStackConfig) -> None:
    empty = DataCatalog(cfg)
    with pytest.raises(DataError, match="no price data"):
        empty.load_panel()


def test_experiment_registry(catalog: DataCatalog) -> None:
    experiment_id = catalog.record_experiment("discovery", trial_count=None)
    catalog.update_trial_count(experiment_id, 123)
    with catalog.connect() as con:
        row = con.execute(
            "SELECT kind, trial_count, config_hash FROM experiments WHERE experiment_id = ?",
            [experiment_id],
        ).fetchone()
    assert row is not None
    assert row[0] == "discovery"
    assert row[1] == 123
    assert row[2] == catalog.cfg.config_hash()


def test_atomic_write_and_path_safety(tmp_path) -> None:
    target = tmp_path / "sub" / "file.bin"
    atomic_write_bytes(target, b"hello")
    assert target.read_bytes() == b"hello"
    leftovers = [p for p in target.parent.iterdir() if p.suffix == ".tmp"]
    assert not leftovers
    with pytest.raises(DataError, match="unsafe path"):
        safe_child_path(tmp_path, "..\\..\\evil.parquet")
    with pytest.raises(DataError, match="unsafe path"):
        safe_child_path(tmp_path, "nested/..\\../evil.parquet")
    with pytest.raises(DataError, match="unsafe path"):
        safe_child_path(tmp_path, "C:\\evil.parquet")
    assert (
        safe_child_path(tmp_path, "nested\\safe.parquet")
        == (tmp_path / "nested" / "safe.parquet").resolve()
    )


def test_corporate_actions_round_trip_and_change_data_version(catalog: DataCatalog) -> None:
    before = catalog.data_manifest_hash()
    actions = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "date": ["2019-06-03", "2019-09-03"],
            "action_type": ["dividend", "split"],
            "value": [0.25, 2.0],
        }
    )
    catalog.write_corporate_actions(actions, provider="fixture")

    loaded = catalog.load_corporate_actions(("AAA",))
    assert loaded[["action_type", "value"]].to_dict(orient="records") == [
        {"action_type": "dividend", "value": 0.25},
        {"action_type": "split", "value": 2.0},
    ]
    assert catalog.data_manifest_hash() != before


def test_intraday_round_trip_changes_data_version(catalog: DataCatalog) -> None:
    before = catalog.data_manifest_hash()
    timestamps = pd.date_range("2026-07-16T13:30:00Z", periods=3, freq="h")
    bars = pd.DataFrame(
        {
            "symbol": "AAA",
            "timestamp": timestamps,
            "open": [100.0, 101.0, 100.5],
            "high": [101.2, 101.5, 101.0],
            "low": [99.8, 100.4, 99.9],
            "close": [101.0, 100.5, 100.8],
            "volume": [10_000.0, 11_000.0, 9_000.0],
        }
    )

    catalog.write_intraday_bars(bars, provider="fixture")
    loaded = catalog.load_intraday_bars("AAA")

    assert len(loaded) == 3
    assert str(loaded["timestamp"].dt.tz) == "UTC"
    assert catalog.data_manifest_hash() != before
