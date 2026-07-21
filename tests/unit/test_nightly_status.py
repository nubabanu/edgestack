"""Nightly status-ledger tests (isolated tmp files only)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import nightly_status


def test_start_record_and_aggregate_ok(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    nightly_status.start_run("2026-07-21", path=path)
    nightly_status.record_stage("nightly", "ok", exit_code=0, duration_s=12.34, path=path)
    nightly_status.record_stage(
        "data_update",
        "ok",
        detail={"provider_by_symbol_counts": {"yahoo": 5, "stooq": 1}},
        path=path,
    )
    overall, code = nightly_status.aggregate(path=path)
    assert (overall, code) == ("ok", 0)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_date"] == "2026-07-21"
    assert data["overall"] == "ok" and data["exit_code"] == 0
    assert data["stages"]["nightly"]["duration_s"] == 12.3
    assert data["stages"]["data_update"]["detail"]["provider_by_symbol_counts"]["stooq"] == 1


def test_rerecord_overwrites_so_retry_wins(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    nightly_status.start_run(path=path)
    nightly_status.record_stage("nightly", "failed", exit_code=1, path=path)
    nightly_status.record_stage("nightly", "ok", exit_code=0, path=path)
    assert nightly_status.aggregate(path=path) == ("ok", 0)


def test_core_failure_beats_auxiliary(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    nightly_status.start_run(path=path)
    nightly_status.record_stage("publish", "failed", path=path)
    nightly_status.record_stage("tranche_watch", "ok", path=path)
    assert nightly_status.aggregate(path=path) == ("failed", 2)


def test_auxiliary_only_failure_is_degraded(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    nightly_status.start_run(path=path)
    nightly_status.record_stage("nightly", "ok", path=path)
    nightly_status.record_stage("tranche_watch", "failed", exit_code=1, path=path)
    assert nightly_status.aggregate(path=path) == ("degraded", 1)


def test_missing_and_corrupt_files_are_tolerated(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    assert nightly_status.load_status(path=missing) == {}
    assert nightly_status.aggregate(path=missing) == ("failed", 2)

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    nightly_status.record_stage("nightly", "ok", path=corrupt)
    assert nightly_status.aggregate(path=corrupt) == ("ok", 0)
