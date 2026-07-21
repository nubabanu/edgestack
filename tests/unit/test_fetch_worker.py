"""Fetch-worker tests (synthetic provider only, no network)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog
from edgestack.data.fetch_worker import fetch_and_store, main


def test_fetch_and_store_writes_catalog(cfg: EdgeStackConfig) -> None:
    result = fetch_and_store(cfg, ("AAA", "BBB"), date(2018, 1, 2), date(2018, 3, 1))
    assert set(result["written"]) == {"AAA", "BBB"}
    assert all(rows > 0 for rows in result["written"].values())
    assert result["missing"] == []
    assert result["provider_by_symbol"] == {"AAA": "synthetic", "BBB": "synthetic"}
    catalog = DataCatalog(cfg)
    assert (catalog.prices_dir / "AAA.parquet").exists()


def test_main_writes_result_json(cfg: EdgeStackConfig, tmp_path: Path) -> None:
    config_file = tmp_path / "worker.yaml"
    config_file.write_text(
        "paths:\n"
        f"  data_dir: {Path(cfg.paths.data_dir).as_posix()}\n"
        f"  artifacts_dir: {Path(cfg.paths.artifacts_dir).as_posix()}\n"
        "universe:\n"
        "  symbols: [AAA]\n"
        "  source: synthetic\n"
        "validation:\n"
        "  final_test_start: '2020-01-01'\n",
        encoding="utf-8",
    )
    out = tmp_path / "result.json"
    code = main(
        [
            "--config",
            str(config_file),
            "--symbols",
            "aaa",
            "--start",
            "2018-01-02",
            "--end",
            "2018-03-01",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["provider_by_symbol"] == {"AAA": "synthetic"}
    assert result["missing"] == []


def test_update_data_isolated_runs_real_subprocess(cfg: EdgeStackConfig, tmp_path: Path) -> None:
    from scripts import nightly

    config_file = tmp_path / "worker.yaml"
    config_file.write_text(
        "paths:\n"
        f"  data_dir: {Path(cfg.paths.data_dir).as_posix()}\n"
        f"  artifacts_dir: {Path(cfg.paths.artifacts_dir).as_posix()}\n"
        "universe:\n"
        "  symbols: [AAA, BBB]\n"
        "  source: synthetic\n"
        "validation:\n"
        "  final_test_start: '2020-01-01'\n",
        encoding="utf-8",
    )
    summary = nightly.update_data(
        cfg, date(2018, 3, 1), config_path=config_file, isolate_fetch=True
    )
    assert summary["failed_batches"] == []
    # Policy symbols (SPY/TLT/SHY/GLD) join AAA/BBB as required fetches.
    assert summary["provider_by_symbol_counts"] == {"synthetic": 6}
    assert (DataCatalog(cfg).prices_dir / "SPY.parquet").exists()
    # No temp result files left behind.
    assert not list(Path(cfg.paths.artifacts_dir).glob(".fetch_batch_*"))


def test_main_handled_error_writes_error_json(tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    code = main(
        [
            "--config",
            str(tmp_path / "does_not_exist.yaml"),
            "--symbols",
            "AAA",
            "--start",
            "2018-01-02",
            "--end",
            "2018-03-01",
            "--out",
            str(out),
        ]
    )
    assert code == 1
    assert "error" in json.loads(out.read_text(encoding="utf-8"))
