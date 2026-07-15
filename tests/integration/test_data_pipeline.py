"""End-to-end data pipeline: download (synthetic) -> catalog -> quality."""

from __future__ import annotations

from datetime import date

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog
from edgestack.pipelines import run_data_download, run_data_validate


def test_download_and_validate_synthetic(cfg: EdgeStackConfig, capsys) -> None:
    run_data_download(cfg, date(2018, 1, 1), date(2020, 12, 31))
    out = capsys.readouterr().out
    assert "AAA" in out and "BBB" in out
    assert "survivorship bias" in out  # universe limitation surfaced

    catalog = DataCatalog(cfg)
    assert catalog.list_symbols() == ("AAA", "BBB")
    # download must be audited
    assert len(catalog.audit_events("data_download")) == 1
    # guard boundary respected by default loads
    panel = catalog.load_panel()
    assert str(panel["date"].max().date()) < "2020-01-01"

    run_data_validate(cfg)
    out = capsys.readouterr().out
    assert "passed" in out
