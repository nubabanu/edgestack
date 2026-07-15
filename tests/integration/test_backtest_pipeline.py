"""Backtest pipeline over the shared prepared research environment."""

from __future__ import annotations

import json

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog
from edgestack.pipelines import run_backtest, run_report_backtest


def test_backtest_run_and_report(prepared: EdgeStackConfig, capsys) -> None:
    run_backtest(prepared)
    out = capsys.readouterr().out
    assert "out-of-sample intents" in out
    assert "not advice" in out

    catalog = DataCatalog(prepared)
    with catalog.connect() as con:
        row = con.execute(
            "SELECT run_id, payload FROM backtests ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    assert row is not None
    run_id, payload = row[0], json.loads(row[1])
    metrics = payload["metrics"]
    assert metrics["n_trades"] > 30  # the Thursday edge fires often out of sample
    assert "cumulative_return" in metrics
    assert payload["cost_scenario"] == "CONSERVATIVE"

    out_dir = catalog.data_dir / "reports" / "backtests" / run_id
    assert (out_dir / "report.json").exists()
    html = (out_dir / "report.html").read_text(encoding="utf-8")
    assert "EdgeStack backtest report" in html
    assert "survivorship" in html  # disclaimer present

    run_report_backtest(prepared, run_id=run_id)
    out = capsys.readouterr().out
    assert run_id in out
    assert "report.html" in out
