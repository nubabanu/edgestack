"""Backtest metrics and report rendering (JSON + self-contained HTML)."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from string import Template
from typing import Any

import numpy as np
import pandas as pd

from edgestack.backtest.ledger import Ledger
from edgestack.data.catalog import DataCatalog, atomic_write_bytes
from edgestack.types import CostScenario
from edgestack.validation.bootstrap import block_bootstrap_ci
from edgestack.validation.metrics import (
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    var_es,
)

DISCLAIMER = (
    "Research backtest on historical data. Results carry survivorship bias when "
    "point-in-time universes are unavailable, and past performance does not "
    "guarantee future results. Not investment advice."
)


def compute_metrics(ledger: Ledger, *, rng: np.random.Generator, n_boot: int = 1000) -> dict:
    equity = ledger.equity_frame()
    trades = ledger.trades_frame()
    out: dict = {"n_sessions": len(equity), "n_trades": len(trades)}
    if equity.empty:
        return out

    curve = equity["equity"].to_numpy()
    returns = np.diff(curve) / curve[:-1]
    returns = returns[np.isfinite(returns)]
    years = max(len(returns) / 252.0, 1e-9)

    out["initial_equity"] = float(curve[0])
    out["final_equity"] = float(curve[-1])
    out["cumulative_return"] = float(curve[-1] / curve[0] - 1.0)
    out["cagr"] = float((curve[-1] / curve[0]) ** (1.0 / years) - 1.0)
    if len(returns) >= 3:
        out["annualized_volatility"] = float(returns.std(ddof=1) * np.sqrt(252))
        out["sharpe"] = sharpe_ratio(returns)
        out["sortino"] = sortino_ratio(returns)
        out["max_drawdown"] = max_drawdown(returns)
        var, es = var_es(returns)
        out["daily_var_95"] = var
        out["daily_es_95"] = es
        calmar_denom = abs(out["max_drawdown"])
        out["calmar"] = float(out["cagr"] / calmar_denom) if calmar_denom > 0 else None
        lo, hi = block_bootstrap_ci(returns, rng=rng, block_length=20, n_boot=n_boot, stat=np.mean)
        out["mean_daily_return_ci"] = [float(lo), float(hi)]
        sr_lo, sr_hi = block_bootstrap_ci(
            returns,
            rng=rng,
            block_length=20,
            n_boot=n_boot,
            stat=lambda x: float(x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else 0.0,
        )
        out["daily_sharpe_ci"] = [float(sr_lo), float(sr_hi)]
    out["total_borrow_paid"] = float(equity["borrow_paid"].sum())
    out["avg_gross_exposure"] = float((equity["gross_exposure"] / equity["equity"]).mean())

    if not trades.empty:
        pnl = trades["net_pnl"].to_numpy()
        out["hit_rate"] = hit_rate(pnl) if len(pnl) >= 2 else None
        out["profit_factor"] = profit_factor(pnl) if len(pnl) >= 2 else None
        out["expectancy"] = float(pnl.mean())
        winners, losers = pnl[pnl > 0], pnl[pnl < 0]
        out["avg_winner"] = float(winners.mean()) if len(winners) else 0.0
        out["avg_loser"] = float(losers.mean()) if len(losers) else 0.0
        holding = (
            pd.to_datetime(trades["exit_session"]) - pd.to_datetime(trades["entry_session"])
        ).dt.days
        out["avg_holding_days"] = float(holding.mean())
        out["exit_reasons"] = trades["exit_reason"].value_counts().to_dict()
        out["long_trades"] = int((trades["side"] == "LONG").sum())
        out["short_trades"] = int((trades["side"] == "SHORT").sum())
    return out


_HTML = Template(
    """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>EdgeStack backtest $run_id</title>
<style>
body{font-family:system-ui,sans-serif;margin:2rem;max-width:60rem}
table{border-collapse:collapse;margin:1rem 0}
td,th{border:1px solid #ccc;padding:.35rem .7rem;text-align:right}
th{background:#f0f0f0}.warn{background:#fff3cd;padding:.75rem;border:1px solid #ffc107}
</style></head><body>
<h1>EdgeStack backtest report</h1>
<p>run <code>$run_id</code> — scenario <b>$scenario</b> — generated $generated</p>
<div class="warn">$disclaimer</div>
<h2>Metrics</h2>
<table><tr><th>metric</th><th>value</th></tr>$metric_rows</table>
<h2>Equity curve (sampled)</h2>
<table><tr><th>session</th><th>equity</th><th>gross exposure</th></tr>$equity_rows</table>
</body></html>"""
)


def save_backtest_report(
    catalog: DataCatalog, ledger: Ledger, metrics: dict, scenario: CostScenario
) -> tuple[str, Path]:
    run_id = uuid.uuid4().hex[:12]
    out_dir = catalog.data_dir / "reports" / "backtests" / run_id
    generated = datetime.now(UTC).isoformat(timespec="seconds")

    payload = {
        "run_id": run_id,
        "generated_at": generated,
        "cost_scenario": scenario.value,
        "config_hash": catalog.cfg.config_hash(),
        "disclaimer": DISCLAIMER,
        "metrics": metrics,
        "trades": ledger.trades_frame().to_dict(orient="records"),
    }
    atomic_write_bytes(out_dir / "report.json", json.dumps(payload, indent=2, default=str).encode())

    metric_rows = "".join(
        f"<tr><td style='text-align:left'>{k}</td><td>{_fmt(v)}</td></tr>"
        for k, v in metrics.items()
    )
    equity = ledger.equity_frame()
    step = max(1, len(equity) // 40)

    def equity_html_row(row: Any) -> str:
        session = pd.Timestamp(row.session)
        return (
            f"<tr><td>{session.date()}</td><td>{float(row.equity):,.0f}</td>"
            f"<td>{float(row.gross_exposure):,.0f}</td></tr>"
        )

    equity_rows = (
        "".join(equity_html_row(row) for row in equity.iloc[::step].itertuples(index=False))
        if not equity.empty
        else ""
    )
    html = _HTML.substitute(
        run_id=run_id,
        scenario=scenario.value,
        generated=generated,
        disclaimer=DISCLAIMER,
        metric_rows=metric_rows,
        equity_rows=equity_rows,
    )
    atomic_write_bytes(out_dir / "report.html", html.encode("utf-8"))

    with catalog.connect() as con:
        con.execute(
            "INSERT INTO backtests VALUES (?, ?, ?, ?, ?)",
            [
                run_id,
                datetime.now(UTC),
                catalog.cfg.config_hash(),
                scenario.value,
                json.dumps(payload, default=str),
            ],
        )
    return run_id, out_dir


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:,.4f}"
    return str(value)
