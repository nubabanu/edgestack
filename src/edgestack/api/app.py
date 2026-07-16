"""Read-only API over the research catalog.

Strictly read-only in v1: no endpoint mutates state, exposes secrets or takes
filesystem paths. Research/backtest POST operations are a later, task-queued
extension.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from edgestack import __version__
from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog
from edgestack.discovery.edge_store import current_statuses, load_edges
from edgestack.exceptions import DataError, EdgeStackError
from edgestack.reporting.signal_report import load_report
from edgestack.types import EdgeStatus

DISCLAIMER = "Research output only. Not investment advice."


def create_app(cfg: EdgeStackConfig):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise DataError(
            "FastAPI is not installed; install the api extra: pip install edgestack[api]"
        ) from exc

    app = FastAPI(title="EdgeStack API", version=__version__,
                  description=DISCLAIMER)
    catalog = DataCatalog(cfg)

    def _report(as_of: date | None = None):
        try:
            return load_report(catalog, as_of)
        except DataError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/version")
    def version() -> dict:
        return {"version": __version__, "config_hash": cfg.config_hash(),
                "disclaimer": DISCLAIMER}

    @app.get("/edges")
    def edges() -> list[dict]:
        statuses = current_statuses(catalog).set_index("edge_id")["status"].to_dict()
        all_edges = load_edges(catalog, statuses=tuple(EdgeStatus))
        return [
            {
                "edge_id": e.identity.edge_id,
                "name": e.identity.name,
                "family": e.identity.family.value,
                "direction": e.identity.direction.value,
                "horizon": e.identity.holding_horizon,
                "status": statuses.get(e.identity.edge_id, e.lifecycle.status.value),
                "net_mean_return": e.stats.net_mean_return,
                "q_value": e.stats.q_value,
                "deflated_sharpe": e.stats.deflated_sharpe_ratio,
                "sample_size": e.stats.sample_size,
            }
            for e in all_edges
        ]

    @app.get("/edges/{edge_id}")
    def edge_detail(edge_id: str) -> dict:
        for e in load_edges(catalog, statuses=tuple(EdgeStatus)):
            if e.identity.edge_id == edge_id:
                return json.loads(e.model_dump_json())
        raise HTTPException(status_code=404, detail=f"unknown edge {edge_id}")

    @app.get("/board")
    def board() -> dict:
        path = Path("artifacts") / "live_board.json"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail="live board not generated; run scripts/live_signals.py",
            )
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/paper")
    def paper() -> dict:
        state_path = catalog.artifacts_dir / "paper" / "state.json"
        if not state_path.exists():
            raise HTTPException(
                status_code=404,
                detail="no paper state; run `edgestack paper run` first",
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        history = []
        try:
            rows = catalog.audit_events("paper_session")
            for _, row in rows.iterrows():
                detail = json.loads(row["detail"]) if row["detail"] else {}
                if "equity" in detail:
                    history.append({"date": str(row["reason"]),
                                    "equity": detail["equity"]})
        except Exception:  # audit table is best-effort for the app
            pass
        return {"state": state, "equity_history": history,
                "disclaimer": DISCLAIMER}

    @app.get("/picks")
    def picks() -> dict:
        path = Path("artifacts") / "picks.json"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail="picks not generated; run scripts/make_picks.py",
            )
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/master")
    def master() -> dict:
        path = Path("artifacts") / "master_signal.json"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail="master signal not generated; run scripts/master_signal.py",
            )
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/signals/latest")
    def signals_latest() -> dict:
        return json.loads(_report().model_dump_json())

    @app.get("/signals/{as_of}")
    def signals_by_date(as_of: date) -> dict:
        return json.loads(_report(as_of).model_dump_json())

    @app.get("/candidates/long")
    def candidates_long() -> list[dict]:
        return [json.loads(c.model_dump_json()) for c in _report().long_candidates]

    @app.get("/candidates/short")
    def candidates_short() -> list[dict]:
        return [json.loads(c.model_dump_json()) for c in _report().short_candidates]

    @app.get("/monitoring/edges")
    def monitoring_edges() -> list[dict]:
        return current_statuses(catalog).to_dict(orient="records")

    @app.get("/backtests")
    def backtests() -> list[dict]:
        with catalog.connect() as con:
            rows = con.execute(
                "SELECT run_id, created_at, cost_scenario FROM backtests "
                "ORDER BY created_at DESC"
            ).fetchall()
        return [
            {"run_id": r[0], "created_at": str(r[1]), "cost_scenario": r[2]}
            for r in rows
        ]

    @app.get("/backtests/{run_id}")
    def backtest_detail(run_id: str) -> dict:
        with catalog.connect() as con:
            row = con.execute(
                "SELECT payload FROM backtests WHERE run_id = ?", [run_id]
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        return json.loads(row[0])

    @app.exception_handler(EdgeStackError)
    def _edgestack_error(request, exc: EdgeStackError):  # pragma: no cover
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app
