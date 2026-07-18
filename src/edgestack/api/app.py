"""Verified canonical recommendation API and non-actionable research metadata.

The preview POST is stateless sizing/stress computation. It cannot research,
promote, persist, or mutate the atomically published bundle.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from edgestack import __version__
from edgestack.api.contracts import (
    InstrumentAnalysisRequestV2,
    InstrumentRecheckRequestV2,
    PatternLeaderRequestV2,
    RecommendationPreviewRequestV2,
)
from edgestack.api.sniper_contracts import SniperPreviewRequestV2
from edgestack.config import EdgeStackConfig
from edgestack.data.calendar import TradingCalendar
from edgestack.data.catalog import DataCatalog
from edgestack.discovery.edge_store import current_statuses, load_edges
from edgestack.exceptions import DataError, EdgeStackError
from edgestack.recommendation.compatibility import (
    board_projection,
    master_projection,
    picks_projection,
    signals_projection,
)
from edgestack.recommendation.instrument import analyze_instrument, resolve_instrument
from edgestack.recommendation.instrument_schemas import (
    InstrumentAnalysisV2,
    InstrumentRecheckV2,
    PatternLeaderBoardV2,
    PatternLeaderV2,
)
from edgestack.recommendation.oil import build_oil_decision_from_catalog
from edgestack.recommendation.oil_schemas import OilDecisionRequestV2, OilDecisionSnapshotV2
from edgestack.recommendation.schemas import (
    CanonicalRecommendationBundleV2,
    PortfolioRecommendationV2,
)
from edgestack.recommendation.service import (
    CanonicalBundleRepository,
    CanonicalRecommendationService,
)
from edgestack.recommendation.sniper import build_sniper_plan
from edgestack.recommendation.sniper_schemas import SniperPlanV2
from edgestack.recommendation.trade_calendar import compare_recheck
from edgestack.types import EdgeStatus

DISCLAIMER = "Research output only. Not investment advice."


def create_app(cfg: EdgeStackConfig):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise DataError(
            "FastAPI is not installed; install the api extra: pip install edgestack[api]"
        ) from exc

    app = FastAPI(title="EdgeStack API", version=__version__, description=DISCLAIMER)
    catalog = DataCatalog(cfg)
    recommendations = CanonicalRecommendationService(
        CanonicalBundleRepository(catalog.artifacts_dir)
    )

    def _canonical() -> CanonicalRecommendationBundleV2:
        try:
            return recommendations.latest()
        except DataError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def _deprecated(payload: dict | list):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            content=payload,
            headers={
                "Deprecation": "true",
                "Sunset": "Thu, 31 Dec 2026 23:59:59 GMT",
                "Link": '</recommendations/latest>; rel="successor-version"',
                "X-EdgeStack-Canonical": "true",
            },
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/version")
    def version() -> dict:
        return {"version": __version__, "config_hash": cfg.config_hash(), "disclaimer": DISCLAIMER}

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

    @app.get("/recommendations/latest", response_model=CanonicalRecommendationBundleV2)
    def recommendations_latest() -> CanonicalRecommendationBundleV2:
        return _canonical()

    @app.post("/recommendations/preview", response_model=PortfolioRecommendationV2)
    def recommendations_preview(
        request: RecommendationPreviewRequestV2,
    ) -> PortfolioRecommendationV2:
        try:
            return recommendations.preview(
                profile=request.profile,
                state=request.risk_state,
                equity_override=request.equity_override,
                reset_requested=request.reset_requested,
            )
        except DataError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/instruments/analyze", response_model=InstrumentAnalysisV2)
    def instruments_analyze(request: InstrumentAnalysisRequestV2) -> InstrumentAnalysisV2:
        """Analyze one instrument without changing portfolio selection or promotion state."""
        try:
            return _analyze_instrument(request)
        except DataError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/instruments/recheck", response_model=InstrumentRecheckV2)
    def instruments_recheck(request: InstrumentRecheckRequestV2) -> InstrumentRecheckV2:
        """Re-run the server-owned analysis and compare it with the prior signed contract."""
        try:
            current = _analyze_instrument(request.request)
            return compare_recheck(request.previous_analysis, current)
        except DataError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/oil/decision", response_model=OilDecisionSnapshotV2)
    def oil_decision(request: OilDecisionRequestV2) -> OilDecisionSnapshotV2:
        """Build a stateless, paper-only eToro OIL decision snapshot."""
        try:
            return build_oil_decision_from_catalog(cfg, request)
        except DataError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/instruments/pattern-leaders", response_model=PatternLeaderBoardV2)
    def instruments_pattern_leaders(request: PatternLeaderRequestV2) -> PatternLeaderBoardV2:
        """Rank explicitly requested symbols without turning the scan into a recommendation."""
        bundle = _canonical()
        candidates = []
        skipped = []
        searched_cells = 0
        for symbol in request.symbols:
            try:
                analysis = _analyze_instrument(
                    InstrumentAnalysisRequestV2(symbol=symbol, include_news=False)
                )
                calendar = next(
                    (
                        item
                        for item in analysis.tailwind_calendars
                        if item.resolution is request.resolution and item.horizon is request.horizon
                    ),
                    None,
                )
                if calendar is None or not calendar.cells:
                    skipped.append(symbol)
                    continue
                searched_cells += len(calendar.cells)
                candidates.append((analysis.resolution.resolved_symbol, calendar.cells[0]))
            except DataError:
                skipped.append(symbol)
        candidates.sort(key=lambda item: item[1].score.win_score, reverse=True)
        leaders = tuple(
            PatternLeaderV2(
                rank=rank,
                symbol=symbol,
                resolution=request.resolution,
                horizon=request.horizon,
                strongest_slot=cell.model_copy(
                    update={
                        "score": cell.score.model_copy(
                            update={
                                "multiple_testing_adjusted_pvalue": min(
                                    1.0,
                                    cell.score.multiple_testing_adjusted_pvalue
                                    * searched_cells
                                    / cell.score.candidates_ranked,
                                ),
                                "explanation": (
                                    cell.score.explanation
                                    + f" Leader scan adjusted over {searched_cells} cells."
                                ),
                            }
                        )
                    }
                ),
            )
            for rank, (symbol, cell) in enumerate(candidates[: request.limit], start=1)
        )
        return PatternLeaderBoardV2(
            as_of=bundle.as_of,
            resolution=request.resolution,
            horizon=request.horizon,
            searched_symbols=request.symbols,
            skipped_symbols=tuple(skipped),
            searched_cells=searched_cells,
            leaders=leaders,
        )

    def _analyze_instrument(request: InstrumentAnalysisRequestV2) -> InstrumentAnalysisV2:
        bundle = recommendations.latest()
        resolution = resolve_instrument(
            request.symbol,
            requested_kind=request.instrument_kind,
            canonical=bundle,
        )
        current_data_version = catalog.data_manifest_hash()
        if current_data_version != bundle.data_version:
            raise DataError(
                "market data changed after canonical publication; republish before analysis"
            )
        with catalog.guard.unlock(
            reason=(
                "user-selected descriptive instrument analysis; previously accessed period; "
                "not promotion evidence"
            )
        ) as key:
            daily = catalog.load_panel(
                symbols=(resolution.resolved_symbol,),
                end=bundle.session,
                unlock_key=key,
            )
        intended_entry = request.intended_entry_at
        if intended_entry is None and request.intended_entry_date is not None:
            intended_entry = datetime.combine(
                request.intended_entry_date,
                time(9, 30),
                tzinfo=ZoneInfo("America/New_York"),
            )
        return analyze_instrument(
            bundle=bundle,
            resolution=resolution,
            daily_bars=daily,
            intended_entry_at=intended_entry,
            rate_intraday_choice=request.intended_entry_at is not None,
            intraday_bars=catalog.load_intraday_bars(
                resolution.resolved_symbol,
                interval_minutes=60,
                end=bundle.as_of,
            ),
            fifteen_minute_bars=catalog.load_intraday_bars(
                resolution.resolved_symbol,
                interval_minutes=15,
                end=bundle.as_of,
            ),
            timing_artifacts=recommendations.repository.timing_artifacts(),
            news=(
                recommendations.repository.news_evidence(resolution.resolved_symbol)
                if request.include_news
                else ()
            ),
            round_trip_cost_bps=request.round_trip_cost_bps,
        )

    @app.get("/sniper/latest", response_model=SniperPlanV2)
    def sniper_latest() -> SniperPlanV2:
        bundle = _canonical()
        return _sniper_plan(
            account_equity=bundle.default_risk_profile.account_equity,
            max_tolerable_loss=bundle.default_risk_profile.account_equity * 0.0025,
            vehicle="SPY",
        )

    @app.post("/sniper/preview", response_model=SniperPlanV2)
    def sniper_preview(request: SniperPreviewRequestV2) -> SniperPlanV2:
        return _sniper_plan(
            account_equity=request.account_equity,
            max_tolerable_loss=request.max_tolerable_loss,
            vehicle=request.vehicle,
        )

    def _sniper_plan(
        *, account_equity: float, max_tolerable_loss: float, vehicle: str
    ) -> SniperPlanV2:
        try:
            bundle = recommendations.latest()
            if catalog.data_manifest_hash() != bundle.data_version:
                raise DataError("market data changed; republish before sniper evaluation")
            requested = vehicle.strip().upper()
            symbols = tuple(dict.fromkeys(("SPY", requested)))
            with catalog.guard.unlock(
                reason="sniper shadow evaluation; descriptive, not promotion evidence"
            ) as key:
                panel = catalog.load_panel(
                    symbols=symbols,
                    end=bundle.session,
                    unlock_key=key,
                )
            return build_sniper_plan(
                bundle=bundle,
                panel=panel,
                calendar=TradingCalendar(cfg.data.calendar),
                account_equity=account_equity,
                max_tolerable_loss=max_tolerable_loss,
                vehicle=requested,
            )
        except DataError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/board", deprecated=True)
    def board():
        return _deprecated(board_projection(_canonical()))

    @app.get("/paper")
    def paper() -> dict:
        repository = recommendations.repository
        if not repository.pointer_path.exists():
            raise HTTPException(
                status_code=404,
                detail="no canonical paper state; publish a recommendation first",
            )
        try:
            pointer = repository.pointer()
            repository.latest()
            wrapper = json.loads(
                (repository.run_dir(pointer) / "paper_state.json").read_text(encoding="utf-8")
            )
            state = wrapper["payload"]
        except (DataError, OSError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        history = [
            {
                "date": item["session"],
                "equity": item["ending_equity"],
                "actual_fill_return": item["actual_fill_return"],
            }
            for item in state.get("realized_returns", [])
        ]
        return {"state": state, "equity_history": history, "disclaimer": DISCLAIMER}

    @app.get("/picks", deprecated=True)
    def picks():
        return _deprecated(picks_projection(_canonical()))

    @app.get("/master", deprecated=True)
    def master():
        return _deprecated(master_projection(_canonical()))

    @app.get("/signals/latest", deprecated=True)
    def signals_latest():
        return _deprecated(signals_projection(_canonical()))

    @app.get("/signals/{as_of}", deprecated=True)
    def signals_by_date(as_of: date):
        bundle = _canonical()
        if as_of != bundle.as_of.date():
            raise HTTPException(
                status_code=404,
                detail="only the atomically published canonical session is available",
            )
        return _deprecated(signals_projection(bundle))

    @app.get("/candidates/long", deprecated=True)
    def candidates_long():
        _canonical()
        return _deprecated([])

    @app.get("/candidates/short", deprecated=True)
    def candidates_short():
        _canonical()
        return _deprecated([])

    @app.get("/monitoring/edges")
    def monitoring_edges() -> dict:
        from edgestack.paper.canonical import load_monitoring_payload

        try:
            return load_monitoring_payload(recommendations.repository)
        except DataError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/backtests")
    def backtests() -> list[dict]:
        with catalog.connect() as con:
            rows = con.execute(
                "SELECT run_id, created_at, cost_scenario FROM backtests ORDER BY created_at DESC"
            ).fetchall()
        return [{"run_id": r[0], "created_at": str(r[1]), "cost_scenario": r[2]} for r in rows]

    @app.get("/backtests/{run_id}")
    def backtest_detail(run_id: str) -> dict:
        with catalog.connect() as con:
            row = con.execute("SELECT payload FROM backtests WHERE run_id = ?", [run_id]).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        return json.loads(row[0])

    @app.exception_handler(EdgeStackError)
    def _edgestack_error(request, exc: EdgeStackError):  # pragma: no cover
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app
