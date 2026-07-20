"""EdgeStack command-line interface.

Thin wrapper: every command loads + validates config, then delegates to a
pipeline function. Exit codes: 0 success, 1 runtime failure, 2 not implemented
or bad usage.
"""

from __future__ import annotations

import json
import logging as _stdlib_logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import typer

from edgestack import __version__
from edgestack.config import EdgeStackConfig, load_config
from edgestack.exceptions import EdgeStackError
from edgestack.logging import configure, get_logger, log_event

app = typer.Typer(
    name="edgestack",
    help="EdgeStack — statistical edge research platform. Research / paper trading only.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Download and validate market data.", no_args_is_help=True)
features_app = typer.Typer(help="Build feature datasets.", no_args_is_help=True)
edges_app = typer.Typer(help="Discover and validate edges.", no_args_is_help=True)
models_app = typer.Typer(help="Train and calibrate models.", no_args_is_help=True)
signals_app = typer.Typer(help="Generate and rank signal candidates.", no_args_is_help=True)
backtest_app = typer.Typer(help="Run backtests.", no_args_is_help=True)
monitor_app = typer.Typer(help="Monitor edge health and lifecycle.", no_args_is_help=True)
report_app = typer.Typer(help="Produce reports.", no_args_is_help=True)
paper_app = typer.Typer(help="Paper-trading session management.", no_args_is_help=True)
risk_app = typer.Typer(help="Canonical risk-state management.", no_args_is_help=True)
api_app = typer.Typer(help="Serve the read-only API.", no_args_is_help=True)
dashboard_app = typer.Typer(help="Serve the research dashboard.", no_args_is_help=True)
oil_app = typer.Typer(help="Paper-only eToro OIL decision support.", no_args_is_help=True)
research_app = typer.Typer(help="Continuous edge-factory worker controls.", no_args_is_help=True)

for name, sub in [
    ("data", data_app),
    ("features", features_app),
    ("edges", edges_app),
    ("models", models_app),
    ("signals", signals_app),
    ("backtest", backtest_app),
    ("monitor", monitor_app),
    ("report", report_app),
    ("paper", paper_app),
    ("risk", risk_app),
    ("api", api_app),
    ("dashboard", dashboard_app),
    ("oil", oil_app),
    ("research", research_app),
]:
    app.add_typer(sub, name=name)

log = get_logger("cli")

_CONFIG_OPT = typer.Option(None, "--config", "-c", help="Path to YAML config file.")
_READABLE_FILE_ARG = typer.Argument(..., exists=True, dir_okay=False, readable=True)
_STRANGE_MANIFEST_OPT = typer.Option(
    Path("configs/strange_edges.yaml"),
    "--manifest",
    exists=True,
    dir_okay=False,
    readable=True,
    help="Frozen strange-edge campaign manifest.",
)
_PUBLIC_DAILY_THROUGH_OPT = typer.Option(
    "2022-12-30",
    "--through",
    help="Last research date; must precede the configured final holdout.",
)
_CROSS_ASSET_MANIFEST_OPT = typer.Option(
    Path("configs/cross_asset_momentum.yaml"),
    "--manifest",
    exists=True,
    dir_okay=False,
    readable=True,
    help="Frozen cross-asset momentum campaign manifest.",
)
_EARNINGS_FILING_MANIFEST_OPT = typer.Option(
    Path("configs/earnings_filing_drift.yaml"),
    "--manifest",
    exists=True,
    dir_okay=False,
    readable=True,
    help="Frozen filing-time EPS-drift campaign manifest.",
)


def _setup(config_path: Path | None) -> EdgeStackConfig:
    configure()
    cfg = load_config(config_path)
    log_event(
        log,
        _stdlib_logging.INFO,
        "config loaded",
        config_hash=cfg.config_hash()[:12],
        seed=cfg.project.random_seed,
    )
    return cfg


def _run(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except NotImplementedError as exc:
        typer.secho(f"not implemented yet: {exc}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(2) from exc
    except EdgeStackError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


@app.command()
def version() -> None:
    """Print the EdgeStack version."""
    typer.echo(f"edgestack {__version__} (python {sys.version.split()[0]})")


@app.command("config-show")
def config_show(config: Path | None = _CONFIG_OPT) -> None:
    """Validate and print the resolved configuration."""
    cfg = _setup(config)
    typer.echo(json.dumps(cfg.model_dump(mode="json"), indent=2, default=str))
    typer.echo(f"config_hash: {cfg.config_hash()}")


@research_app.command("status")
def research_status(config: Path | None = _CONFIG_OPT) -> None:
    """Show worker, queue, coverage, and campaign-funnel state."""
    cfg = _setup(config)
    from edgestack.data.catalog import DataCatalog
    from edgestack.research.service import ResearchQueryService

    overview = ResearchQueryService(DataCatalog(cfg)).overview()
    typer.echo(overview.model_dump_json(indent=2))


@research_app.command("queue")
def research_queue(config: Path | None = _CONFIG_OPT) -> None:
    """Inspect resumable acquisition and evaluation jobs."""
    cfg = _setup(config)
    from edgestack.data.catalog import DataCatalog
    from edgestack.research.store import ResearchStore

    jobs = ResearchStore(DataCatalog(cfg)).jobs()
    typer.echo(json.dumps([item.model_dump(mode="json") for item in jobs], indent=2, default=str))


@research_app.command("proposal-import")
def research_proposal_import(
    proposal_file: Path = _READABLE_FILE_ARG,
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Register a finite external hypothesis and enqueue its viable data gaps."""
    cfg = _setup(config)
    from edgestack.research.proposals import register_and_plan_proposal
    from edgestack.research.schemas import CandidateProposalV1

    proposal = CandidateProposalV1.model_validate_json(proposal_file.read_text(encoding="utf-8"))
    audit = register_and_plan_proposal(cfg, proposal)
    typer.echo(audit.model_dump_json(indent=2))


@research_app.command("proposal-attempt")
def research_proposal_attempt(
    attempt_file: Path = _READABLE_FILE_ARG,
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Append one immutable hypothesis/query/result/revision audit event."""
    cfg = _setup(config)
    from edgestack.data.catalog import DataCatalog
    from edgestack.research.proposals import ProposalRegistry
    from edgestack.research.schemas import ProposalAttemptV1

    attempt = ProposalAttemptV1.model_validate_json(attempt_file.read_text(encoding="utf-8"))
    audit = ProposalRegistry(DataCatalog(cfg)).append_attempt(attempt)
    typer.echo(audit.model_dump_json(indent=2))


@research_app.command("proposals")
def research_proposals(config: Path | None = _CONFIG_OPT) -> None:
    """Inspect proposal manifests and promotion-boundary audits."""
    cfg = _setup(config)
    from edgestack.data.catalog import DataCatalog
    from edgestack.research.proposals import ProposalRegistry

    registry = ProposalRegistry(DataCatalog(cfg))
    payload = [
        {
            "proposal": proposal.model_dump(mode="json"),
            "audit": registry.audit(proposal.proposal_id).model_dump(mode="json"),
        }
        for proposal in registry.proposals()
    ]
    typer.echo(json.dumps(payload, indent=2, default=str))


@research_app.command("factor-audit")
def research_factor_audit(
    factor_file: Path = _READABLE_FILE_ARG,
    factor_name: str = typer.Option(..., help="Frozen factor name."),
    quantiles: int = typer.Option(5, min=2, max=20),
    group_column: str | None = typer.Option(None, help="Optional sector/group column."),
) -> None:
    """Run non-promotional IC, quantile, turnover, and decay diagnostics."""
    import pandas as pd

    from edgestack.research.factor_diagnostics import analyze_factor

    if factor_file.suffix.lower() == ".parquet":
        frame = pd.read_parquet(factor_file)
    elif factor_file.suffix.lower() == ".csv":
        frame = pd.read_csv(factor_file)
    else:
        raise typer.BadParameter("factor input must be .parquet or .csv")
    diagnostics = analyze_factor(
        frame,
        factor_name=factor_name,
        quantiles=quantiles,
        group_column=group_column,
    )
    typer.echo(diagnostics.model_dump_json(indent=2))


@research_app.command("strange-edges")
def research_strange_edges(
    manifest: Path = _STRANGE_MANIFEST_OPT,
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Acquire a new provider snapshot instead of reusing the frozen request cache.",
    ),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Test every feasible strange-edge trial and expose exact blockers for the rest."""
    cfg = _setup(config)
    from edgestack.research.strange_edges import run_strange_edges_campaign

    report = run_strange_edges_campaign(cfg, manifest_path=manifest, refresh=refresh)
    typer.echo(json.dumps(report, indent=2, default=str))


@research_app.command("public-daily")
def research_public_daily(
    through: str = _PUBLIC_DAILY_THROUGH_OPT,
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Run the bounded public-price daily family without opening the holdout."""
    cfg = _setup(config)
    from edgestack.research.public_daily import run_public_daily_campaign

    try:
        cutoff = date.fromisoformat(through)
    except ValueError as exc:
        raise typer.BadParameter("--through must be YYYY-MM-DD") from exc
    report = run_public_daily_campaign(cfg, through=cutoff)
    typer.echo(json.dumps(report, indent=2, default=str))


@research_app.command("cross-asset-momentum")
def research_cross_asset_momentum(
    manifest: Path = _CROSS_ASSET_MANIFEST_OPT,
    verify_recompute: bool = typer.Option(
        False,
        "--verify-recompute",
        help="Recompute a terminal run and verify its content-addressed result.",
    ),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Evaluate the frozen cross-asset family without opening the holdout."""
    cfg = _setup(config)
    from edgestack.research.cross_asset_momentum import (
        run_cross_asset_momentum_campaign,
    )

    report = run_cross_asset_momentum_campaign(
        cfg,
        manifest_path=manifest,
        verify_recompute=verify_recompute,
    )
    typer.echo(json.dumps(report, indent=2, default=str))


@research_app.command("earnings-filing-drift")
def research_earnings_filing_drift(
    manifest: Path = _EARNINGS_FILING_MANIFEST_OPT,
    verify_recompute: bool = typer.Option(False, "--verify-recompute"),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Evaluate the frozen SEC filing-time EPS family without opening the holdout."""
    cfg = _setup(config)
    from edgestack.research.earnings_filing_drift import run_earnings_filing_campaign

    report = run_earnings_filing_campaign(
        cfg,
        manifest_path=manifest,
        verify_recompute=verify_recompute,
    )
    typer.echo(json.dumps(report, indent=2, default=str))


@research_app.command("sync-opening-state")
def research_sync_opening_state(config: Path | None = _CONFIG_OPT) -> None:
    """Publish forward intraday archive progress to the read-only Edge Lab."""
    cfg = _setup(config)
    from edgestack.research.opening_state import sync_opening_fade_state

    typer.echo(sync_opening_fade_state(cfg).model_dump_json(indent=2))


@research_app.command("pause")
def research_pause(config: Path | None = _CONFIG_OPT) -> None:
    """Pause new research leases without interrupting an active write."""
    cfg = _setup(config)
    from edgestack.research.worker import ResearchWorker

    typer.echo(ResearchWorker(cfg).pause().model_dump_json(indent=2))


@research_app.command("resume")
def research_resume(config: Path | None = _CONFIG_OPT) -> None:
    """Allow the persistent worker to lease jobs again."""
    cfg = _setup(config)
    from edgestack.research.worker import ResearchWorker

    typer.echo(ResearchWorker(cfg).resume().model_dump_json(indent=2))


@research_app.command("run")
def research_run(
    once: bool = typer.Option(False, "--once", help="Run one bounded worker cycle and exit."),
    poll_seconds: float = typer.Option(5.0, min=1.0, help="Continuous-mode polling interval."),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Run continuously at below-normal priority, or execute one resumable cycle."""
    cfg = _setup(config)
    from edgestack.research.worker import ResearchWorker, set_below_normal_priority

    worker = ResearchWorker(cfg)
    set_below_normal_priority()
    if once:
        completed = worker.run_cycle()
        typer.echo(
            json.dumps([item.model_dump(mode="json") for item in completed], indent=2, default=str)
        )
        return
    try:
        worker.run_forever(poll_seconds=poll_seconds)
    except KeyboardInterrupt:
        typer.echo("research worker stopped")


@data_app.command("download")
def data_download(
    start: str = typer.Option("2010-01-04", help="Start date YYYY-MM-DD."),
    end: str = typer.Option(str(date.today()), help="End date YYYY-MM-DD."),
    provider: str = typer.Option(None, help="Provider override (local|stooq|synthetic)."),
    symbols: str = typer.Option(None, help="Comma-separated symbol override."),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Download daily bars into the local catalog."""
    cfg = _setup(config)
    from edgestack.pipelines import run_data_download

    sym = tuple(s.strip().upper() for s in symbols.split(",")) if symbols else None
    _run(
        run_data_download,
        cfg,
        date.fromisoformat(start),
        date.fromisoformat(end),
        provider=provider,
        symbols=sym,
    )


@data_app.command("intraday-download")
def intraday_download(
    symbols: str = typer.Option(..., help="Comma-separated symbols or tradable proxies."),
    start: str = typer.Option(
        str(date.today() - timedelta(days=365)), help="Start date YYYY-MM-DD."
    ),
    end: str = typer.Option(str(date.today()), help="End date YYYY-MM-DD."),
    provider: str = typer.Option("yahoo", help="Intraday-capable provider."),
    interval: str = typer.Option("60m", help="Bar interval: 1m, 5m, 15m, or 60m."),
    include_prepost: bool = typer.Option(
        False, help="Include provider-supplied premarket and after-hours trades."
    ),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Download hourly bars used by day/hour timing analysis."""
    cfg = _setup(config)
    from edgestack.pipelines import run_intraday_download

    wanted = tuple(item.strip().upper() for item in symbols.split(",") if item.strip())
    if not wanted:
        raise typer.BadParameter("at least one symbol is required")
    _run(
        run_intraday_download,
        cfg,
        date.fromisoformat(start),
        date.fromisoformat(end),
        symbols=wanted,
        provider=provider,
        interval=interval,
        include_prepost=include_prepost,
    )


@data_app.command("validate")
def data_validate(config: Path | None = _CONFIG_OPT) -> None:
    """Run data-quality checks over the catalog and print a report."""
    cfg = _setup(config)
    from edgestack.pipelines import run_data_validate

    _run(run_data_validate, cfg)


@features_app.command("build")
def features_build(config: Path | None = _CONFIG_OPT) -> None:
    """Compute all registered features into a versioned Parquet dataset."""
    cfg = _setup(config)
    from edgestack.pipelines import run_features_build

    _run(run_features_build, cfg)


@edges_app.command("discover")
def edges_discover(config: Path | None = _CONFIG_OPT) -> None:
    """Run event studies and conditional rule mining to produce candidate edges."""
    cfg = _setup(config)
    from edgestack.pipelines import run_edges_discover

    _run(run_edges_discover, cfg)


@edges_app.command("validate")
def edges_validate(
    batch: str = typer.Option(None, help="Discovery batch id (default: latest)."),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Walk-forward validate a discovery batch with FDR control."""
    cfg = _setup(config)
    from edgestack.pipelines import run_edges_validate

    _run(run_edges_validate, cfg, batch_id=batch)


@models_app.command("train")
def models_train(config: Path | None = _CONFIG_OPT) -> None:
    """Train baseline and tree models with out-of-fold probability calibration."""
    cfg = _setup(config)
    from edgestack.pipelines import run_models_train

    _run(run_models_train, cfg)


@signals_app.command("generate")
def signals_generate(
    as_of: str = typer.Option(None, "--date", help="Analysis date YYYY-MM-DD (default: latest)."),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Generate ranked long/short candidates and abstentions for a date."""
    cfg = _setup(config)
    from edgestack.pipelines import run_signals_generate

    _run(run_signals_generate, cfg, as_of=date.fromisoformat(as_of) if as_of else None)


@signals_app.command("rank")
def signals_rank(
    top: int = typer.Option(20, help="Show the top N candidates per side."),
    as_of: str = typer.Option(None, "--date", help="Analysis date YYYY-MM-DD (default: latest)."),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Print the ranked candidate tables from the latest signal report."""
    cfg = _setup(config)
    from edgestack.pipelines import run_signals_rank

    _run(run_signals_rank, cfg, top=top, as_of=date.fromisoformat(as_of) if as_of else None)


@backtest_app.command("run")
def backtest_run(
    scenario: str = typer.Option(None, help="Cost scenario override."),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Run the cost-aware backtest over stored signals."""
    cfg = _setup(config)
    from edgestack.pipelines import run_backtest

    _run(run_backtest, cfg, scenario=scenario)


@monitor_app.command("run")
def monitor_run(config: Path | None = _CONFIG_OPT) -> None:
    """Update edge health metrics and lifecycle states."""
    cfg = _setup(config)
    from edgestack.pipelines import run_monitor

    _run(run_monitor, cfg)


@report_app.command("backtest")
def report_backtest(
    run_id: str = typer.Option(None, help="Backtest run id (default: latest)."),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Render HTML + JSON reports for a backtest run."""
    cfg = _setup(config)
    from edgestack.pipelines import run_report_backtest

    _run(run_report_backtest, cfg, run_id=run_id)


@paper_app.command("run")
def paper_run(
    as_of: str = typer.Option(None, "--date", help="Session date YYYY-MM-DD (default: latest)."),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Run one paper-trading session (simulated fills only)."""
    cfg = _setup(config)
    from edgestack.pipelines import run_paper_session

    _run(run_paper_session, cfg, as_of=date.fromisoformat(as_of) if as_of else None)


@risk_app.command("reset-latch")
def risk_reset_latch(config: Path | None = _CONFIG_OPT) -> None:
    """Reset the persisted cash latch only after all eligibility gates pass."""
    cfg = _setup(config)
    from edgestack.recommendation.reset import reset_persisted_risk_state

    publication = reset_persisted_risk_state(cfg)
    typer.echo(f"risk latch reset in canonical run {publication.run_id}")


@oil_app.command("refresh")
def oil_refresh(
    through: str = typer.Option(str(date.today()), help="Refresh through YYYY-MM-DD."),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Refresh free daily, hourly, and 15-minute oil research inputs."""
    cfg = _setup(config)
    from edgestack.recommendation.oil import refresh_oil_data

    _run(refresh_oil_data, cfg, through=date.fromisoformat(through))


@oil_app.command("check")
def oil_check(
    intended_entry_at: str = typer.Option(
        ..., help="Timezone-aware intended entry, for example 2026-07-20T15:30:00+02:00."
    ),
    observed_at: str = typer.Option(..., help="Timezone-aware eToro quote timestamp."),
    bid: float = typer.Option(..., min=0.000001, help="Displayed eToro OIL bid."),
    ask: float = typer.Option(..., min=0.000001, help="Displayed eToro OIL ask."),
    offered_leverage: float = typer.Option(10.0, min=1.0, max=10.0),
    modeled_leverage: float = typer.Option(10.0, min=1.0, max=10.0),
    event_flags: str = typer.Option(
        "",
        help=(
            "Comma-separated manual vetoes: WEEKEND_SUPPLY_ESCALATION, "
            "SHIPPING_DISRUPTION, WTI_ROLLOVER_EXPIRY, BROKER_MAINTENANCE."
        ),
    ),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Create and persist one content-addressed paper-only OIL snapshot."""
    cfg = _setup(config)
    from edgestack.recommendation.oil import (
        build_oil_decision_from_catalog,
        persist_oil_snapshot,
    )
    from edgestack.recommendation.oil_schemas import (
        OilBrokerQuoteV2,
        OilDecisionRequestV2,
        OilEventFlag,
    )

    try:
        flags = tuple(
            OilEventFlag(item.strip().upper()) for item in event_flags.split(",") if item.strip()
        )
        request = OilDecisionRequestV2(
            intended_entry_at=datetime.fromisoformat(intended_entry_at),
            quote=OilBrokerQuoteV2(
                observed_at=datetime.fromisoformat(observed_at),
                bid=bid,
                ask=ask,
                offered_leverage=offered_leverage,
            ),
            modeled_leverage=modeled_leverage,
            event_flags=flags,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    snapshot = build_oil_decision_from_catalog(cfg, request)
    path = persist_oil_snapshot(Path(cfg.paths.artifacts_dir), snapshot)
    typer.echo(snapshot.model_dump_json(indent=2))
    typer.echo(f"immutable snapshot: {path}")


@api_app.command("serve")
def api_serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Serve the read-only FastAPI app (requires the [api] extra)."""
    cfg = _setup(config)
    from edgestack.pipelines import run_api_serve

    _run(run_api_serve, cfg, host=host, port=port)


@dashboard_app.command("serve")
def dashboard_serve(config: Path | None = _CONFIG_OPT) -> None:
    """Serve the Streamlit dashboard (requires the [dashboard] extra)."""
    _setup(config)
    import importlib.util
    import subprocess

    if importlib.util.find_spec("streamlit") is None:
        typer.secho(
            "streamlit is not installed; install with: pip install edgestack[dashboard]",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(2)
    from edgestack import dashboard as dashboard_pkg

    script = Path(dashboard_pkg.__file__).parent / "app.py"
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(script)]))


if __name__ == "__main__":
    app()
