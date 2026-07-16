"""EdgeStack command-line interface.

Thin wrapper: every command loads + validates config, then delegates to a
pipeline function. Exit codes: 0 success, 1 runtime failure, 2 not implemented
or bad usage.
"""

from __future__ import annotations

import json
import logging as _stdlib_logging
import sys
from datetime import date
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
]:
    app.add_typer(sub, name=name)

log = get_logger("cli")

_CONFIG_OPT = typer.Option(None, "--config", "-c", help="Path to YAML config file.")


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
