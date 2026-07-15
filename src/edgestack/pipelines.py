"""Pipeline entry points wired to the CLI.

Each function is replaced with a real implementation as its milestone lands;
unimplemented stages raise ``NotImplementedError`` so the CLI exits cleanly
with code 2 instead of pretending to work.
"""

from __future__ import annotations

from datetime import date

from edgestack.config import EdgeStackConfig


def run_data_download(cfg: EdgeStackConfig, start: date, end: date, *,
                      provider: str | None = None,
                      symbols: tuple[str, ...] | None = None) -> None:
    raise NotImplementedError("data download (milestone 3)")


def run_data_validate(cfg: EdgeStackConfig) -> None:
    raise NotImplementedError("data validation (milestone 3)")


def run_features_build(cfg: EdgeStackConfig) -> None:
    raise NotImplementedError("feature build (milestone 4)")


def run_edges_discover(cfg: EdgeStackConfig) -> None:
    raise NotImplementedError("edge discovery (milestone 7)")


def run_edges_validate(cfg: EdgeStackConfig, *, batch_id: str | None = None) -> None:
    raise NotImplementedError("edge validation (milestone 7)")


def run_models_train(cfg: EdgeStackConfig) -> None:
    raise NotImplementedError("model training (milestone 8)")


def run_signals_generate(cfg: EdgeStackConfig, *, as_of: date | None = None) -> None:
    raise NotImplementedError("signal generation (milestone 9)")


def run_signals_rank(cfg: EdgeStackConfig, *, top: int = 20, as_of: date | None = None) -> None:
    raise NotImplementedError("signal ranking (milestone 9)")


def run_backtest(cfg: EdgeStackConfig, *, scenario: str | None = None) -> None:
    raise NotImplementedError("backtest (milestone 10)")


def run_monitor(cfg: EdgeStackConfig) -> None:
    raise NotImplementedError("monitoring (milestone 11)")


def run_report_backtest(cfg: EdgeStackConfig, *, run_id: str | None = None) -> None:
    raise NotImplementedError("backtest report (milestone 10)")


def run_paper_session(cfg: EdgeStackConfig, *, as_of: date | None = None) -> None:
    raise NotImplementedError("paper trading (milestone 12)")


def run_api_serve(cfg: EdgeStackConfig, *, host: str, port: int) -> None:
    raise NotImplementedError("API server (milestone 12)")
