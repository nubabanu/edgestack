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
    from edgestack.data.catalog import DataCatalog
    from edgestack.data.providers.registry import get_price_provider
    from edgestack.data.universe import static_universe

    provider_name = provider or cfg.universe.source
    price_provider = get_price_provider(provider_name, cfg)
    universe = static_universe(cfg, end)
    wanted = symbols or universe.symbols
    print(f"downloading {len(wanted)} symbols {start}..{end} from {provider_name}")
    for limitation in price_provider.metadata.limitations:
        print(f"  provider limitation: {limitation}")
    for limitation in universe.limitations:
        print(f"  universe limitation: {limitation}")

    bars = price_provider.fetch_daily_bars(tuple(wanted), start, end)
    catalog = DataCatalog(cfg)
    written = catalog.write_bars(bars, provider=provider_name)
    catalog.audit("data_download", reason=provider_name,
                  symbols=len(written), rows=len(bars))
    got = set(written)
    for symbol in wanted:
        status = f"{written[symbol]} rows" if symbol in got else "NO DATA"
        print(f"  {symbol}: {status}")
    print(f"catalog now holds {len(catalog.list_symbols())} symbols")


def run_data_validate(cfg: EdgeStackConfig) -> None:
    from edgestack.data.calendar import TradingCalendar
    from edgestack.data.catalog import DataCatalog
    from edgestack.data.quality import assess_panel

    catalog = DataCatalog(cfg)
    # Quality assessment may inspect the full history including the guarded
    # test period: it validates data plumbing, not trading hypotheses.
    with catalog.guard.unlock(reason="data quality validation") as key:
        panel = catalog.load_panel(unlock_key=key)
    report = assess_panel(panel, TradingCalendar(cfg.data.calendar))
    print(report.summary())
    if report.quarantined:
        print("some symbols failed quality gates; fix or exclude them before research")


def run_features_build(cfg: EdgeStackConfig) -> None:
    import io

    from edgestack.data.catalog import DataCatalog, atomic_write_bytes
    from edgestack.features.registry import all_specs, build_features, featureset_id

    catalog = DataCatalog(cfg)
    panel = catalog.load_panel()  # guard-truncated: features never see the test period
    specs = all_specs()
    print(f"building {len(specs)} features over {panel['symbol'].nunique()} symbols, "
          f"{len(panel)} bars (featureset {featureset_id(specs)})")
    feats = build_features(panel, cfg, specs)

    out_path = catalog.data_dir / "features" / "features.parquet"
    buf = io.BytesIO()
    feats.to_parquet(buf, index=False)
    atomic_write_bytes(out_path, buf.getvalue())
    catalog.audit("features_build", reason=featureset_id(specs),
                  rows=len(feats), columns=len(feats.columns) - 2)
    non_null = feats.drop(columns=["symbol", "date"]).notna().mean().mean()
    print(f"wrote {len(feats)} rows x {len(feats.columns) - 2} features to {out_path}")
    print(f"average feature coverage: {non_null:.1%}")


def load_feature_frame(cfg: EdgeStackConfig) -> object:
    """Load the feature dataset written by ``run_features_build``."""
    from pathlib import Path

    import pandas as pd

    from edgestack.exceptions import DataError

    path = Path(cfg.paths.data_dir) / "features" / "features.parquet"
    if not path.exists():
        raise DataError("no feature dataset found; run `edgestack features build` first")
    return pd.read_parquet(path)


def _load_research_frames(cfg: EdgeStackConfig):
    """Feature + label frames for research (guard-truncated panel)."""
    import pandas as pd

    from edgestack.data.catalog import DataCatalog
    from edgestack.labels.forward_returns import forward_return_labels

    catalog = DataCatalog(cfg)
    features = load_feature_frame(cfg)
    assert isinstance(features, pd.DataFrame)
    panel = catalog.load_panel()
    horizons = tuple(sorted(set(cfg.signals.horizons) | set(cfg.discovery.horizons)))
    labels = forward_return_labels(
        panel, horizons,
        benchmark_symbol=cfg.universe.benchmark_symbol,
        execution_delay=cfg.signals.execution_delay_sessions,
    )
    return catalog, features, labels


def run_edges_discover(cfg: EdgeStackConfig) -> None:
    from edgestack.discovery.candidate_generation import generate_candidates
    from edgestack.discovery.edge_store import save_batch

    catalog, features, labels = _load_research_frames(cfg)
    experiment_id = catalog.record_experiment(
        "discovery",
        start_date=features["date"].min().date(),
        end_date=features["date"].max().date(),
    )
    batch = generate_candidates(features, labels, cfg, experiment_id)
    save_batch(catalog, batch)
    promising = sum(1 for c in batch.candidates if c.in_sample_p_value < 0.05)
    print(f"discovery batch {batch.batch_id}: {batch.trial_count} rules evaluated "
          f"({promising} look promising in-sample — expect most to die out of sample)")
    print(f"validate with: edgestack edges validate --batch {batch.batch_id}")


def run_edges_validate(cfg: EdgeStackConfig, *, batch_id: str | None = None) -> None:
    from edgestack.discovery.edge_store import load_batch, save_edges
    from edgestack.types import EdgeStatus
    from edgestack.validation.walk_forward import validate_batch

    catalog, features, labels = _load_research_frames(cfg)
    batch = load_batch(catalog, batch_id)
    print(f"validating batch {batch.batch_id}: {batch.trial_count} candidates "
          f"(trial count for FDR/DSR = {batch.trial_count})")
    edges = validate_batch(batch, features, labels, cfg)
    save_edges(catalog, edges)

    validated = [e for e in edges if e.lifecycle.status is EdgeStatus.VALIDATED]
    print(f"result: {len(validated)} VALIDATED, {len(edges) - len(validated)} REJECTED")
    for edge in sorted(validated, key=lambda e: -e.stats.net_mean_return)[:15]:
        s = edge.stats
        print(f"  {edge.identity.name}: net {s.net_mean_return:+.4f}/trade, "
              f"n={s.sample_size} (eff {s.effective_sample_size:.0f}), "
              f"q={s.q_value:.3f}, DSR={s.deflated_sharpe_ratio:.2f}")
    if not validated:
        print("no edges survived — that is a valid (and common) research outcome")


def run_models_train(cfg: EdgeStackConfig) -> None:
    from pathlib import Path

    from edgestack.models.registry import save_models
    from edgestack.models.training import train_models

    catalog, features, labels = _load_research_frames(cfg)
    catalog.record_experiment(
        "model_training",
        start_date=features["date"].min().date(),
        end_date=features["date"].max().date(),
        feature_set=str(len(features.columns) - 2),
    )
    models = train_models(features, labels, cfg)
    models_dir = Path(cfg.paths.artifacts_dir) / "models"
    save_models(models_dir, models)
    catalog.audit("models_train", reason=f"{len(models)} models")
    print(f"trained {len(models)} models -> {models_dir}")
    print(f"{'horizon':>7} {'side':>5} {'model':>18} {'OOF logloss':>12} "
          f"{'Brier':>7} {'ECE':>6} {'n':>6}")
    for m in models:
        print(f"{m.horizon:>7} {m.side:>5} {m.name:>18} "
              f"{m.metrics['oof_log_loss']:>12.4f} {m.metrics['oof_brier']:>7.4f} "
              f"{m.metrics['oof_ece']:>6.3f} {int(m.metrics['oof_n']):>6}")


def run_signals_generate(cfg: EdgeStackConfig, *, as_of: date | None = None) -> None:
    from pathlib import Path

    import pandas as pd

    from edgestack.data.catalog import DataCatalog
    from edgestack.discovery.edge_store import load_edges
    from edgestack.exceptions import DataError
    from edgestack.models.registry import load_models
    from edgestack.reporting.signal_report import render_tables, save_report
    from edgestack.scoring.signal_engine import generate_signal_report

    catalog = DataCatalog(cfg)
    features = load_feature_frame(cfg)
    assert isinstance(features, pd.DataFrame)
    panel = catalog.load_panel()
    edges = load_edges(catalog)
    if not edges:
        raise DataError(
            "no VALIDATED/ACTIVE edges in the store; run `edgestack edges "
            "discover` and `edgestack edges validate` first"
        )
    try:
        models = load_models(Path(cfg.paths.artifacts_dir) / "models",
                             expected_config_hash=cfg.config_hash())
    except DataError:
        models = []
        print("note: no trained models found — probabilities fall back to edge "
              "posteriors and are flagged as uncalibrated")

    report = generate_signal_report(cfg, features, panel, edges, models, as_of)
    path = save_report(catalog, report)
    catalog.audit("signals_generate", reason=str(report.as_of_date),
                  longs=len(report.long_candidates), shorts=len(report.short_candidates),
                  abstentions=len(report.abstentions))
    print(render_tables(report))
    print(f"\nfull machine-readable report: {path}")


def run_signals_rank(cfg: EdgeStackConfig, *, top: int = 20, as_of: date | None = None) -> None:
    from edgestack.data.catalog import DataCatalog
    from edgestack.reporting.signal_report import load_report, render_tables

    report = load_report(DataCatalog(cfg), as_of)
    print(render_tables(report, top=top))
    # Show the strongest explanation as a sample of the reasoning trail.
    best = (report.long_candidates or report.short_candidates)
    if best:
        print("\n--- top candidate explanation ---")
        print(best[0].explanation)


def run_backtest(cfg: EdgeStackConfig, *, scenario: str | None = None) -> None:
    import numpy as np
    import pandas as pd

    from edgestack.backtest.engine import BacktestEngine
    from edgestack.backtest.intents import build_trade_intents
    from edgestack.backtest.reports import compute_metrics, save_backtest_report
    from edgestack.data.catalog import DataCatalog
    from edgestack.discovery.edge_store import load_edges
    from edgestack.exceptions import DataError
    from edgestack.types import CostScenario

    catalog = DataCatalog(cfg)
    features = load_feature_frame(cfg)
    assert isinstance(features, pd.DataFrame)
    panel = catalog.load_panel()
    edges = load_edges(catalog)
    if not edges:
        raise DataError("no VALIDATED/ACTIVE edges to backtest; run discovery/validation")

    chosen = CostScenario(scenario.upper()) if scenario else cfg.costs.scenario
    intents = build_trade_intents(features, panel, edges, cfg)
    print(f"replaying {len(intents)} out-of-sample intents from {len(edges)} edges "
          f"under {chosen.value} costs")
    if not intents:
        print("no out-of-sample signals — nothing to backtest")
        return

    engine = BacktestEngine(panel, cfg, chosen)
    ledger = engine.run(intents, initial_cash=cfg.paper.initial_cash)
    rng = np.random.default_rng(cfg.project.random_seed)
    metrics = compute_metrics(ledger, rng=rng)
    run_id, out_dir = save_backtest_report(catalog, ledger, metrics, chosen)
    catalog.audit("backtest_run", reason=run_id, scenario=chosen.value,
                  trades=metrics.get("n_trades", 0))

    print(f"backtest run {run_id} -> {out_dir}")
    for key in ("cumulative_return", "cagr", "sharpe", "sortino", "max_drawdown",
                "hit_rate", "profit_factor", "expectancy", "n_trades"):
        if key in metrics and metrics[key] is not None:
            print(f"  {key}: {metrics[key]:.4f}" if isinstance(metrics[key], float)
                  else f"  {key}: {metrics[key]}")
    if "daily_sharpe_ci" in metrics:
        lo, hi = metrics["daily_sharpe_ci"]
        print(f"  daily sharpe 95% CI: [{lo:.3f}, {hi:.3f}] (block bootstrap)")
    print("  NOTE: research backtest; survivorship-biased universe; not advice")


def run_monitor(cfg: EdgeStackConfig) -> None:
    from edgestack.monitoring.lifecycle import run_monitoring

    catalog, features, labels = _load_research_frames(cfg)
    transitions = run_monitoring(catalog, features, labels, cfg)
    catalog.audit("monitor_run", reason=f"{len(transitions)} transitions")
    if not transitions:
        print("monitoring: no lifecycle transitions (edges healthy or too few "
              "recent signals to judge)")
        return
    print(f"monitoring: {len(transitions)} lifecycle transitions")
    for edge_id, old, new, rule in transitions:
        print(f"  {edge_id}: {old.value} -> {new.value} ({rule})")


def run_report_backtest(cfg: EdgeStackConfig, *, run_id: str | None = None) -> None:
    import json

    from edgestack.data.catalog import DataCatalog
    from edgestack.exceptions import DataError

    catalog = DataCatalog(cfg)
    with catalog.connect() as con:
        if run_id is None:
            row = con.execute(
                "SELECT run_id, payload FROM backtests ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        else:
            row = con.execute(
                "SELECT run_id, payload FROM backtests WHERE run_id = ?", [run_id]
            ).fetchone()
    if row is None:
        raise DataError("no backtest runs stored; run `edgestack backtest run` first")
    run_id, payload = row[0], json.loads(row[1])
    out_dir = catalog.data_dir / "reports" / "backtests" / run_id
    print(f"backtest {run_id} (scenario {payload['cost_scenario']})")
    print(f"  json: {out_dir / 'report.json'}")
    print(f"  html: {out_dir / 'report.html'}")
    for key, value in payload["metrics"].items():
        print(f"  {key}: {value}")
    print(f"  {payload['disclaimer']}")


def run_paper_session(cfg: EdgeStackConfig, *, as_of: date | None = None) -> None:
    raise NotImplementedError("paper trading (milestone 12)")


def run_api_serve(cfg: EdgeStackConfig, *, host: str, port: int) -> None:
    raise NotImplementedError("API server (milestone 12)")
