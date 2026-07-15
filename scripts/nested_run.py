"""Phase 3-4: fully nested discovery -> validation -> trading on the broad
point-in-time universe, with the benchmark/alpha framework.

For each outer year Y in {2021, 2022, 2023}:
  1. using ONLY data before Y: enumerate ~756 candidate rules, validate them
     with inner purged walk-forward + FDR + deflated Sharpe;
  2. choose the family-vote threshold F on the inner validation windows
     (recording neighbor performance for stability reporting);
  3. freeze edges + F + quantile thresholds (fit on all pre-Y data);
  4. trade year Y out-of-sample under conservative costs, restricted to
     point-in-time index members;
  5. compare against SPY total return, exposure-matched SPY, equal-weight
     PIT universe, and random-entry benchmarks with matched trade counts.

Only outer-year performance is reported as evidence.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from edgestack.backtest.engine import BacktestEngine, TradeIntent
from edgestack.config import load_config
from edgestack.data.catalog import DataCatalog, atomic_write_bytes
from edgestack.data.universe_pit import PitSP500Universe
from edgestack.discovery.candidate_generation import (
    BINARY_RULE_FEATURES,
    CONTINUOUS_RULE_FEATURES,
    generate_candidates,
)
from edgestack.discovery.conditions import evaluate_condition
from edgestack.features.binning import QuantileBinner
from edgestack.features.registry import build_features, get_spec
from edgestack.labels.forward_returns import forward_return_labels
from edgestack.logging import configure
from edgestack.reporting.benchmarks import (
    alpha_regression,
    concentration_diagnostics,
    daily_total_returns,
    equal_weight_pit_returns,
    summarize_returns,
)
from edgestack.types import EdgeStatus, Side
from edgestack.validation.metrics import sharpe_ratio
from edgestack.validation.splits import PurgedWalkForwardSplitter
from edgestack.validation.walk_forward import validate_batch

OUTER_YEARS = (2021, 2022, 2023)
F_GRID = (2, 3, 4, 5)
HORIZON = 10
RANDOM_REPS = 10

FEATURE_SET = tuple(dict.fromkeys([
    *CONTINUOUS_RULE_FEATURES, *BINARY_RULE_FEATURES,
    "bench_trend_200", "bench_vol_20", "natr_14", "realized_vol_20", "mom_60",
]))


def eligible(merged: pd.DataFrame, member: pd.Series) -> pd.Series:
    return member & (merged["close"] >= 5.0)


def make_intents(rows: pd.DataFrame, cfg) -> list[TradeIntent]:
    out = []
    for row in rows.itertuples(index=False):
        close = float(row.close)
        natr = getattr(row, "natr_14", np.nan)
        atr = natr * close if np.isfinite(natr) and natr > 0 else 0.02 * close
        out.append(TradeIntent(
            symbol=str(row.symbol), signal_session=pd.Timestamp(row.date),
            side=Side.LONG, horizon=HORIZON,
            stop_price=max(close - 2 * atr, 0.01), target_price=close + 2.5 * atr))
    return out


def run_window(panel: pd.DataFrame, intents: list[TradeIntent], cfg,
               start: pd.Timestamp, end: pd.Timestamp, sessions: pd.DatetimeIndex):
    """Engine over [start, end]; returns (daily returns, trades frame, avg gross)."""
    lo = sessions[max(0, sessions.searchsorted(start) - 30)]
    hi = sessions[min(len(sessions) - 1, sessions.searchsorted(end) + 15)]
    p = panel.loc[(pd.to_datetime(panel["date"]) >= lo)
                  & (pd.to_datetime(panel["date"]) <= hi)]
    engine = BacktestEngine(p, cfg, None)
    ledger = engine.run(intents, initial_cash=1_000_000)
    eq = ledger.equity_frame()
    eq = eq.loc[(eq["session"] >= start) & (eq["session"] <= end)]
    rets = pd.Series(
        np.diff(eq["equity"].to_numpy()) / eq["equity"].to_numpy()[:-1],
        index=pd.to_datetime(eq["session"].iloc[1:]),
    )
    gross = float((eq["gross_exposure"] / eq["equity"]).mean()) if len(eq) else 0.0
    return rets, ledger.trades_frame(), gross


def main() -> int:
    configure()
    t0 = time.time()
    cfg = load_config("configs/broad.yaml")
    catalog = DataCatalog(cfg)
    universe = PitSP500Universe(Path("data/cache/universe"), earliest=date(2011, 1, 1))

    print("loading panel ...")
    panel = catalog.load_panel()  # guard-truncated before 2024
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).unique()))
    print(f"panel: {panel['symbol'].nunique()} symbols, {len(panel):,} bars")

    print("building features (restricted set) ...")
    specs = tuple(get_spec(n) for n in FEATURE_SET)
    features = build_features(panel, cfg, specs)
    for c in features.columns:
        if features[c].dtype == np.float64:
            features[c] = features[c].astype(np.float32)
    labels = forward_return_labels(panel, (HORIZON,), benchmark_symbol="SPY")

    merged = features.merge(
        labels[["symbol", "date", "label_end", "gross_ret"]],
        on=["symbol", "date"], how="inner",
    ).merge(panel[["symbol", "date", "close"]], on=["symbol", "date"]).reset_index(drop=True)
    print(f"research frame: {len(merged):,} rows ({time.time()-t0:.0f}s)")

    print("applying point-in-time membership mask ...")
    member = universe.membership_mask(merged["symbol"], merged["date"])
    keep = eligible(merged, member)
    merged = merged.loc[keep].reset_index(drop=True)
    print(f"eligible point-in-time rows: {len(merged):,}")

    spy = daily_total_returns(panel, "SPY")
    pm = universe.membership_mask(panel["symbol"], panel["date"])
    ew = equal_weight_pit_returns(panel, pm & (panel["close"] >= 5.0))

    binnable = tuple(c for c in (*CONTINUOUS_RULE_FEATURES, "bench_trend_200",
                                 "bench_vol_20") if c in merged.columns)
    results = []
    outer_rets: list[pd.Series] = []
    all_trades: list[pd.DataFrame] = []

    for year in OUTER_YEARS:
        cutoff = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31")
        pre = merged.loc[merged["date"] < cutoff].reset_index(drop=True)
        outer = merged.loc[(merged["date"] >= cutoff)
                           & (merged["date"] <= year_end)].reset_index(drop=True)
        print(f"\n=== OUTER {year}: research on {len(pre):,} pre rows, trade "
              f"{len(outer):,} outer rows ===")

        experiment_id = catalog.record_experiment(
            "nested_outer", start_date=pre["date"].min().date(),
            end_date=date(year, 12, 31), notes=f"outer={year}")

        # discovery/validation expect a pure feature frame; they merge labels
        # themselves, so the label columns must not ride along.
        features_pre = pre.drop(columns=["label_end", "gross_ret", "close"])
        batch = generate_candidates(features_pre, labels, cfg, experiment_id)
        print(f"  candidates: {batch.trial_count}")
        edges = validate_batch(batch, features_pre, labels, cfg)
        validated = [e for e in edges
                     if e.lifecycle.status is EdgeStatus.VALIDATED
                     and e.identity.direction is Side.LONG]
        print(f"  validated long edges: {len(validated)} "
              f"({time.time()-t0:.0f}s elapsed)")
        if not validated:
            results.append({"year": year, "validated": 0, "note": "abstained all year"})
            outer_rets.append(pd.Series(dtype=float))
            continue

        families = sorted({e.identity.family for e in validated})

        def fam_votes(frame: pd.DataFrame, binner, *, validated=validated,
                      families=families) -> np.ndarray:
            votes = np.zeros((len(frame), len(families)), dtype=bool)
            for e in validated:
                mask = evaluate_condition(e.identity.condition, frame, binner).to_numpy()
                votes[:, families.index(e.identity.family)] |= mask
            return votes.sum(axis=1)

        # --- inner threshold selection on the pre-period validation windows
        splitter = PurgedWalkForwardSplitter.from_config(cfg)
        folds = splitter.split_frame(pre["date"], pre["label_end"])
        inner_scores: dict[int, list[float]] = {f: [] for f in F_GRID}
        for fold in folds:
            train = pre.iloc[fold.train_idx]
            test = pre.iloc[fold.test_idx]
            binner = QuantileBinner(quantiles=cfg.discovery.quantile_bins).fit(
                train, binnable)
            votes = fam_votes(test, binner)
            for f in F_GRID:
                intents = make_intents(test.loc[votes >= f], cfg)
                if len(intents) < 20:
                    inner_scores[f].append(-9.0)
                    continue
                rets, _, _ = run_window(panel, intents, cfg,
                                        fold.test_start, fold.test_end, sessions)
                inner_scores[f].append(sharpe_ratio(rets.to_numpy())
                                       if len(rets) > 30 else -9.0)
        mean_scores = {f: float(np.mean(v)) for f, v in inner_scores.items()}
        best_f = max(mean_scores, key=mean_scores.get)

        # --- freeze and trade the outer year
        pre_binner = QuantileBinner(quantiles=cfg.discovery.quantile_bins).fit(
            pre, binnable)
        votes = fam_votes(outer, pre_binner)
        chosen_rows = outer.loc[votes >= best_f]
        intents = make_intents(chosen_rows, cfg)
        rets, trades, gross = run_window(panel, intents, cfg, cutoff, year_end, sessions)
        outer_rets.append(rets)
        if not trades.empty:
            all_trades.append(trades)

        # --- benchmarks for the same year
        spy_y = spy.loc[(spy.index >= cutoff) & (spy.index <= year_end)]
        ew_y = ew.loc[(ew.index >= cutoff) & (ew.index <= year_end)]
        matched = spy_y * gross
        rand_sharpes = []
        pool = outer.reset_index(drop=True)
        for rep in range(RANDOM_REPS):
            rrng = np.random.default_rng(1000 + rep)
            idx = rrng.choice(len(pool), size=min(len(intents), len(pool)),
                              replace=False)
            r_intents = make_intents(pool.iloc[idx], cfg)
            r_rets, _, _ = run_window(panel, r_intents, cfg, cutoff, year_end, sessions)
            rand_sharpes.append(sharpe_ratio(r_rets.to_numpy())
                                if len(r_rets) > 30 else 0.0)

        row = {
            "year": year,
            "candidates": batch.trial_count,
            "validated_long_edges": len(validated),
            "families": [f.value for f in families],
            "inner_scores": mean_scores,
            "chosen_F": best_f,
            "signals": len(intents),
            "trades": len(trades),
            "strategy_sharpe": sharpe_ratio(rets.to_numpy()) if len(rets) > 30 else None,
            "strategy_return": float(np.prod(1 + rets) - 1) if len(rets) else 0.0,
            "avg_gross_exposure": gross,
            "spy_return": float(np.prod(1 + spy_y) - 1),
            "spy_sharpe": sharpe_ratio(spy_y.to_numpy()),
            "exposure_matched_spy_return": float(np.prod(1 + matched) - 1),
            "equal_weight_pit_return": float(np.prod(1 + ew_y) - 1),
            "random_entry_sharpe_mean": float(np.mean(rand_sharpes)),
            "random_entry_sharpe_p90": float(np.quantile(rand_sharpes, 0.9)),
        }
        results.append(row)
        print(f"  F chosen={best_f} (inner scores {mean_scores})")
        print(f"  outer: {len(trades)} trades, return "
              f"{row['strategy_return']:+.1%} (SPY {row['spy_return']:+.1%}, "
              f"exposure-matched {row['exposure_matched_spy_return']:+.1%}, "
              f"EW-PIT {row['equal_weight_pit_return']:+.1%}), "
              f"Sharpe {row['strategy_sharpe']}")

    # --- pooled outer evidence ------------------------------------------------
    pooled = pd.concat([r for r in outer_rets if len(r)]) if outer_rets else pd.Series(dtype=float)
    report: dict = {"windows": results}
    if len(pooled) > 60:
        rng2 = np.random.default_rng(7)
        report["pooled_strategy"] = summarize_returns(pooled, rng=rng2, label="strategy")
        spy_pool = spy.loc[pooled.index.min():pooled.index.max()]
        report["pooled_spy"] = summarize_returns(spy_pool, rng=rng2, label="SPY")
        alpha = alpha_regression(pooled, spy, rng=rng2)
        report["alpha_vs_spy"] = {
            "beta": alpha.beta, "annualized_alpha": alpha.annualized_alpha,
            "alpha_ci": list(alpha.alpha_ci), "correlation": alpha.correlation,
        }
    if all_trades:
        trades_all = pd.concat(all_trades, ignore_index=True)
        report["concentration"] = concentration_diagnostics(trades_all)

    atomic_write_bytes(Path(cfg.paths.artifacts_dir) / "nested_run.json",
                       json.dumps(report, indent=2, default=str).encode())
    catalog.audit("nested_run", reason="phases 3-4", windows=len(results))
    print(f"\nfull report -> artifacts/nested_run.json ({time.time()-t0:.0f}s total)")
    print(json.dumps({k: v for k, v in report.items() if k != "windows"},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
