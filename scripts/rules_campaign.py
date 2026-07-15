"""Rules campaign: EBM -> RuleFit -> stability selection -> distillation,
fully nested per outer year, judged by the paper-grade test battery.

Stages (each fits a background-run budget; artifacts are resumable):

    --stage prep            build rules_merged.parquet (features + excess-net
                            + MAE targets, point-in-time eligible rows)
    --stage year --year Y   full pipeline on data < Y, trade year Y frozen
    --stage report          pooled evidence, SPA / StepM / Ledoit-Wolf /
                            Newey-West, acceptance checklist, verdict label
    --stage canary          synthetic acceptance test: injected effect must be
                            rediscovered stably; noise must yield ~nothing

Research software. Verdicts come only from frozen outer-window evidence.
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

from edgestack.config import load_config
from edgestack.data.catalog import DataCatalog, atomic_write_bytes
from edgestack.data.universe_pit import PitSP500Universe
from edgestack.discovery.candidate_generation import (
    BINARY_RULE_FEATURES,
    CONTINUOUS_RULE_FEATURES,
)
from edgestack.discovery.distill import distill_policy
from edgestack.discovery.ebm_shapes import ebm_feature_report
from edgestack.discovery.rulefit_engine import _fast_mask
from edgestack.discovery.stability import StableRule, stability_select
from edgestack.execution.costs import CostModel
from edgestack.features.registry import build_features, get_spec
from edgestack.labels.adverse_excursion import adverse_excursion_labels
from edgestack.labels.forward_returns import forward_return_labels
from edgestack.logging import configure
from edgestack.reporting.benchmarks import (
    concentration_diagnostics,
    daily_total_returns,
    equal_weight_pit_returns,
    summarize_returns,
)
from edgestack.types import Predicate, Side
from edgestack.validation.advanced_tests import (
    newey_west_alpha,
    sharpe_difference_test,
    spa_test,
)
from edgestack.validation.metrics import max_drawdown, sharpe_ratio

HORIZON = 10
OUTER_YEARS = (2021, 2022, 2023)
MERGED_PATH = Path("data/features/rules_merged.parquet")
WINDOWS_DIR = Path("artifacts/rules_windows")

FEATURE_SET = tuple(dict.fromkeys([
    *CONTINUOUS_RULE_FEATURES, *BINARY_RULE_FEATURES,
    "bench_trend_200", "bench_vol_20", "natr_14", "realized_vol_20", "mom_60",
]))


def rule_feature_columns(available: set[str]) -> tuple[str, ...]:
    """Only registered, causal features may enter discovery."""
    cols = []
    for name in FEATURE_SET:
        if name in available:
            get_spec(name)  # raises for anything not in the registry
            cols.append(name)
    return tuple(cols)


def stage_prep(cfg, catalog) -> None:
    t0 = time.time()
    universe = PitSP500Universe(Path("data/cache/universe"), earliest=date(2011, 1, 1))
    panel = catalog.load_panel()
    print(f"panel: {panel['symbol'].nunique()} symbols, {len(panel):,} bars")
    specs = tuple(get_spec(n) for n in FEATURE_SET)
    features = build_features(panel, cfg, specs)
    for c in features.columns:
        if features[c].dtype == np.float64:
            features[c] = features[c].astype(np.float32)

    roundtrip = CostModel.from_config(cfg).roundtrip_cost(Side.LONG, HORIZON)
    labels = forward_return_labels(panel, (HORIZON,), benchmark_symbol="SPY",
                                   roundtrip_cost=roundtrip)
    mae = adverse_excursion_labels(panel, HORIZON)
    merged = (features
              .merge(labels[["symbol", "date", "label_end", "gross_ret",
                             "excess_net_ret"]], on=["symbol", "date"], how="inner")
              .merge(mae[["symbol", "date", "mae"]], on=["symbol", "date"], how="left")
              .merge(panel[["symbol", "date", "close"]], on=["symbol", "date"]))
    member = universe.membership_mask(merged["symbol"], merged["date"])
    merged = merged.loc[(member & (merged["close"] >= 5.0)).to_numpy()]
    merged = merged.reset_index(drop=True)
    MERGED_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(MERGED_PATH, index=False)
    print(f"rules frame: {len(merged):,} eligible rows -> {MERGED_PATH} "
          f"({time.time()-t0:.0f}s; roundtrip cost {roundtrip:.4f})")


def _pipeline_on(pre: pd.DataFrame, cfg, seed: int, label: str) -> dict:
    """EBM -> stability -> distill on one training frame. Returns freeze dict."""
    t0 = time.time()
    feature_cols = rule_feature_columns(set(pre.columns))
    ebm = ebm_feature_report(pre, pre["excess_net_ret"], feature_cols, seed=seed)
    shortlist = tuple(ebm["shortlist"]) or feature_cols
    print(f"  [{label}] EBM shortlist: {len(shortlist)}/{len(feature_cols)} features "
          f"({time.time()-t0:.0f}s)")

    rules, trials = stability_select(
        pre, shortlist, "excess_net_ret", seed=seed, mae_col="mae",
    )
    # Tradeable rules: stable, long-directional, AND absolutely positive net
    # of costs (a rule can be a stable relative edge yet still lose money).
    passing = [r for r in rules
               if r.passes and r.direction > 0 and r.inner_abs_mean > 0]
    print(f"  [{label}] structural rules: {len(rules)}, PASSING long: "
          f"{len(passing)}, trials counted: {trials} ({time.time()-t0:.0f}s)")

    freeze: dict = {
        "ebm": {k: v for k, v in ebm.items() if k != "feature_importance"},
        "trials": trials,
        "rules": [_rule_row(r) for r in rules],
        "n_passing": len(passing),
    }
    if not passing:
        return freeze

    indicators = _indicator_frame(pre, passing)
    policy = distill_policy(indicators, pre["excess_net_ret"].to_numpy(), seed=seed)
    freeze["policy"] = {
        "distiller": policy.distiller,
        "text": policy.text[:4000],
        "fallback_notes": list(policy.fallback_notes),
    }
    freeze["_passing"] = passing      # objects, stripped before JSON
    freeze["_policy"] = policy
    return freeze


def _rule_row(r: StableRule) -> dict:
    return {
        "signature": r.signature, "direction": r.direction,
        "freq": round(r.selection_freq, 3), "sign_cons": round(r.sign_consistency, 3),
        "inner_n": r.inner_n, "inner_mean_excess": round(r.inner_mean_excess, 5),
        "inner_abs_mean": round(r.inner_abs_mean, 5),
        "inner_t": round(r.inner_t, 2),
        "inner_mean_mae": None if r.inner_mean_mae is None else round(r.inner_mean_mae, 4),
        "passes": r.passes, "fail_reasons": list(r.fail_reasons),
        "condition": r.condition.describe(),
    }


def _indicator_frame(frame: pd.DataFrame, rules: list[StableRule]) -> pd.DataFrame:
    feats = frame[[c for c in frame.columns
                   if c not in ("symbol", "date", "label_end", "gross_ret",
                                "excess_net_ret", "mae", "close")]].astype(np.float64)
    feats = feats.fillna(feats.median(numeric_only=True))
    out = {}
    for r in rules:
        preds = ((r.condition,) if isinstance(r.condition, Predicate)
                 else r.condition.conditions)
        out[r.signature] = _fast_mask(feats, list(preds))
    return pd.DataFrame(out, index=frame.index)


def stage_year(cfg, catalog, year: int) -> None:
    t0 = time.time()
    panel = catalog.load_panel()
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).unique()))
    merged = pd.read_parquet(MERGED_PATH)
    cutoff, year_end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    pre = merged.loc[(merged["date"] < cutoff)
                     & merged["excess_net_ret"].notna()].reset_index(drop=True)
    outer = merged.loc[(merged["date"] >= cutoff)
                       & (merged["date"] <= year_end)].reset_index(drop=True)
    print(f"=== RULES OUTER {year}: {len(pre):,} pre rows, {len(outer):,} outer ===")
    experiment_id = catalog.record_experiment("rules_outer", end_date=date(year, 12, 31),
                                              notes=f"rules campaign outer={year}")

    freeze = _pipeline_on(pre, cfg, cfg.project.random_seed, str(year))
    catalog.update_trial_count(experiment_id, freeze["trials"])
    row: dict = {"year": year, "trials": freeze["trials"],
                 "n_structural_rules": len(freeze["rules"]),
                 "n_passing": freeze["n_passing"],
                 "rules": freeze["rules"],
                 "ebm": freeze["ebm"],
                 "policy": freeze.get("policy")}

    if freeze["n_passing"] == 0:
        row["note"] = "no stable rules passed: campaign abstains this year"
        _persist(year, row, pd.Series(dtype=float), pd.DataFrame())
        print(f"  {row['note']} ({time.time()-t0:.0f}s)")
        return

    passing, policy = freeze["_passing"], freeze["_policy"]
    outer_ind = _indicator_frame(outer, passing)
    tiers = policy.tier(outer_ind)
    traded = outer.loc[tiers >= 0.5]
    row["tier_counts"] = {str(t): int((tiers == t).sum()) for t in (0.0, 0.5, 1.0)}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nested_run import make_intents, run_window

    intents = make_intents(traded, cfg)
    rets, trades, gross = run_window(panel, intents, cfg, cutoff, year_end, sessions)

    spy = daily_total_returns(panel, "SPY")
    spy_y = spy.loc[(spy.index >= cutoff) & (spy.index <= year_end)]
    row.update({
        "signals": len(intents), "trades": len(trades),
        "avg_gross_exposure": gross,
        "strategy_return": float(np.prod(1 + rets) - 1) if len(rets) else 0.0,
        "strategy_sharpe": sharpe_ratio(rets.to_numpy()) if len(rets) > 30 else None,
        "strategy_maxdd": max_drawdown(rets.to_numpy()) if len(rets) > 30 else None,
        "spy_return": float(np.prod(1 + spy_y) - 1),
        "exposure_matched_spy_return": float(np.prod(1 + spy_y * gross) - 1),
    })
    _persist(year, row, rets, trades)
    print(f"  distiller={row['policy']['distiller']}, passing rules={len(passing)}, "
          f"signals={len(intents)}")
    print(f"  outer: return {row['strategy_return']:+.1%} "
          f"(SPY {row['spy_return']:+.1%}, matched "
          f"{row['exposure_matched_spy_return']:+.1%}), "
          f"Sharpe {row['strategy_sharpe']} ({time.time()-t0:.0f}s)")


def _persist(year: int, row: dict, rets: pd.Series, trades: pd.DataFrame) -> None:
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in row.items() if not k.startswith("_")}
    atomic_write_bytes(WINDOWS_DIR / f"{year}.json",
                       json.dumps(clean, indent=2, default=str).encode())
    pd.DataFrame({"date": rets.index, "ret": rets.to_numpy()}).to_parquet(
        WINDOWS_DIR / f"returns_{year}.parquet", index=False)
    trades.to_parquet(WINDOWS_DIR / f"trades_{year}.parquet", index=False)


def stage_report(cfg, catalog) -> None:
    panel = catalog.load_panel()
    spy = daily_total_returns(panel, "SPY")
    universe = PitSP500Universe(Path("data/cache/universe"), earliest=date(2011, 1, 1))
    pm = universe.membership_mask(panel["symbol"], panel["date"])
    ew = equal_weight_pit_returns(panel, pm & (panel["close"] >= 5.0))

    windows, rets_list, trades_list = [], [], []
    for year in OUTER_YEARS:
        path = WINDOWS_DIR / f"{year}.json"
        if not path.exists():
            print(f"missing outer window {year}")
            continue
        windows.append(json.loads(path.read_text()))
        r = pd.read_parquet(WINDOWS_DIR / f"returns_{year}.parquet")
        rets_list.append(pd.Series(r["ret"].to_numpy(),
                                   index=pd.to_datetime(r["date"])))
        t = pd.read_parquet(WINDOWS_DIR / f"trades_{year}.parquet")
        if not t.empty:
            trades_list.append(t)

    pooled = pd.concat([r for r in rets_list if len(r)]) if rets_list else pd.Series(dtype=float)
    report: dict = {"windows": windows}
    if len(pooled) > 120:
        rng = np.random.default_rng(7)
        spy_pool = spy.reindex(pooled.index).fillna(0.0)
        ew_pool = ew.reindex(pooled.index).fillna(0.0)
        report["pooled_strategy"] = summarize_returns(pooled, rng=rng, label="rules")
        report["pooled_spy"] = summarize_returns(spy_pool, rng=rng, label="SPY")
        report["newey_west_alpha"] = newey_west_alpha(pooled, spy_pool)
        report["ledoit_wolf_vs_equal_weight"] = sharpe_difference_test(pooled, ew_pool)
        models = pd.DataFrame({"rules_policy": pooled, "equal_weight_pit": ew_pool})
        report["spa_vs_spy"] = spa_test(spy_pool, models)
    if trades_list:
        report["concentration"] = concentration_diagnostics(
            pd.concat(trades_list, ignore_index=True))
    report["verdict"] = _verdict(report)
    atomic_write_bytes(Path(cfg.paths.artifacts_dir) / "rules_campaign.json",
                       json.dumps(report, indent=2, default=str).encode())
    catalog.audit("rules_campaign_report", reason=report["verdict"]["label"])
    print(json.dumps(report["verdict"], indent=2))
    print("full report -> artifacts/rules_campaign.json")


def _verdict(report: dict) -> dict:
    checks: dict[str, bool | None] = {}
    windows = report.get("windows", [])
    traded = [w for w in windows if w.get("trades")]
    checks["stable_rules_found_every_window"] = all(
        w.get("n_passing", 0) > 0 for w in windows) if windows else False
    pos = [w for w in traded if (w.get("strategy_return") or 0) > 0]
    checks["positive_most_outer_windows"] = (
        len(pos) > len(traded) / 2 if traded else False)
    nw = report.get("newey_west_alpha")
    checks["nw_alpha_positive_and_significant"] = (
        bool(nw and nw["alpha_ann"] > 0 and nw["alpha_t"] >= 2.0) if nw else None)
    spa = report.get("spa_vs_spy")
    checks["spa_beats_spy"] = bool(spa and spa["p_consistent"] < 0.05) if spa else None
    conc = report.get("concentration")
    checks["not_dependent_on_one_stock"] = (
        bool(conc and conc.get("best_symbol_share") is not None
             and conc["best_symbol_share"] < 0.5) if conc else None)
    beats = [w for w in traded
             if (w.get("strategy_return") or -9) > w.get("exposure_matched_spy_return", 9)]
    checks["beats_exposure_matched_most_windows"] = (
        len(beats) > len(traded) / 2 if traded else False)

    n_true = sum(1 for v in checks.values() if v is True)
    n_applicable = sum(1 for v in checks.values() if v is not None)
    if not windows or all(w.get("n_passing", 0) == 0 for w in windows):
        label = "Rejected (no stable rules discovered)"
    elif checks["nw_alpha_positive_and_significant"] and checks["spa_beats_spy"] \
            and checks["positive_most_outer_windows"]:
        label = "Promising structure, unproven edge"  # never 'Supported' from 3 windows
    elif n_true >= n_applicable / 2:
        label = "Interesting hypothesis"
    else:
        label = "Rejected"
    return {"label": label, "checks": checks}


# ---------------------------------------------------------------------------
# Canary: the acceptance test of the whole apparatus
# ---------------------------------------------------------------------------


def run_canary(*, with_effect: bool, seed: int = 42) -> dict:
    """Mini end-to-end on synthetic data. With an injected persistent effect
    the pipeline must find >=1 passing stable rule; on pure noise ~none."""
    from edgestack.config import EdgeStackConfig
    from edgestack.data.providers.synthetic import GBM, CalendarEffect, SyntheticMarket

    cfg = EdgeStackConfig.model_validate({
        "universe": {"symbols": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                     "benchmark_symbol": "AAA"},
        "validation": {"final_test_start": "2026-01-01"},
    })
    effects = (CalendarEffect(facts_column="is_turn_of_month", drift_bps=80.0),) \
        if with_effect else ()
    market = SyntheticMarket(seed=seed, base=GBM(mu=0.0, sigma=0.05), effects=effects)
    symbols = ("AAA", "BBB", "CCC", "DDD", "EEE")
    panel = market.generate(symbols, date(2012, 1, 1), date(2021, 12, 31))

    names = ("cal_turn_of_month", "cal_is_friday", "cal_is_monday", "rsi_14_pctile",
             "mom_60", "rel_volume_20", "vol_pctile_252", "natr_14", "close_vs_sma20")
    specs = tuple(get_spec(n) for n in names)
    features = build_features(panel, cfg, specs)
    roundtrip = 0.001
    labels = forward_return_labels(panel, (5,), benchmark_symbol="AAA",
                                   roundtrip_cost=roundtrip)
    merged = features.merge(
        labels[["symbol", "date", "label_end", "gross_ret", "excess_net_ret"]],
        on=["symbol", "date"], how="inner").dropna(subset=["excess_net_ret"])
    # benchmark-adjusting a market-wide calendar effect nets it out; for the
    # canary use net (cost-adjusted) raw returns as the target.
    merged = merged.assign(excess_net_ret=merged["gross_ret"] - roundtrip)

    rules, trials = stability_select(
        merged.reset_index(drop=True), names, "excess_net_ret", seed=seed,
        min_inner_n=30, max_fit_rows=30_000, min_symbols=3,
    )
    passing = [r for r in rules if r.passes]
    hits = [r for r in passing if "cal_turn_of_month" in r.signature]
    return {
        "with_effect": with_effect,
        "trials": trials,
        "structural_rules": len(rules),
        "passing": len(passing),
        "tom_rule_found": bool(hits),
        "tom_best": _rule_row(hits[0]) if hits else None,
        "passing_rows": [_rule_row(r) for r in passing[:8]],
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prep", "year", "report", "canary"],
                        required=True)
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args()
    configure()
    if args.stage == "canary":
        for with_effect in (True, False):
            result = run_canary(with_effect=with_effect)
            print(json.dumps(result, indent=2, default=str))
        return 0
    cfg = load_config("configs/broad.yaml")
    catalog = DataCatalog(cfg)
    if args.stage == "prep":
        stage_prep(cfg, catalog)
    elif args.stage == "year":
        if args.year is None:
            raise SystemExit("--year required")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        stage_year(cfg, catalog, args.year)
    else:
        stage_report(cfg, catalog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
