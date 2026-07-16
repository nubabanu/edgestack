"""Live tilt scoring: apply the PRE-2024-FROZEN edge system to the latest session.

Deployment semantics, honestly simulated:
- edges discovered + validated on data strictly before 2024-01-01 (FDR, DSR,
  cost gates) — untouched by anything after;
- quantile thresholds frozen from pre-2024 data;
- scored on the most recent session in the catalog (2.5 years out of sample).

Output: ranked long tilts, regime context, and the historically-weakest
cohorts (research-only short list — NO borrow data, not executable).
Everything is a small statistical tilt, not a prediction.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, date, datetime
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
    generate_candidates,
)
from edgestack.discovery.conditions import evaluate_condition
from edgestack.features.binning import QuantileBinner
from edgestack.features.registry import build_features, get_spec
from edgestack.labels.forward_returns import forward_return_labels
from edgestack.logging import configure
from edgestack.scoring.conviction import ConvictionInputs, conviction
from edgestack.scoring.ranking import aggregate_evidence
from edgestack.types import EdgeStatus, Side
from edgestack.validation.walk_forward import validate_batch

HORIZON = 10
FREEZE = pd.Timestamp("2024-01-01")

FEATURE_SET = tuple(dict.fromkeys([
    *CONTINUOUS_RULE_FEATURES, *BINARY_RULE_FEATURES,
    "bench_trend_200", "bench_vol_20", "natr_14", "realized_vol_20", "mom_60",
]))


def main() -> int:
    configure()
    t0 = time.time()
    cfg = load_config("configs/live.yaml")
    catalog = DataCatalog(cfg)
    universe = PitSP500Universe(Path("data/cache/universe"), earliest=date(2011, 1, 1))

    panel = catalog.load_panel()
    latest = pd.to_datetime(panel["date"]).max()
    print(f"panel through {latest.date()}; building features ...")
    specs = tuple(get_spec(n) for n in FEATURE_SET)
    features = build_features(panel, cfg, specs)
    labels = forward_return_labels(panel, (HORIZON,), benchmark_symbol="SPY")
    merged = features.merge(
        labels[["symbol", "date", "label_end", "gross_ret"]],
        on=["symbol", "date"], how="left",
    ).merge(panel[["symbol", "date", "close", "volume"]], on=["symbol", "date"])
    member = universe.membership_mask(merged["symbol"], merged["date"])
    merged = merged.loc[(member & (merged["close"] >= 5.0)).to_numpy()].reset_index(drop=True)
    print(f"eligible rows: {len(merged):,} ({time.time()-t0:.0f}s)")

    # --- frozen research: discovery + validation strictly before 2024 -------
    pre = merged.loc[(merged["date"] < FREEZE)
                     & merged["gross_ret"].notna()].reset_index(drop=True)
    features_pre = pre.drop(columns=["label_end", "gross_ret", "close", "volume"])
    labels_frame = pre[["symbol", "date", "label_end", "gross_ret"]].assign(
        horizon=HORIZON)
    experiment_id = catalog.record_experiment("live_frozen_edges",
                                              end_date=FREEZE.date())
    batch = generate_candidates(features_pre, labels_frame, cfg, experiment_id)
    edges = validate_batch(batch, features_pre, labels_frame, cfg)
    validated = [e for e in edges if e.lifecycle.status is EdgeStatus.VALIDATED
                 and e.identity.direction is Side.LONG]
    print(f"frozen (pre-2024) validated long edges: {len(validated)} "
          f"({time.time()-t0:.0f}s)")

    binnable = tuple(c for c in (*CONTINUOUS_RULE_FEATURES, "bench_trend_200",
                                 "bench_vol_20") if c in merged.columns)
    frozen_binner = QuantileBinner(quantiles=cfg.discovery.quantile_bins).fit(
        pre, binnable)

    # --- score the latest session -------------------------------------------
    today = merged.loc[merged["date"] == latest].reset_index(drop=True)
    bench_trend = float(today["bench_trend_200"].iloc[0])
    bench_vol = float(today["bench_vol_20"].iloc[0])
    regime = ("UP" if bench_trend > 0.02 else
              "DOWN" if bench_trend < -0.02 else "SIDEWAYS")
    vol_state = "HIGH" if bench_vol > 0.20 else ("LOW" if bench_vol < 0.12 else "MEDIUM")
    print(f"\nCURRENT REGIME: market {bench_trend:+.1%} vs 200dma -> {regime}; "
          f"vol {bench_vol:.0%} -> {vol_state}")

    match_matrix = {}
    for e in validated:
        match_matrix[e.identity.edge_id] = evaluate_condition(
            e.identity.condition, today, frozen_binner).to_numpy()

    rows = []
    for i in range(len(today)):
        matched = [e for e in validated if match_matrix[e.identity.edge_id][i]]
        agg = aggregate_evidence(matched, cfg.scoring)
        if agg is None:
            continue
        r = today.iloc[i]
        prob = float(np.mean([e.stats.probability_of_positive_net_return
                              for e in matched]))
        vol_ann = float(r.get("realized_vol_20", np.nan))
        per_trade_vol = (vol_ann / np.sqrt(252) * np.sqrt(HORIZON)
                         if np.isfinite(vol_ann) else 0.05)
        dv = float(r["close"] * r["volume"])
        result = conviction(ConvictionInputs(
            calibrated_probability=prob,
            expected_net_return=agg.expected_net_return,
            expected_risk=per_trade_vol,
            effective_sample_size=agg.effective_n,
            stability_score=agg.stability,
            out_of_sample_score=agg.out_of_sample,
            deflated_sharpe=agg.deflated_sharpe,
            regime_similarity=1.0 if regime == "UP" else 0.4,
            liquidity_score=float(np.clip((np.log10(max(dv, 1)) - 6) / 3, 0, 1)),
            data_quality_score=float(r.notna().mean()),
            cost_survival_fraction=agg.cost_survival,
            ci_width=agg.ci_width, tail_risk=agg.tail_risk,
        ), shrinkage_min_sample=cfg.scoring.shrinkage_min_sample,
           logistic_slope=cfg.scoring.logistic_slope)
        natr = float(r.get("natr_14", np.nan))
        atr = natr * r["close"] if np.isfinite(natr) and natr > 0 else 0.02 * r["close"]
        rows.append({
            "symbol": r["symbol"], "close": float(r["close"]),
            "conviction": result.score, "E[net]_10d": agg.expected_net_return,
            "hit": prob, "n_edges": len(matched),
            "families": len({e.identity.family for e in matched}),
            "top_edges": "; ".join(e.identity.name.split(": ", 1)[1]
                                    for e in matched[:2]),
            "stop": float(r["close"] - 2 * atr),
            "target": float(r["close"] + 2.5 * atr),
        })

    board = pd.DataFrame(rows).sort_values(
        ["conviction", "E[net]_10d"], ascending=False)
    print(f"\n=== TOP LONG TILTS for {latest.date()} "
          f"(frozen pre-2024 system, {len(board)} symbols matched) ===")
    cols = ["symbol", "close", "conviction", "E[net]_10d", "hit", "n_edges",
            "families", "stop", "target"]
    print(board[cols].head(10).to_string(index=False,
          formatters={"E[net]_10d": "{:+.2%}".format, "hit": "{:.0%}".format,
                      "conviction": "{:.0f}".format, "close": "{:.2f}".format,
                      "stop": "{:.2f}".format, "target": "{:.2f}".format}))
    print("\ntop candidate evidence:", board.iloc[0]["top_edges"])

    # --- JSON export for the API (/board) and the Android companion app -----
    head = board.head(10).rename(columns={"E[net]_10d": "e_net_10d"})
    payload = {
        "schema_version": 1,
        "as_of": str(latest.date()),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "disclaimer": "Research output only. Not investment advice.",
        "regime": {"bench_trend_200": bench_trend, "bench_vol_20": bench_vol,
                   "trend": regime, "vol": vol_state},
        "rows": json.loads(head.to_json(orient="records")),
    }
    atomic_write_bytes(Path("artifacts") / "live_board.json",
                       json.dumps(payload, indent=1).encode())
    print("board exported -> artifacts/live_board.json")

    # --- research-only short list: historically weakest cohorts NOW ---------
    weak = today.loc[
        ((today["overnight_gap"] < -0.03)
         | ((today["breakout_20"] == 1.0) & (bench_trend <= 0.02))
         | ((today["vol_ratio_5_60"] > 1.5) & (abs(bench_trend) <= 0.02)
            & (bench_vol > 0.2))),
        ["symbol", "close", "overnight_gap", "breakout_20", "vol_ratio_5_60"],
    ]
    print("\n=== HISTORICALLY WEAK COHORTS today (research-only short list; "
          "NO validated short edge exists, NO borrow data) ===")
    print(weak.head(10).to_string(index=False) if len(weak) else
          "  (no symbols in the weak cohorts today)")
    print(f"\nelapsed {time.time()-t0:.0f}s — research output, not investment advice")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
