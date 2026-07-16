"""Multi-horizon ML sweep: at which horizon (1d / 1w / 2w / 1m) can ANY model
beat the base rate out of fold, on the broad point-in-time universe?

Uses the existing model ladder (base-rate -> logistic -> HistGB) with purged
walk-forward OOF evaluation, pre-2024 data only. A subsample cap keeps each
fit tractable; the ladder's selection rule already demands a >=1% OOF
log-loss improvement before crowning a complex model.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from edgestack.config import EdgeStackConfig, load_config
from edgestack.data.catalog import DataCatalog, atomic_write_bytes
from edgestack.labels.forward_returns import forward_return_labels
from edgestack.logging import configure
from edgestack.models.training import train_models
from edgestack.types import Side

HORIZONS = (1, 5, 10, 21)
MAX_ROWS = 250_000


def main() -> int:
    configure()
    t0 = time.time()
    base = load_config("configs/broad.yaml")
    raw = base.model_dump(mode="json")
    raw["signals"]["horizons"] = list(HORIZONS)
    raw["validation"]["embargo_sessions"] = 21
    cfg = EdgeStackConfig.model_validate(raw)

    catalog = DataCatalog(cfg)  # guard truncates before 2024
    panel = catalog.load_panel()
    merged = pd.read_parquet("data/features/rules_merged.parquet")
    features = merged.drop(columns=["label_end", "gross_ret", "excess_net_ret", "mae", "close"])
    features = features.loc[features["date"] < pd.Timestamp("2024-01-01")]
    if len(features) > MAX_ROWS:
        rng = np.random.default_rng(cfg.project.random_seed)
        idx = np.sort(rng.choice(len(features), size=MAX_ROWS, replace=False))
        features = features.iloc[idx].reset_index(drop=True)
    labels = forward_return_labels(panel, HORIZONS, benchmark_symbol="SPY")
    print(
        f"features: {len(features):,} rows (subsampled), "
        f"labels: {len(labels):,} ({time.time() - t0:.0f}s)"
    )

    models = train_models(features, labels, cfg, sides=(Side.LONG,))
    rows = []
    for m in models:
        edge = m.metrics["base_rate_log_loss"] - m.metrics["oof_log_loss"]
        rows.append(
            {
                "horizon": m.horizon,
                "selected_model": m.name,
                "oof_log_loss": round(m.metrics["oof_log_loss"], 5),
                "base_rate_log_loss": round(m.metrics["base_rate_log_loss"], 5),
                "improvement": round(edge, 5),
                "oof_ece": round(m.metrics["oof_ece"], 4),
                "oof_n": int(m.metrics["oof_n"]),
            }
        )
    print(
        f"\n{'horizon':>7} {'selected':>18} {'OOF ll':>9} {'base ll':>9} {'improve':>9} {'ECE':>7}"
    )
    for r in rows:
        print(
            f"{r['horizon']:>7} {r['selected_model']:>18} "
            f"{r['oof_log_loss']:>9.4f} {r['base_rate_log_loss']:>9.4f} "
            f"{r['improvement']:>9.5f} {r['oof_ece']:>7.3f}"
        )
    atomic_write_bytes(
        Path("artifacts") / "horizon_sweep.json", json.dumps(rows, indent=2).encode()
    )
    print(f"\nsaved -> artifacts/horizon_sweep.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
