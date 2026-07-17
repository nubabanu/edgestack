"""EdgeStack end-to-end demonstration.

Runs the complete research pipeline on a small real-data watchlist (free
Yahoo endpoint); if the network or the feed is unavailable it falls back to
the synthetic market so the demo always completes offline.

    python scripts/demo.py [--config configs/demo.yaml] [--offline]

Steps: download -> validate -> features -> discover -> validate edges ->
train models -> generate + rank signals -> backtest -> report.
Everything printed is research output, not investment advice.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgestack import pipelines
from edgestack.config import load_config
from edgestack.exceptions import EdgeStackError, ProviderError
from edgestack.logging import configure


def _step(title: str):
    print(f"\n{'=' * 70}\n== {title}\n{'=' * 70}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument(
        "--offline", action="store_true", help="use the synthetic provider instead of Yahoo"
    )
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default=str(date.today()))
    args = parser.parse_args()

    configure()
    cfg = load_config(args.config)
    t0 = time.time()

    _step("1. data download")
    provider = "synthetic" if args.offline else None
    try:
        pipelines.run_data_download(
            cfg,
            date.fromisoformat(args.start),
            date.fromisoformat(args.end),
            provider=provider,
        )
    except (ProviderError, OSError) as exc:
        print(f"real-data provider unavailable ({exc}); falling back to synthetic data")
        pipelines.run_data_download(
            cfg,
            date.fromisoformat(args.start),
            date.fromisoformat(args.end),
            provider="synthetic",
        )

    _step("2. data quality validation")
    pipelines.run_data_validate(cfg)

    _step("3. feature build")
    pipelines.run_features_build(cfg)

    _step("4. edge discovery (every evaluated rule is counted as a trial)")
    pipelines.run_edges_discover(cfg)

    _step("5. walk-forward validation with FDR control")
    pipelines.run_edges_validate(cfg)

    _step("6. model training + out-of-fold calibration")
    pipelines.run_models_train(cfg)

    _step("7. signal generation and ranking")
    try:
        pipelines.run_signals_generate(cfg)
        pipelines.run_signals_rank(cfg, top=10)
    except EdgeStackError as exc:
        print(f"signal generation: {exc}")
        print(
            "(no validated edges on real data is a legitimate outcome — the "
            "system abstains rather than inventing signals)"
        )

    _step("8. out-of-sample backtest under conservative costs")
    try:
        pipelines.run_backtest(cfg)
        pipelines.run_report_backtest(cfg)
    except EdgeStackError as exc:
        print(f"backtest: {exc}")

    _step("done")
    print(f"total wall time: {time.time() - t0:.0f}s")
    print(
        "Reminder: research and paper-trading output only; free data carries "
        "survivorship bias; nothing here is investment advice."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
