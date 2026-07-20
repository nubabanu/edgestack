"""Run the bounded opening-spike/fade research campaign.

Research and paper trading only.  This command cannot create orders, broker
tickets, canonical weights, or promotion decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/opening_fade.yaml")
    parser.add_argument("--symbols", help="Comma-separated bounded subset")
    parser.add_argument("--interval", choices=("1m", "5m", "15m"))
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--audit-data", action="store_true")
    parser.add_argument("--collect-forward", action="store_true")
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--run-strategies", action="store_true")
    parser.add_argument("--cost-sensitivity", action="store_true")
    parser.add_argument("--render-report", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    return parser


def _symbols(raw: str | None, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return defaults
    values = tuple(dict.fromkeys(item.strip().upper() for item in raw.split(",") if item.strip()))
    if not values:
        raise ValueError("--symbols must contain at least one ticker")
    return values


def _collect(
    cfg, symbols: tuple[str, ...], interval: int | None, start: date | None, end: date | None
) -> dict:
    from edgestack.config import load_config
    from edgestack.pipelines import run_intraday_download

    repository_cfg = load_config(ROOT / cfg.repository_config)
    through = end or date.today()
    intervals = (interval,) if interval else (60, 15, 5, 1)
    rows = []
    limits = {1: 6, 5: 58, 15: 58, 60: 728}
    for minutes in intervals:
        since = start or (through - timedelta(days=min(7, limits[minutes])))
        run_intraday_download(
            repository_cfg,
            since,
            through,
            symbols=symbols,
            provider=cfg.provider,
            interval=f"{minutes}m",
            include_prepost=True,
        )
        rows.append(
            {
                "interval_minutes": minutes,
                "start": since.isoformat(),
                "end": through.isoformat(),
                "symbols": list(symbols),
                "extended_hours_requested": True,
            }
        )
    return {"status": "COLLECTED", "runs": rows}


def _smoke(cfg) -> dict:
    from edgestack.research.opening_fade import (
        DataSufficiency,
        build_daily_context,
        build_session_features,
        generate_candidate_trades,
        occurrence_analysis,
        synthetic_intraday_fixture,
    )
    from edgestack.types import CostScenario

    intraday, daily = synthetic_intraday_fixture(sessions=80, injected_fade=0.004)
    features, bars = build_session_features(
        intraday,
        daily_context=build_daily_context(daily),
        definition=cfg.definition,
    )
    audit = {
        "premarket_available": False,
        "economic_calendar": {"available": False},
        "breadth": {"research_usable": False},
        "index_futures": [],
        "bid_ask": "not available historically",
    }
    trades, ledger = generate_candidate_trades(
        cfg, features, bars, audit, scenario=CostScenario.CONSERVATIVE
    )
    return {
        "status": "SMOKE_OK",
        "classification": DataSufficiency.EXPLORATORY,
        "eligible_sessions": int(features["eligible"].sum()),
        "trades": len(trades),
        "trial_count": len(ledger),
        "occurrence": occurrence_analysis(features, cfg.definition),
        "paper_only": True,
    }


def main() -> int:
    from edgestack.research.opening_fade import (
        audit_local_data,
        load_campaign_config,
        run_opening_fade_campaign,
    )

    parser = _parser()
    args = parser.parse_args()
    if args.describe_only and args.run_strategies:
        parser.error("--describe-only and --run-strategies are mutually exclusive")
    cfg = load_campaign_config(ROOT / args.config)
    symbols = _symbols(args.symbols, cfg.primary_symbols)
    interval = int(args.interval.removesuffix("m")) if args.interval else None
    try:
        if args.smoke:
            payload = _smoke(cfg)
        elif args.collect_forward:
            payload = _collect(cfg, symbols, interval, args.start, args.end)
        elif args.audit_data:
            payload = audit_local_data(ROOT, cfg)
        else:
            result, trades, destination = run_opening_fade_campaign(
                cfg,
                repo_root=ROOT,
                symbols=symbols,
                interval_minutes=interval,
                start=args.start,
                end=args.end,
                publish=not args.no_publish,
                run_strategies=not args.describe_only,
            )
            payload = {
                "summary": result["summary"],
                "conclusion": result["conclusion"],
                "classification": result["data_audit"]["overall_classification"],
                "eligible_sessions": result["occurrence"].get("total_eligible_sessions", 0),
                "trades": len(trades),
                "trial_count": result["validation"]["trial_count"],
                "paper_observation_candidates": result["validation"][
                    "paper_observation_candidates"
                ],
                "artifact_path": str(destination) if destination else None,
                "next_research_step": result["next_research_step"],
                "paper_only": True,
            }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        else:
            print(json.dumps(payload, indent=2, default=str))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc), "paper_only": True}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
