"""End-to-end artifact and bounded-family checks for opening-fade research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from edgestack.research.opening_fade import (
    DataSufficiency,
    build_daily_context,
    build_session_features,
    generate_candidate_trades,
    load_campaign_config,
    occurrence_analysis,
    publish_immutable_run,
    synthetic_intraday_fixture,
)
from edgestack.types import CostScenario

ROOT = Path(__file__).resolve().parents[2]


def test_pipeline_counts_unsupported_trials_and_publishes_atomically(tmp_path) -> None:
    config = load_campaign_config(ROOT / "configs" / "opening_fade.yaml")
    intraday, daily = synthetic_intraday_fixture(sessions=20, injected_fade=0.003)
    features, bars = build_session_features(
        intraday,
        daily_context=build_daily_context(daily),
        definition=config.definition,
    )
    audit = {
        "overall_classification": DataSufficiency.INSUFFICIENT,
        "provider": "fixture",
        "timestamps": "UTC",
        "premarket_available": False,
        "bid_ask": "not available historically",
        "economic_calendar": {"available": False},
        "breadth": {"research_usable": False, "reason": "fixture has no breadth"},
        "index_futures": [],
    }
    trades, ledger = generate_candidate_trades(
        config,
        features,
        bars,
        audit,
        scenario=CostScenario.CONSERVATIVE,
    )
    assert len(ledger) == len(config.candidates)
    assert {item["family"] for item in ledger} == set("ABCDEFGH")
    assert any(item["status"] == "UNSUPPORTED" for item in ledger)
    assert {
        "signal_available_time_utc",
        "previous_close",
        "premarket_last",
        "official_open",
        "gap_pct",
        "overnight_high",
        "opening_range_high",
        "opening_spike_return",
        "signal_vwap",
        "intraday_atr_at_signal",
        "opening_volume_relative_to_prior_median",
        "previous_day_return",
        "short_trend_state",
        "earnings_flag",
        "cross_index_confirmation_state",
        "modeled_half_spread_bps_per_leg",
        "fill_fraction",
    }.issubset(trades.columns)
    assert (trades["entry_time_utc"] >= trades["signal_available_time_utc"]).all()

    capacity_config = config.model_copy(
        update={
            "costs": config.costs.model_copy(
                update={"assumed_participation": 0.10, "maximum_bar_participation": 0.05}
            )
        }
    )
    capacity_trades, _ = generate_candidate_trades(
        capacity_config,
        features,
        bars,
        audit,
        scenario=CostScenario.CONSERVATIVE,
    )
    assert set(capacity_trades["fill_fraction"]) == {0.5}
    assert set(capacity_trades["fill_assumption"]) == {"PARTIAL_AT_MAXIMUM_BAR_PARTICIPATION"}

    validation_rows = [
        {
            "candidate_id": item.candidate_id,
            "family": item.family,
            "metrics": {
                "trade_count": int((trades.get("candidate_id", "") == item.candidate_id).sum())
            },
            "q_value": 1.0,
            "deflated_sharpe_probability": 0.0,
            "failure_reasons": ["INSUFFICIENT_WALK_FORWARD_HISTORY"],
        }
        for item in config.candidates
    ]
    result = {
        "manifest": {
            "created_at": "2026-07-20T00:00:00+00:00",
            "trial_count": len(config.candidates),
            "paper_only": True,
            "canonical_integration": False,
        },
        "definition": config.definition.model_dump(mode="json"),
        "validation_plan": config.validation.model_dump(mode="json"),
        "data_audit": audit,
        "occurrence": occurrence_analysis(features, config.definition),
        "validation": {
            "trial_count": len(config.candidates),
            "walk_forward_folds": 0,
            "paper_observation_candidates": [],
            "candidates": validation_rows,
        },
        "cost_sensitivity": {
            "monotonic": True,
            "cfd_sensitivity": {"status": "NOT_RUN"},
        },
        "controls": {"paired_comparisons": []},
        "rankings": {"movement_ranking": [], "tradability_edge_ranking": []},
        "warnings": ["fixture is insufficient"],
    }
    destination = publish_immutable_run(result, trades, artifacts_root=tmp_path)
    expected = {
        "manifest.json",
        "data_audit.json",
        "occurrence.json",
        "trades.parquet",
        "metrics.json",
        "report.md",
        "report.html",
    }
    assert {path.name for path in destination.iterdir()} == expected
    pointer = json.loads((tmp_path / "current.json").read_text(encoding="utf-8"))
    assert pointer["paper_only"] is True
    assert pointer["run_id"] == destination.name
    assert pd.read_parquet(destination / "trades.parquet").shape == trades.shape
    occurrence = json.loads((destination / "occurrence.json").read_text(encoding="utf-8"))
    assert occurrence["time_to_fade"]["50"]["count"] > 0
    assert len(occurrence["per_session"]) == occurrence["total_eligible_sessions"]

    # Same content is idempotent and never mutates a completed run.
    assert publish_immutable_run(result, trades, artifacts_root=tmp_path) == destination

    # Audit/publication clocks are metadata and do not fork identical evidence.
    result["manifest"]["created_at"] = "2026-07-21T00:00:00+00:00"
    result["data_audit"]["checked_at"] = "2026-07-21T00:00:00+00:00"
    assert publish_immutable_run(result, trades, artifacts_root=tmp_path) == destination
    assert (
        json.loads((tmp_path / "current.json").read_text(encoding="utf-8"))["published_at"]
        == "2026-07-20T00:00:00+00:00"
    )
