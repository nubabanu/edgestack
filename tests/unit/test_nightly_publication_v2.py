"""Canonical nightly baseline assembly test."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.calendar import TradingCalendar
from edgestack.data.catalog import DataCatalog
from edgestack.paper.canonical import load_paper_state
from edgestack.recommendation.financing import FundingRateObservation
from edgestack.recommendation.nightly import (
    build_and_publish_canonical_baseline,
    load_legacy_watchlist,
)
from edgestack.recommendation.schemas import EvidenceGrade, RecommendationStatus
from edgestack.recommendation.service import CanonicalBundleRepository


def test_nightly_publishes_fixed_baseline_from_adjusted_opens(tmp_path: Path) -> None:
    cfg = EdgeStackConfig.model_validate(
        {
            "paths": {
                "data_dir": tmp_path / "data",
                "artifacts_dir": tmp_path / "artifacts",
            },
            "universe": {
                "source": "synthetic",
                "symbols": ["SPY", "TLT", "SHY", "GLD"],
            },
            "validation": {"final_test_start": "2030-01-01"},
        }
    )
    session = date(2026, 7, 16)
    next_session = date(2026, 7, 17)
    sessions = TradingCalendar().sessions(date(2024, 1, 1), next_session)[-321:]
    frames = []
    for offset, symbol in enumerate(("SPY", "TLT", "SHY", "GLD")):
        rng = np.random.default_rng(offset + 10)
        returns = rng.normal(0.0002, 0.006 + offset * 0.0005, len(sessions))
        close = (100 + offset * 10) * np.cumprod(1 + returns)
        open_price = close * (1 + rng.normal(0, 0.001, len(sessions)))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": sessions,
                    "open": open_price,
                    "high": np.maximum(open_price, close) * 1.005,
                    "low": np.minimum(open_price, close) * 0.995,
                    "close": close,
                    "volume": np.full(len(sessions), 10_000_000.0),
                    "adj_close": close,
                }
            )
        )
    DataCatalog(cfg).write_bars(pd.concat(frames, ignore_index=True), provider="synthetic")
    funding = FundingRateObservation(
        series_id="DGS3MO",
        as_of=session,
        annualized_rate=0.04,
        fetched_at=datetime(2026, 7, 16, 21, tzinfo=UTC),
    )

    publication = build_and_publish_canonical_baseline(
        cfg,
        run_date=session,
        now=datetime(2026, 7, 16, 21, tzinfo=UTC),
        funding_observation=funding,
        bootstrap_replications=100,
    )
    repository = CanonicalBundleRepository(cfg.paths.artifacts_dir)
    bundle = repository.latest()

    assert publication.run_id == repository.pointer().run_id
    assert bundle.base_recommendation.status is RecommendationStatus.BASELINE_ONLY
    assert {
        weight.symbol: weight.weight for weight in bundle.base_recommendation.unlevered_base_weights
    } == {
        "SPY": 0.25,
        "TLT": 0.25,
        "SHY": 0.25,
        "GLD": 0.25,
    }
    assert bundle.base_recommendation.promoted_sleeves == ()
    assert bundle.default_recommendation.effective_leverage == 0.25
    assert bundle.default_recommendation.binding_constraints == ("session_increase",)

    next_funding = FundingRateObservation(
        series_id="DGS3MO",
        as_of=next_session,
        annualized_rate=0.04,
        fetched_at=datetime(2026, 7, 17, 21, tzinfo=UTC),
    )
    build_and_publish_canonical_baseline(
        cfg,
        run_date=next_session,
        now=datetime(2026, 7, 17, 21, tzinfo=UTC),
        funding_observation=next_funding,
        bootstrap_replications=100,
    )
    next_bundle = repository.latest()
    paper = load_paper_state(
        repository,
        initial_equity=cfg.paper.initial_cash,
        risk_state=next_bundle.default_recommendation.output_risk_state,
    )

    assert paper.last_session == next_session
    assert len(paper.fills) == 4
    assert len(paper.positions) == 4
    assert paper.realized_returns[-1].session == next_session
    assert paper.pending_target is not None
    assert next_bundle.default_risk_profile.account_equity == paper.current_equity


def test_legacy_board_is_imported_only_as_insufficient_zero_weight_watchlist(
    tmp_path: Path,
) -> None:
    (tmp_path / "live_board.json").write_text(
        json.dumps({"rows": [{"symbol": "aaa", "conviction": 99}, {"symbol": "AAA"}]}),
        encoding="utf-8",
    )

    watchlist = load_legacy_watchlist(tmp_path)

    assert len(watchlist) == 1
    assert watchlist[0].symbol == "AAA"
    assert watchlist[0].evidence_grade is EvidenceGrade.INSUFFICIENT
    assert watchlist[0].prospective_sessions == 0
    assert "legacy artifact is ineligible" in watchlist[0].zero_weight_reason
