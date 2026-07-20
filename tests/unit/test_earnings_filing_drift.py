from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from edgestack.data.calendar import TradingCalendar
from edgestack.data.catalog import DataCatalog
from edgestack.research.earnings_filing_drift import (
    EarningsFilingManifest,
    _first_open_after,
    filing_portfolio_returns,
    load_earnings_manifest,
    register_earnings_campaign,
    seasonal_random_walk_sue,
)
from edgestack.research.store import ResearchStore

MANIFEST = Path("configs/earnings_filing_drift.yaml")


def test_earnings_manifest_is_finite_registered_and_historical_only(cfg) -> None:
    manifest = load_earnings_manifest(MANIFEST)
    assert len(manifest.candidates()) == 18
    assert manifest.research_end < manifest.final_holdout_start
    assert not manifest.promotion_eligible
    payload = manifest.model_dump(mode="json")
    payload["promotion_eligible"] = True
    with pytest.raises(ValidationError, match="promotion-ineligible"):
        EarningsFilingManifest.model_validate(payload)

    store = ResearchStore(DataCatalog(cfg))
    register_earnings_campaign(store, manifest)
    register_earnings_campaign(store, manifest)
    with store.catalog.connect() as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM trial_ledger_v2 WHERE experiment_id = ?",
            [manifest.campaign_id],
        ).fetchone()
    assert row is not None and row[0] == 18

    mutation = load_earnings_manifest(Path("configs/earnings_seasonal_sue.yaml"))
    register_earnings_campaign(store, mutation)
    assert mutation.feature_definition.method == "seasonal_random_walk_sue"
    assert store.family_trial_count(manifest.family) == 36
    assert store.family_trial_count_as_of(manifest.campaign_id, manifest.family) == 18
    assert store.family_trial_count_as_of(mutation.campaign_id, manifest.family) == 36


def test_seasonal_sue_scale_excludes_the_current_surprise() -> None:
    events = pd.DataFrame(
        {
            "period_end": pd.date_range("2018-03-31", periods=16, freq="QE-DEC"),
            "eps": [
                1.0,
                1.2,
                0.9,
                1.1,
                1.2,
                1.35,
                1.15,
                1.3,
                1.55,
                1.5,
                1.35,
                1.65,
                1.75,
                1.9,
                1.7,
                1.85,
            ],
        }
    )
    first = seasonal_random_walk_sue(events)
    doubled = events.copy()
    prior_year_eps = float(doubled.loc[len(doubled) - 5, "eps"])
    original_difference = float(doubled.loc[len(doubled) - 1, "eps"]) - prior_year_eps
    doubled.loc[len(doubled) - 1, "eps"] = prior_year_eps + 2.0 * original_difference
    second = seasonal_random_walk_sue(doubled)

    assert np.isfinite(first.iloc[-1])
    assert np.isclose(second.iloc[-1], 2.0 * first.iloc[-1])


def test_filing_availability_uses_first_exchange_open_strictly_after_timestamp() -> None:
    sessions = pd.DatetimeIndex(["2020-01-31", "2020-02-03", "2020-02-04"])
    calendar = TradingCalendar()
    opens_ns = np.asarray(
        [calendar.market_open_at(session).tz_convert("UTC").value for session in sessions],
        dtype=np.int64,
    )
    premarket = pd.Timestamp("2020-02-03T13:00:00Z")
    after_close = pd.Timestamp("2020-02-03T22:00:00Z")

    assert _first_open_after(premarket, sessions, opens_ns) == pd.Timestamp("2020-02-03")
    assert _first_open_after(after_close, sessions, opens_ns) == pd.Timestamp("2020-02-04")


def test_filing_strategy_excludes_the_move_before_its_executable_open() -> None:
    index = pd.bdate_range("2020-01-30", periods=5)
    adjusted_open = pd.DataFrame(
        {"AAA": [100.0, 100.0, 120.0, 132.0, 132.0]},
        index=index,
    )
    events = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "execution_date": [index[2]],
            "eps_change": [1.5],
            "pre_event_momentum_126": [0.2],
        }
    )
    cash = pd.Series(0.0, index=index)
    candidate: dict[str, int | float | bool] = {
        "minimum_eps_change": 1.0,
        "require_positive_pre_event_momentum": True,
        "horizon_sessions": 1,
    }
    returns, weights, _ = filing_portfolio_returns(
        adjusted_open,
        cash,
        events,
        candidate,
        roundtrip_cost_bps=0.0,
    )

    assert weights.loc[index[1], "AAA"] == 0.0
    assert weights.loc[index[2], "AAA"] == 1.0
    assert np.isclose(returns.loc[index[2]], 0.10)
