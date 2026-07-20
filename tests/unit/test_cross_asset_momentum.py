from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from edgestack.data.catalog import DataCatalog
from edgestack.research.cross_asset_momentum import (
    CrossAssetMomentumManifest,
    build_monthly_momentum_weights,
    cross_asset_portfolio_returns,
    load_cross_asset_manifest,
    register_cross_asset_campaign,
)
from edgestack.research.schemas import CampaignLifecycle
from edgestack.research.store import ResearchStore

MANIFEST = Path("configs/cross_asset_momentum.yaml")


def _parameters(*, maximum_asset_weight: float = 0.5) -> dict[str, int | float]:
    return {
        "lookback_sessions": 2,
        "skip_sessions": 0,
        "top_k": 3,
        "trend_lookback_sessions": 0,
        "volatility_window_sessions": 2,
        "maximum_asset_weight": maximum_asset_weight,
    }


def test_manifest_preregisters_exact_family_and_cannot_enable_promotion(cfg) -> None:
    manifest = load_cross_asset_manifest(MANIFEST)
    assert len(manifest.candidates()) == 144
    assert manifest.research_end < manifest.final_holdout_start
    assert manifest.previously_accessed
    assert not manifest.promotion_eligible

    payload = manifest.model_dump(mode="json")
    payload["promotion_eligible"] = True
    with pytest.raises(ValidationError, match="promotion-ineligible"):
        CrossAssetMomentumManifest.model_validate(payload)

    store = ResearchStore(DataCatalog(cfg))
    first = register_cross_asset_campaign(store, manifest)
    second = register_cross_asset_campaign(store, manifest)
    assert first.manifest_hash == second.manifest_hash
    assert first.lifecycle is CampaignLifecycle.READY
    with store.catalog.connect() as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM trial_ledger_v2 WHERE experiment_id = ?",
            [manifest.campaign_id],
        ).fetchone()
    assert row is not None and row[0] == 144


def test_month_end_signal_activates_only_at_next_open_and_never_looks_ahead() -> None:
    index = pd.bdate_range("2020-01-27", periods=8)
    close = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0, 104.0, 1.0, 1.0, 1.0],
            "B": [100.0, 99.0, 98.0, 97.0, 96.0, 200.0, 200.0, 200.0],
        },
        index=index,
    )
    weights = build_monthly_momentum_weights(close, _parameters())

    assert weights.loc[pd.Timestamp("2020-01-31")].sum() == 0.0
    assert weights.loc[pd.Timestamp("2020-02-03"), "A"] == 0.5
    assert weights.loc[pd.Timestamp("2020-02-03"), "B"] == 0.0


def test_weight_cap_leaves_unused_cash_and_execution_excludes_prior_overnight() -> None:
    index = pd.bdate_range("2020-01-27", periods=8)
    close = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 104.0, 106.0, 106.0, 106.0, 106.0],
            "B": [100.0, 100.5, 101.5, 102.0, 103.0, 103.0, 103.0, 103.0],
            "C": [100.0, 102.0, 103.0, 105.0, 107.0, 107.0, 107.0, 107.0],
        },
        index=index,
    )
    adjusted_open = close.copy()
    adjusted_open.loc[pd.Timestamp("2020-01-31"), "A"] = 100.0
    adjusted_open.loc[pd.Timestamp("2020-02-03"), "A"] = 120.0
    adjusted_open.loc[pd.Timestamp("2020-02-04"), "A"] = 132.0
    cash = pd.Series(0.0, index=index)
    parameters = _parameters(maximum_asset_weight=0.2)

    returns, weights, _turnover = cross_asset_portfolio_returns(
        close,
        adjusted_open,
        cash,
        parameters,
        roundtrip_cost_bps=0.0,
    )
    execution_date = pd.Timestamp("2020-02-03")
    assert float(weights.loc[execution_date].max()) <= 0.2
    assert float(weights.loc[execution_date].sum()) <= 0.6 + 1e-12
    # The Jan-31 close signal earns only Feb-03-open to Feb-04-open.  The
    # Jan-31-open to Feb-03-open jump is already gone when the order fills.
    expected = float(weights.loc[execution_date, "A"]) * 0.10
    expected += sum(
        float(weights.loc[execution_date, symbol])
        * (
            float(adjusted_open.loc[pd.Timestamp("2020-02-04"), symbol])
            / float(adjusted_open.loc[execution_date, symbol])
            - 1.0
        )
        for symbol in ("B", "C")
    )
    assert np.isclose(returns.loc[execution_date], expected)
