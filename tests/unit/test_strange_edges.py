from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from edgestack.data.catalog import DataCatalog
from edgestack.data.providers.usaspending import USAspendingContractProvider
from edgestack.research.store import ResearchStore
from edgestack.research.strange_edges import (
    _portfolio_path,
    build_contract_feature,
    load_strange_edges_manifest,
    register_strange_edge_campaigns,
    registered_government_trials,
)


class _Response:
    status_code = 200
    headers: ClassVar[dict[str, str]] = {}

    def json(self):
        return {
            "group": "month",
            "results": [
                {
                    "time_period": {"fiscal_year": "2020", "month": "1"},
                    "Contract_Obligations": 10.0,
                },
                {
                    "time_period": {"fiscal_year": "2020", "month": "4"},
                    "Contract_Obligations": -2.0,
                },
            ],
            "messages": [],
        }


class _Session:
    def __init__(self) -> None:
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        return _Response()


def test_usaspending_maps_fiscal_months_and_reuses_request_cache(tmp_path: Path) -> None:
    session = _Session()
    provider = USAspendingContractProvider(raw_dir=tmp_path, session=session)
    first = provider.fetch_monthly_contract_obligations(
        {"ABC": ("ACME",)},
        date(2019, 10, 1),
        date(2020, 1, 31),
    )
    assert session.calls == 1
    october = first.loc[
        first["event_time"] == pd.Timestamp("2019-10-31"), "contract_obligations"
    ].item()
    january = first.loc[
        first["event_time"] == pd.Timestamp("2020-01-31"), "contract_obligations"
    ].item()
    assert october == 10
    assert january == -2
    second = provider.fetch_monthly_contract_obligations(
        {"ABC": ("ACME",)},
        date(2019, 10, 1),
        date(2020, 1, 31),
    )
    assert session.calls == 1
    pd.testing.assert_frame_equal(first, second)


def test_manifest_retains_complete_scope_and_108_blind_trials() -> None:
    manifest = load_strange_edges_manifest(Path("configs/strange_edges.yaml"))
    trials = registered_government_trials(manifest)
    assert len(manifest.hypotheses) == 9
    assert len(manifest.combinations) == 5
    assert len(trials) == 108
    assert len({item.trial_id for item in trials}) == 108
    assert manifest.government_contract_study.acquisition_end < manifest.final_holdout_start
    assert manifest.promotion_eligible is False


def test_contract_feature_is_past_only_and_preserves_negative_obligations() -> None:
    dates = pd.date_range("2018-01-31", periods=48, freq="ME")
    values = np.linspace(10.0, 100.0, len(dates))
    values[30] = -25.0
    frame = pd.DataFrame({"symbol": "ABC", "event_time": dates, "contract_obligations": values})
    original = build_contract_feature(frame, method="trailing_mad", minimum_months_history=24)
    changed = frame.copy()
    changed.loc[changed.index[-1], "contract_obligations"] = 10**12
    revised = build_contract_feature(changed, method="trailing_mad", minimum_months_history=24)
    pd.testing.assert_series_equal(
        original.loc[original.index[:-1], "score"],
        revised.loc[revised.index[:-1], "score"],
    )
    assert np.isfinite(original.loc[30, "score"])


def test_portfolio_path_waits_in_cash_and_enters_only_on_registered_entry() -> None:
    dates = pd.bdate_range("2020-01-01", periods=12)
    returns = pd.DataFrame({"ABC": 0.01, "XYZ": -0.01}, index=dates)
    cash = pd.Series(0.001, index=dates)
    feature = pd.DataFrame(
        {
            "signal_date": [dates[2], dates[2]],
            "entry_date": [dates[3], dates[3]],
            "symbol": ["ABC", "XYZ"],
            "selection_score": [2.0, 1.0],
        }
    )
    path, weights = _portfolio_path(
        feature,
        returns,
        cash,
        horizon=3,
        quantile=0.5,
        roundtrip_cost_bps=0,
        minimum_cross_section=2,
    )
    assert (weights.loc[dates[:3], "ABC"] == 0).all()
    assert (weights.loc[dates[3:6], "ABC"] == 1).all()
    assert (weights.loc[dates[6:], "ABC"] == 0).all()
    assert path.loc[dates[0]] == 0.001
    assert path.loc[dates[3]] == 0.01


def test_registration_exposes_every_exact_input_blocker(cfg) -> None:
    manifest = load_strange_edges_manifest(Path("configs/strange_edges.yaml"))
    register_strange_edge_campaigns(cfg, manifest)
    store = ResearchStore(DataCatalog(cfg))
    counts: dict[str, int] = {}
    for campaign in store.campaigns():
        counts[campaign.lifecycle.value] = counts.get(campaign.lifecycle.value, 0) + 1
    assert counts == {
        "READY": 1,
        "NEEDS_DATA": 1,
        "BLOCKED_LICENSE": 3,
        "BLOCKED_POINT_IN_TIME": 2,
        "BLOCKED_COMPLIANCE": 1,
        "BLOCKED_COMPONENTS": 6,
    }
    gaps = store.gaps()
    assert len(gaps) == 13
    assert all(gap.next_action for gap in gaps)
