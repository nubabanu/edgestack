"""Feature correctness spot checks."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from edgestack.config import EdgeStackConfig
from edgestack.data.providers.synthetic import SyntheticMarket
from edgestack.exceptions import LeakageError
from edgestack.features.binning import QuantileBinner
from edgestack.features.momentum import wilder_rsi
from edgestack.features.registry import all_specs, build_features, featureset_id, get_spec


@pytest.fixture(scope="module")
def feature_cfg() -> EdgeStackConfig:
    return EdgeStackConfig.model_validate(
        {
            "universe": {"symbols": ["AAA", "BBB"], "benchmark_symbol": "AAA"},
            "validation": {"final_test_start": "2021-01-01"},
        }
    )


@pytest.fixture(scope="module")
def feats(feature_cfg: EdgeStackConfig) -> pd.DataFrame:
    panel = SyntheticMarket(seed=21).generate(("AAA", "BBB"), date(2016, 1, 1), date(2019, 12, 31))
    return build_features(panel, feature_cfg)


def test_registry_has_broad_coverage() -> None:
    specs = all_specs()
    assert len(specs) >= 55
    families = {s.family for s in specs}
    assert len(families) >= 8
    assert len(featureset_id()) == 12
    assert get_spec("rsi_14").min_history == 15


def test_bounded_features_stay_bounded(feats: pd.DataFrame) -> None:
    assert feats["rsi_14"].dropna().between(0, 100).all()
    assert feats["stoch_k_14"].dropna().between(0, 100).all()
    assert feats["close_loc"].dropna().between(0, 1).all()
    assert feats["donchian_pos_55"].dropna().between(0, 1).all()
    assert feats["trend_consistency_60"].dropna().between(0, 1).all()
    assert set(feats["breakout_20"].dropna().unique()) <= {0.0, 1.0}


def test_rsi_matches_known_direction() -> None:
    rising = pd.Series(np.linspace(100, 150, 60))
    falling = pd.Series(np.linspace(150, 100, 60))
    assert wilder_rsi(rising).iloc[-1] > 90
    assert wilder_rsi(falling).iloc[-1] < 10


def test_engulfing_detector_on_constructed_bars(feature_cfg: EdgeStackConfig) -> None:
    from edgestack.features.candles import candle_bull_engulf

    df = pd.DataFrame(
        {
            "open": [10.0, 9.4],
            "close": [9.5, 10.2],  # down bar then up bar engulfing it
            "high": [10.1, 10.3],
            "low": [9.3, 9.3],
        },
        index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
    )
    out = candle_bull_engulf(df)
    assert out.iloc[1] == 1.0


def test_cross_sectional_ranks_use_only_same_date(feats: pd.DataFrame) -> None:
    # With two symbols, ranks per date must be {0.5, 1.0} (percentile of 2).
    one_date = feats.loc[feats["date"] == feats["date"].max(), "cs_rank_mom_60"]
    assert sorted(one_date.tolist()) == [0.5, 1.0]
    # Breadth is identical across symbols on a given date.
    last = feats.loc[feats["date"] == feats["date"].max(), "breadth_above_sma200"]
    assert last.nunique() == 1


def test_calendar_features_match_calendar(feats: pd.DataFrame) -> None:
    fridays = pd.DatetimeIndex(feats.loc[feats["cal_is_friday"] == 1.0, "date"]).dayofweek
    assert (fridays == 4).all()


def test_quantile_binner_contract() -> None:
    train = pd.DataFrame({"x": np.arange(100.0)})
    binner = QuantileBinner(quantiles=(0.1, 0.5, 0.9)).fit(train, ("x",))
    assert binner.threshold("x", 0.5) == pytest.approx(49.5)
    with pytest.raises(LeakageError):
        binner.threshold("y", 0.5)
    with pytest.raises(LeakageError):
        binner.threshold("x", 0.25)
    with pytest.raises(LeakageError):
        QuantileBinner(quantiles=(0.5,)).threshold("x", 0.5)
