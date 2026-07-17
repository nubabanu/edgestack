"""Generic look-ahead prevention tests.

The core pattern (spec section 38): compute all features, mutate every bar
strictly after a cutoff date T, recompute, and require all values at or
before T to be identical. Because the test iterates the full registry, any
newly registered feature is covered automatically.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from edgestack.config import EdgeStackConfig
from edgestack.data.providers.synthetic import SyntheticMarket
from edgestack.features.registry import all_specs, build_features

CUTOFF = pd.Timestamp("2019-06-14")  # a Friday session in the middle of the panel


@pytest.fixture(scope="module")
def feature_cfg() -> EdgeStackConfig:
    return EdgeStackConfig.model_validate(
        {
            "universe": {"symbols": ["AAA", "BBB", "CCC"], "benchmark_symbol": "AAA"},
            "validation": {"final_test_start": "2021-01-01"},
        }
    )


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return SyntheticMarket(seed=99).generate(
        ("AAA", "BBB", "CCC"), date(2016, 1, 1), date(2020, 12, 31)
    )


def _mutate_after(panel: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Radically alter every bar strictly after the cutoff (OHLC scaled, volume x5)."""
    mutated = panel.copy()
    after = mutated["date"] > cutoff
    for col in ("open", "high", "low", "close", "adj_close"):
        mutated.loc[after, col] *= 3.0
    mutated.loc[after, "volume"] *= 5.0
    return mutated


def test_every_registered_feature_is_causal(
    feature_cfg: EdgeStackConfig, panel: pd.DataFrame
) -> None:
    specs = all_specs()
    base = build_features(panel, feature_cfg, specs)
    mutated = build_features(_mutate_after(panel, CUTOFF), feature_cfg, specs)

    base_past = base.loc[base["date"] <= CUTOFF].reset_index(drop=True)
    mut_past = mutated.loc[mutated["date"] <= CUTOFF].reset_index(drop=True)

    # Every registered feature must be present in the output ...
    missing = [s.name for s in specs if s.name not in base.columns]
    assert not missing, f"features registered but absent from output: {missing}"

    # ... and unchanged before the cutoff, feature by feature for a precise
    # failure message naming the leaking feature.
    leaking = []
    for spec in specs:
        try:
            pd.testing.assert_series_equal(
                base_past[spec.name], mut_past[spec.name], check_exact=False, rtol=1e-12
            )
        except AssertionError:
            leaking.append(spec.name)
    assert not leaking, f"features leaked future information: {leaking}"


def test_removing_future_rows_does_not_change_past(
    feature_cfg: EdgeStackConfig, panel: pd.DataFrame
) -> None:
    specs = all_specs()
    base = build_features(panel, feature_cfg, specs)
    truncated_panel = panel.loc[panel["date"] <= CUTOFF].reset_index(drop=True)
    truncated = build_features(truncated_panel, feature_cfg, specs)

    lag = max(s.availability_lag for s in specs)
    # Features with availability_lag L are undefined for the last L sessions of
    # a truncated panel (their confirming bars are missing), which is exactly
    # the point: compare only fully-confirmed rows.
    sessions = sorted(truncated_panel["date"].unique())
    confirmed_cutoff = sessions[-(lag + 1)]

    base_past = (
        base.loc[base["date"] <= confirmed_cutoff]
        .reset_index(drop=True)
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    trunc_past = (
        truncated.loc[truncated["date"] <= confirmed_cutoff]
        .reset_index(drop=True)
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(base_past, trunc_past, check_exact=False, rtol=1e-12)


def test_labels_never_enter_features(feature_cfg: EdgeStackConfig, panel: pd.DataFrame) -> None:
    """Label columns must not appear in the feature dataset, and feature
    engineering must not consume label frames at all."""
    from edgestack.labels.forward_returns import LABEL_COLUMNS

    feats = build_features(panel, feature_cfg)
    forbidden = set(LABEL_COLUMNS) - {"symbol", "date"}
    assert not forbidden.intersection(feats.columns)
    # No registered feature may be named like a label output either.
    assert not any(s.name in forbidden for s in all_specs())


def test_pivot_confirmation_respects_availability_delay(
    feature_cfg: EdgeStackConfig, panel: pd.DataFrame
) -> None:
    """A confirmed pivot must not be visible before its confirming bars exist."""
    feats = build_features(panel, feature_cfg)
    one = feats.loc[feats["symbol"] == "AAA"].set_index("date")
    pivots = one.index[one["pivot_high_2"] == 1.0]
    assert len(pivots) > 0
    bars = panel.loc[panel["symbol"] == "AAA"].set_index("date")
    for stamp in pivots[:25]:
        # The flag stored at date D refers to the pivot bar two sessions back:
        # that bar's high must exceed its two neighbors on each side, all of
        # which are <= D. Nothing after D is needed.
        loc = bars.index.get_loc(stamp)
        pivot_loc = loc - 2
        window = bars["high"].iloc[pivot_loc - 2 : pivot_loc + 3]
        assert window.iloc[2] == window.max()
        assert window.index.max() <= stamp
