"""Nested selection, session clustering, and double cross-fit tests."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from edgestack.exceptions import ValidationError
from edgestack.models.calibration import cross_fit_isotonic
from edgestack.validation.clustered import (
    aggregate_session_contributions,
    paired_sharpe_improvement,
    session_effective_sample_size,
    stationary_mean_test,
)
from edgestack.validation.nested import annual_outer_folds, freeze_outer_decisions
from edgestack.validation.selection import complete_family_tests


def test_outer_test_mutation_cannot_change_training_decision_hash() -> None:
    dates = pd.bdate_range("2017-01-02", "2024-12-31")
    frame = pd.DataFrame({"date": dates, "x": np.arange(len(dates), dtype=float)})
    label_end = pd.Series(dates).shift(-5).fillna(pd.Timestamp(dates[-1]))
    folds = annual_outer_folds(frame["date"], label_end)

    def selector(train: pd.DataFrame, _fold) -> dict:
        return {"threshold": float(train["x"].median()), "last": str(train["date"].max())}

    before = freeze_outer_decisions(frame, folds, selector)
    mutated = frame.copy()
    first_test = folds[0]
    mutated.loc[first_test.test_idx, "x"] = 1e12
    after = freeze_outer_decisions(mutated, folds, selector)
    # The mutated year cannot alter the decision frozen immediately before it.
    # It may legitimately enter later folds' expanding training windows.
    assert before[0].decision_hash == after[0].decision_hash
    assert all(f.definition.test_start.year == int(f.definition.fold_id) for f in folds)


def test_concurrent_trades_count_as_one_session() -> None:
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=120), 25)
    contributions = pd.DataFrame(
        {"date": dates, "net_return_contribution": np.full(len(dates), 0.0001)}
    )
    daily = aggregate_session_contributions(contributions)
    assert len(daily) == 120
    assert session_effective_sample_size(daily) <= 120


def test_stationary_bootstrap_detects_canary_but_not_centered_null() -> None:
    rng = np.random.default_rng(7)
    null = pd.Series(rng.normal(0.0, 0.01, 800))
    canary = null + 0.002
    assert stationary_mean_test(canary, seed=1)["pvalue_greater"] < 0.05
    assert stationary_mean_test(null - null.mean(), seed=1)["pvalue_greater"] > 0.1


def test_paired_sharpe_uses_aligned_sessions() -> None:
    rng = np.random.default_rng(3)
    benchmark = pd.Series(rng.normal(0.0001, 0.01, 700))
    strategy = benchmark * 0.5 + 0.001
    result = paired_sharpe_improvement(strategy, benchmark, seed=5)
    assert result["ci_low"] > 0


def test_cross_fitted_calibration_is_reproducible_and_complete() -> None:
    rng = np.random.default_rng(4)
    raw = rng.uniform(0.05, 0.95, 600)
    true = (rng.uniform(size=600) < raw).astype(int)
    folds = np.repeat(np.arange(3), 200)
    a = cross_fit_isotonic(raw, true, folds)
    b = cross_fit_isotonic(raw, true, folds)
    np.testing.assert_allclose(a.calibrated_oof, b.calibrated_oof)
    assert np.isfinite(a.calibrated_oof).all()
    assert tuple(a.fold_assignments) == tuple(folds)
    with pytest.raises(ValidationError, match="at least two folds"):
        cross_fit_isotonic(raw, true, np.zeros(600, dtype=int))


def test_spa_refuses_incomplete_trial_family() -> None:
    rng = np.random.default_rng(8)
    benchmark = pd.Series(rng.normal(0, 0.01, 100))
    trials = pd.DataFrame({"standalone": rng.normal(0, 0.01, 100)})
    with pytest.raises(ValidationError, match="searched-family mismatch"):
        complete_family_tests(
            benchmark,
            trials,
            complete_trial_ids=("standalone", "failed_compound"),
            reps=100,
        )


def test_complete_family_tests_full_output_and_persistence(tmp_path) -> None:
    from edgestack.validation.selection import BENCHMARK_COLUMN

    rng = np.random.default_rng(17)
    index = pd.bdate_range("2019-01-01", periods=400)
    benchmark = pd.Series(rng.normal(0.0, 0.01, len(index)), index=index)
    trials = pd.DataFrame(
        {
            "alpha": rng.normal(0.001, 0.01, len(index)),
            "beta": rng.normal(0.0, 0.01, len(index)),
            "gamma": rng.normal(0.0, 0.01, len(index)),
        },
        index=index,
    )
    # reps >= 1000: arch's StepM is numerically unstable at low rep counts
    # on families with no clear winner (production always runs >= 2000).
    result = complete_family_tests(
        benchmark,
        trials,
        complete_trial_ids=("alpha", "beta", "gamma"),
        reps=1_000,
        pbo_partitions=8,
        persist_dir=tmp_path / "trial_returns",
        batch_id="unit-batch",
    )
    assert set(result) == {
        "spa",
        "stepm_superior",
        "mcs",
        "pbo",
        "block_length_diagnostic",
        "trial_returns_path",
    }
    assert "included" in result["mcs"] and "pbo" in result["pbo"]
    assert result["block_length_diagnostic"]["configured"] == 20.0

    parquet_path = tmp_path / "trial_returns" / "unit-batch.parquet"
    assert str(parquet_path) == result["trial_returns_path"]
    stored = pd.read_parquet(parquet_path)
    assert list(stored.columns) == [BENCHMARK_COLUMN, "alpha", "beta", "gamma"]
    assert all(str(dtype) == "float32" for dtype in stored.dtypes)
    meta = json.loads((tmp_path / "trial_returns" / "unit-batch.meta.json").read_text())
    assert meta["sha256"] == hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    assert meta["n_trials"] == 3 and meta["n_rows"] == len(index)


def test_complete_family_tests_persistence_arg_pairing(tmp_path) -> None:
    from edgestack.exceptions import DataError
    from edgestack.validation.selection import persist_trial_returns

    rng = np.random.default_rng(19)
    index = pd.bdate_range("2019-01-01", periods=200)
    benchmark = pd.Series(rng.normal(0, 0.01, len(index)), index=index)
    trials = pd.DataFrame(
        {"a": rng.normal(0, 0.01, len(index)), "b": rng.normal(0, 0.01, len(index))},
        index=index,
    )
    with pytest.raises(ValidationError, match="passed together"):
        complete_family_tests(
            benchmark,
            trials,
            complete_trial_ids=("a", "b"),
            reps=100,
            persist_dir=tmp_path,
        )
    with pytest.raises(DataError, match="unsafe path"):
        persist_trial_returns(benchmark, trials, persist_dir=tmp_path, batch_id="../escape")
