"""Advanced-inference tests on synthetic cases with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edgestack.validation.advanced_tests import (
    CGS_HURDLE_CROSS_SECTION,
    newey_west_alpha,
    passes_cgs_hurdle,
    sharpe_difference_test,
    spa_test,
    stepm_superior,
)

N = 1000


def _series(rng, mean=0.0, sd=0.01, n=N):
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(rng.normal(mean, sd, n), index=idx)


def test_spa_detects_genuine_outperformance() -> None:
    rng = np.random.default_rng(0)
    bench = _series(rng, 0.0)
    models = pd.DataFrame(
        {
            "noise1": _series(rng, 0.0).to_numpy(),
            "good": _series(rng, 0.002).to_numpy(),  # real daily edge
            "noise2": _series(rng, 0.0).to_numpy(),
        },
        index=bench.index,
    )
    result = spa_test(bench, models, reps=300, seed=1)
    assert result["p_consistent"] < 0.05
    assert result["best_model"] == "good"


def test_spa_does_not_reject_on_pure_noise() -> None:
    rng = np.random.default_rng(2)
    bench = _series(rng, 0.0)
    models = pd.DataFrame(
        {f"m{i}": _series(rng, 0.0).to_numpy() for i in range(6)}, index=bench.index
    )
    result = spa_test(bench, models, reps=300, seed=3)
    assert result["p_consistent"] > 0.05


def test_stepm_identifies_only_the_superior_model() -> None:
    rng = np.random.default_rng(4)
    bench = _series(rng, 0.0)
    models = pd.DataFrame(
        {
            "flat": _series(rng, 0.0).to_numpy(),
            "strong": _series(rng, 0.003).to_numpy(),
        },
        index=bench.index,
    )
    superior = stepm_superior(bench, models, reps=300, seed=5)
    assert "strong" in superior
    assert "flat" not in superior


def test_sharpe_difference_test_direction_and_null() -> None:
    rng = np.random.default_rng(6)
    a = _series(rng, 0.002, 0.01)
    b = _series(rng, 0.0, 0.01)
    out = sharpe_difference_test(a, b, n_boot=500, seed=7)
    assert out["delta_sharpe_ann"] > 1.0
    assert out["p_two_sided"] < 0.05
    same = sharpe_difference_test(b, _series(rng, 0.0, 0.01), n_boot=500, seed=8)
    assert same["p_two_sided"] > 0.05


def test_newey_west_recovers_alpha_and_beta() -> None:
    rng = np.random.default_rng(9)
    market = _series(rng, 0.0004, 0.01, n=2000)
    alpha_daily, beta = 0.0005, 0.8
    strat = alpha_daily + beta * market + rng.normal(0, 0.004, 2000)
    out = newey_west_alpha(pd.Series(strat, index=market.index), market)
    assert out["beta"] == pytest.approx(beta, abs=0.05)
    assert out["alpha_ann"] == pytest.approx(alpha_daily * 252, rel=0.35)
    assert out["alpha_t"] > 3


def test_cgs_hurdle() -> None:
    assert passes_cgs_hurdle(CGS_HURDLE_CROSS_SECTION + 0.01)
    assert not passes_cgs_hurdle(2.5)
    assert not passes_cgs_hurdle(3.5, cross_sectional=False)  # ts hurdle is 3.8


def test_mcs_includes_strong_excludes_bad() -> None:
    from edgestack.validation.advanced_tests import model_confidence_set

    rng = np.random.default_rng(10)
    models = pd.DataFrame(
        {
            "strong": _series(rng, 0.003).to_numpy(),
            "bad": _series(rng, -0.003).to_numpy(),
            **{f"noise{i}": _series(rng, 0.0).to_numpy() for i in range(4)},
        },
        index=_series(rng).index,
    )
    result = model_confidence_set(models, reps=300, seed=1)
    assert "strong" in result["included"]
    assert "bad" in result["excluded"]
    assert set(result["pvalues"]) == set(models.columns)
    assert result["n_models"] == 6 and result["n_days"] == N


def test_mcs_keeps_indistinguishable_noise_together() -> None:
    from edgestack.validation.advanced_tests import model_confidence_set

    rng = np.random.default_rng(20)
    models = pd.DataFrame(
        {f"noise{i}": _series(rng, 0.0).to_numpy() for i in range(6)},
        index=_series(rng).index,
    )
    result = model_confidence_set(models, reps=300, seed=2)
    assert len(result["included"]) >= 5  # noise models are not distinguishable


def test_mcs_guards() -> None:
    from edgestack.exceptions import ValidationError
    from edgestack.validation.advanced_tests import model_confidence_set

    rng = np.random.default_rng(30)
    one_col = pd.DataFrame({"only": _series(rng).to_numpy()})
    with pytest.raises(ValidationError, match="at least 2 models"):
        model_confidence_set(one_col)
    short = pd.DataFrame({"a": np.zeros(30), "b": np.ones(30)})
    with pytest.raises(ValidationError, match=">=60"):
        model_confidence_set(short)
    degenerate = pd.DataFrame({"a": np.zeros(100), "b": np.ones(100)})
    with pytest.raises(ValidationError, match="pairwise loss differential"):
        model_confidence_set(degenerate)
