"""CSCV/PBO tests: synthetic matrices with known overfitting character."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from edgestack.exceptions import ValidationError
from edgestack.validation.overfitting import pbo_cscv


def _noise(rng: np.random.Generator, t: int, k: int) -> pd.DataFrame:
    return pd.DataFrame(rng.normal(0.0, 0.01, (t, k)), columns=[f"r{i}" for i in range(k)])


def test_pure_noise_pbo_near_half_on_average() -> None:
    # Single-dataset PBO is highly variable under the null (the C(S, S/2)
    # combinations are correlated), so the unbiasedness check averages seeds.
    values = []
    for seed in range(8):
        frame = _noise(np.random.default_rng(seed), 1024, 20)
        result = pbo_cscv(frame, n_partitions=8)
        values.append(result["pbo"])
    assert 0.30 <= float(np.mean(values)) <= 0.70
    assert result["n_combinations"] == math.comb(8, 4)
    assert result["n_trials"] == 20 and result["n_periods"] == 1024


def test_dominant_trial_pbo_low() -> None:
    rng = np.random.default_rng(11)
    frame = _noise(rng, 1024, 19)
    frame["winner"] = rng.normal(0.005, 0.01, 1024)
    result = pbo_cscv(frame, n_partitions=8)
    assert result["pbo"] <= 0.10
    assert result["prob_oos_loss"] <= 0.10


def test_adversarial_trials_pbo_near_one() -> None:
    # One trial per 4-of-8 block subset, positive exactly in its own blocks:
    # every IS combination's winner is its exact-match trial, whose OOS mean
    # is strongly negative — the canonical maximally overfit family.
    import itertools

    rng = np.random.default_rng(13)
    s, t = 8, 800
    block_of_row = np.repeat(np.arange(s), t // s)
    subsets = list(itertools.combinations(range(s), s // 2))
    data = {}
    for j, subset in enumerate(subsets):
        sign = np.where(np.isin(block_of_row, subset), 1.0, -1.0)
        data[f"trial{j}"] = sign * 0.004 + rng.normal(0, 1e-4, t)
    result = pbo_cscv(pd.DataFrame(data), n_partitions=s)
    assert result["pbo"] >= 0.9
    assert result["prob_oos_loss"] >= 0.9


def test_deterministic_and_column_order_invariant() -> None:
    frame = _noise(np.random.default_rng(3), 512, 12)
    a = pbo_cscv(frame, n_partitions=8)
    b = pbo_cscv(frame, n_partitions=8)
    assert a == b
    shuffled = frame[list(frame.columns[::-1])]
    c = pbo_cscv(shuffled, n_partitions=8)
    assert c["pbo"] == a["pbo"]
    assert c["logit_mean"] == pytest.approx(a["logit_mean"])


def test_validation_errors() -> None:
    frame = _noise(np.random.default_rng(1), 200, 4)
    with pytest.raises(ValidationError, match="even"):
        pbo_cscv(frame, n_partitions=7)
    with pytest.raises(ValidationError, match=r"\[4, 20\]"):
        pbo_cscv(frame, n_partitions=22)
    with pytest.raises(ValidationError, match="at least 2 trials"):
        pbo_cscv(frame[["r0"]], n_partitions=8)
    with pytest.raises(ValidationError, match="2 rows per block"):
        pbo_cscv(frame.head(20), n_partitions=16)


@pytest.mark.statistical
def test_large_noise_family_pbo_near_half_on_average() -> None:
    values = []
    for seed in range(10):
        frame = _noise(np.random.default_rng(21 + seed), 1200, 100)
        result = pbo_cscv(frame, n_partitions=8)
        values.append(result["pbo"])
    assert 0.38 <= float(np.mean(values)) <= 0.62
    big = pbo_cscv(_noise(np.random.default_rng(99), 2000, 40), n_partitions=16)
    assert big["n_combinations"] == math.comb(16, 8)
