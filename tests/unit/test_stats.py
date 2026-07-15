"""Statistics-core unit tests against known reference values."""

from __future__ import annotations

import numpy as np
import pytest

from edgestack.discovery.multiple_testing import (
    benjamini_hochberg,
    benjamini_yekutieli,
    fdr_reject,
)
from edgestack.validation.bayes import (
    beta_credible_interval,
    normal_posterior_mean,
    prob_mean_positive,
    prob_win_rate_above,
)
from edgestack.validation.bootstrap import (
    block_bootstrap_ci,
    bootstrap_ci,
    mean_pvalue_bootstrap,
)
from edgestack.validation.metrics import (
    deflated_sharpe_ratio,
    effective_sample_size,
    expected_max_sharpe,
    hit_rate,
    max_drawdown,
    probabilistic_sharpe_ratio,
    profit_factor,
    sharpe_ratio,
    subperiod_sign_consistency,
    var_es,
)


def test_bh_matches_hand_computed_reference() -> None:
    # Classic textbook vector; adjusted = p * m / rank with step-up monotonicity.
    p = np.array([0.01, 0.04, 0.03, 0.005])
    q = benjamini_hochberg(p)
    # sorted p: .005(1), .01(2), .03(3), .04(4) -> raw: .02, .02, .04, .04
    expected = {0.005: 0.02, 0.01: 0.02, 0.03: 0.04, 0.04: 0.04}
    for pi, qi in zip(p, q):
        assert qi == pytest.approx(expected[float(pi)])


def test_by_is_more_conservative_than_bh() -> None:
    p = np.random.default_rng(0).uniform(0, 1, 50)
    assert (benjamini_yekutieli(p) >= benjamini_hochberg(p) - 1e-12).all()


def test_fdr_reject_under_global_null_is_rare() -> None:
    rng = np.random.default_rng(7)
    # 500 noise "strategies": t-test p-values are uniform under the null.
    pvals = rng.uniform(0, 1, 500)
    reject, _ = fdr_reject(pvals, alpha=0.05)
    assert reject.sum() <= 2  # BH controls FDR; under the null nearly nothing passes


def test_bootstrap_ci_covers_true_mean() -> None:
    rng = np.random.default_rng(1)
    sample = rng.normal(0.5, 1.0, 400)
    lo, hi = bootstrap_ci(sample, rng=rng, n_boot=1000)
    assert lo < 0.5 < hi
    assert hi - lo < 0.5


def test_block_bootstrap_wider_under_autocorrelation() -> None:
    rng = np.random.default_rng(2)
    # AR(1) with strong positive autocorrelation.
    n, rho = 500, 0.8
    eps = rng.normal(0, 1, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    iid_lo, iid_hi = bootstrap_ci(x, rng=np.random.default_rng(3), n_boot=800)
    blk_lo, blk_hi = block_bootstrap_ci(
        x, rng=np.random.default_rng(3), block_length=25, n_boot=800
    )
    assert (blk_hi - blk_lo) > (iid_hi - iid_lo)  # honest about dependence


def test_bootstrap_pvalue_direction() -> None:
    rng = np.random.default_rng(4)
    strong = rng.normal(0.5, 1.0, 200)
    noise = rng.normal(0.0, 1.0, 200)
    assert mean_pvalue_bootstrap(strong, rng=np.random.default_rng(5)) < 0.01
    assert mean_pvalue_bootstrap(noise, rng=np.random.default_rng(5)) > 0.05


def test_basic_metrics() -> None:
    r = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    assert hit_rate(r) == pytest.approx(0.6)
    assert profit_factor(r) == pytest.approx(0.06 / 0.03)
    assert sharpe_ratio(np.full(100, 0.001) + np.random.default_rng(0).normal(0, 1e-6, 100)) > 0
    # A -50% then +100% path has a 50% max drawdown.
    assert max_drawdown(np.array([0.1, -0.5, 1.0])) == pytest.approx(-0.5)
    var, es = var_es(np.linspace(-0.1, 0.1, 101), level=0.95)
    assert es <= var < 0


def test_effective_sample_size_shrinks_with_autocorrelation() -> None:
    rng = np.random.default_rng(6)
    iid = rng.normal(0, 1, 1000)
    assert effective_sample_size(iid) > 700
    ar = np.zeros(1000)
    eps = rng.normal(0, 1, 1000)
    for i in range(1, 1000):
        ar[i] = 0.9 * ar[i - 1] + eps[i]
    assert effective_sample_size(ar) < 200


def test_subperiod_consistency() -> None:
    steady = np.tile([0.01, 0.02, -0.005], 40)
    assert subperiod_sign_consistency(steady, 4) == 1.0
    flip = np.concatenate([np.full(50, 0.01), np.full(50, -0.02)])
    assert subperiod_sign_consistency(flip, 4) <= 0.5


def test_psr_and_dsr_behave() -> None:
    rng = np.random.default_rng(8)
    good = rng.normal(0.002, 0.01, 500)  # strong daily edge
    sr = float(good.mean() / good.std(ddof=1))
    assert probabilistic_sharpe_ratio(sr, 0.0, len(good), 0.0, 3.0) > 0.99
    # More trials -> higher expected max noise SR -> lower DSR.
    dsr_few = deflated_sharpe_ratio(good, n_trials=1)
    dsr_many = deflated_sharpe_ratio(good, n_trials=5000)
    assert dsr_few >= dsr_many
    assert expected_max_sharpe(1000, 1 / 500) > expected_max_sharpe(10, 1 / 500)
    # Pure noise searched heavily should not look convincing.
    noise = rng.normal(0.0, 0.01, 500)
    assert deflated_sharpe_ratio(noise, n_trials=1000) < 0.6


def test_bayes_shrinkage_pulls_small_samples_to_neutral() -> None:
    # 7 wins / 3 losses looks like 70%, but the posterior is much humbler.
    assert prob_win_rate_above(7, 3, 0.5) < 0.9
    lo, hi = beta_credible_interval(7, 3)
    assert lo < 0.5 < hi
    # Small positive sample: shrunk mean far below the raw mean.
    small = np.full(10, 0.01)
    post_mean, _ = normal_posterior_mean(small, prior_mean=0.0, prior_pseudo_n=30)
    assert post_mean == pytest.approx(0.01 * 10 / 40)
    # Large consistent sample overcomes the prior.
    rng = np.random.default_rng(9)
    big = rng.normal(0.01, 0.005, 2000)
    assert prob_mean_positive(big) > 0.999
