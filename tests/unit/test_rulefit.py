"""In-house RuleFit engine and distillation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgestack.discovery.distill import distill_policy
from edgestack.discovery.rulefit_engine import (
    discover_rules,
    prune_vacuous,
    rule_signature,
)
from edgestack.types import Predicate


def _frame(n: int = 30_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({
        "symbol": rng.choice([f"S{i}" for i in range(40)], n),
        "alpha_feat": rng.normal(0, 1, n),
        "beta_feat": rng.normal(0, 1, n),
    })
    # Planted rule: alpha_feat > 1 adds +1.5% to the target.
    y = pd.Series(0.015 * (f["alpha_feat"] > 1.0) + rng.normal(0, 0.02, n))
    return f, y


def test_discover_rules_finds_planted_threshold() -> None:
    f, y = _frame()
    rules, total = discover_rules(f, y, ("alpha_feat", "beta_feat"),
                                  seed=1, min_symbols=10)
    assert total > len(rules) > 0  # honest trial count exceeds survivors
    top_positive = [r for r in rules if r.coef > 0][:5]
    assert any(
        "alpha_feat" in r.signature and ">" in r.signature for r in top_positive
    ), [r.raw for r in top_positive]


def test_signature_merges_nearby_thresholds() -> None:
    deciles = {"vol": np.linspace(0.1, 0.9, 9)}
    a = [Predicate(feature="vol", op="<", value=0.22)]
    b = [Predicate(feature="vol", op="<", value=0.24)]
    far = [Predicate(feature="vol", op="<", value=0.62)]
    assert rule_signature(a, deciles, 1.0) == rule_signature(b, deciles, 1.0)
    assert rule_signature(a, deciles, 1.0) != rule_signature(far, deciles, 1.0)
    # Opposite coefficient sign is a different structural rule.
    assert rule_signature(a, deciles, 1.0) != rule_signature(a, deciles, -1.0)


def test_prune_vacuous_drops_permissive_terms() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"x": rng.normal(0, 1, 5000), "y": rng.normal(0, 1, 5000)})
    tight = Predicate(feature="x", op=">", value=1.5)      # ~7% of rows
    vacuous = Predicate(feature="y", op=">", value=-3.0)   # ~99.9% of rows
    kept = prune_vacuous([tight, vacuous], frame)
    assert kept == [tight]
    # A single predicate is never pruned away entirely.
    assert prune_vacuous([vacuous], frame) == [vacuous]


def test_distill_policy_tiers_are_bounded_and_sane() -> None:
    rng = np.random.default_rng(2)
    n = 8000
    good = rng.uniform(0, 1, n) < 0.3
    indicators = pd.DataFrame({
        "rule_good|+": good,
        "rule_noise|+": rng.uniform(0, 1, n) < 0.5,
    })
    # Positive excess mostly when the good rule fires.
    y = np.where(good, rng.normal(0.01, 0.01, n), rng.normal(-0.002, 0.01, n))
    policy = distill_policy(indicators, y, seed=3)
    assert policy.distiller in ("gosdt", "corels", "greedy_rule_list", "sklearn_tree_d3")
    tiers = policy.tier(indicators)
    assert set(np.unique(tiers)) <= {0.0, 0.5, 1.0}
    # The good-rule rows must receive at least as much size on average.
    assert tiers[good.to_numpy() if hasattr(good, "to_numpy") else good].mean() \
        >= tiers[~good].mean()
