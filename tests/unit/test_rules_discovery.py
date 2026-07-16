"""Unit tests: rule engine mechanics, targets, advanced stats, distillation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edgestack.discovery.distill import distill_policy
from edgestack.discovery.rulefit_engine import (
    discover_rules,
    parse_rule_string,
    prune_vacuous,
    rule_signature,
)
from edgestack.labels.adverse_excursion import adverse_excursion_labels
from edgestack.labels.forward_returns import forward_return_labels
from edgestack.types import Predicate
from edgestack.validation.advanced_tests import (
    newey_west_alpha,
    passes_cgs_hurdle,
    sharpe_difference_test,
    spa_test,
)

# --- rule parsing / canonicalization ---------------------------------------


def test_parse_rule_string_round_trip() -> None:
    preds = parse_rule_string("mom_60 <= 0.08 and natr_14 > 0.02", {"mom_60", "natr_14"})
    assert preds is not None and len(preds) == 2
    assert preds[0].feature == "mom_60" and preds[0].op == "<=" and preds[0].value == 0.08
    assert parse_rule_string("ghost <= 1", {"mom_60"}) is None


def test_signature_merges_nearby_thresholds() -> None:
    deciles = {"vol": np.linspace(0.1, 0.9, 9)}
    a = rule_signature([Predicate(feature="vol", op="<", value=0.22)], deciles, 1.0)
    b = rule_signature([Predicate(feature="vol", op="<=", value=0.24)], deciles, 1.0)
    far = rule_signature([Predicate(feature="vol", op="<", value=0.85)], deciles, 1.0)
    assert a == b  # 0.22 and 0.24 share a decile bucket and direction
    assert a != far  # a genuinely different threshold does not merge
    neg = rule_signature([Predicate(feature="vol", op="<", value=0.22)], deciles, -1.0)
    assert neg != a  # opposite predicted direction is a different rule


def test_prune_vacuous_drops_always_true_conditions() -> None:
    frame = pd.DataFrame({"x": np.linspace(0, 1, 1000), "y": np.linspace(0, 1, 1000)})
    tight = Predicate(feature="x", op="<", value=0.1)
    vacuous = Predicate(feature="y", op=">", value=0.01)  # matches ~99% of rows
    kept = prune_vacuous([tight, vacuous], frame)
    assert kept == [tight]
    # A single predicate is never pruned, however permissive.
    assert prune_vacuous([vacuous], frame) == [vacuous]


def test_discover_rules_finds_planted_signal_and_counts_trials() -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    frame = pd.DataFrame(
        {
            "symbol": rng.choice([f"S{i}" for i in range(40)], n),
            "signal_feat": rng.normal(0, 1, n),
            "noise_feat": rng.normal(0, 1, n),
        }
    )
    y = pd.Series(0.02 * (frame["signal_feat"] > 1.0) + rng.normal(0, 0.01, n))
    rules, trials = discover_rules(frame, y, ("signal_feat", "noise_feat"), seed=1, min_symbols=5)
    assert trials > len(rules) > 0  # trial count includes everything generated
    top = max(rules, key=lambda r: abs(r.coef))
    assert "signal_feat" in top.signature
    assert top.coef > 0


# --- targets -----------------------------------------------------------------


def _mini_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=12)
    opens = np.array([100, 100, 100, 95, 110, 100, 100, 100, 100, 100, 100, 100], dtype=float)
    return pd.DataFrame(
        {
            "symbol": "TST",
            "date": dates,
            "open": opens,
            "high": opens + 2,
            "low": opens - 2,
            "close": opens,
            "volume": 1e6,
            "adj_close": opens,
        }
    )


def test_excess_net_target_arithmetic() -> None:
    panel = _mini_panel()
    labels = forward_return_labels(panel, (5,), benchmark_symbol="TST", roundtrip_cost=0.003)
    # Self-benchmark => excess_ret == 0 => excess_net == -cost everywhere.
    assert np.allclose(labels["excess_net_ret"], -0.003)


def test_adverse_excursion_exact() -> None:
    panel = _mini_panel()
    mae = adverse_excursion_labels(panel, horizon=3, execution_delay=1)
    row = mae.loc[mae["date"] == panel["date"].iloc[1]].iloc[0]
    # Signal at t=1: entry open t=2 (100); window t=2..4 lows = 98, 93, 108.
    assert row["mae"] == pytest.approx(93 / 100 - 1)
    assert row["mfe"] == pytest.approx(112 / 100 - 1)  # highs: 102, 97, 112
    assert row["label_end"] == panel["date"].iloc[5]  # same exit as fwd labels


# --- advanced statistics -------------------------------------------------------


def test_spa_detects_superior_model_and_respects_null() -> None:
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2019-01-01", periods=750)
    bench = pd.Series(rng.normal(0, 0.01, 750), index=idx)
    good = bench + 0.0015 + rng.normal(0, 0.002, 750)
    noise = bench + rng.normal(0, 0.002, 750)
    res = spa_test(bench, pd.DataFrame({"good": good, "noise": noise}), reps=300)
    assert res["p_consistent"] < 0.05
    assert res["best_model"] == "good"
    res_null = spa_test(bench, pd.DataFrame({"noise": noise}), reps=300)
    assert res_null["p_consistent"] > 0.10


def test_sharpe_difference_test_directions() -> None:
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2019-01-01", periods=750)
    strong = pd.Series(rng.normal(0.001, 0.01, 750), index=idx)
    weak = pd.Series(rng.normal(0.0, 0.01, 750), index=idx)
    res = sharpe_difference_test(strong, weak, n_boot=500)
    assert res["delta_sharpe_ann"] > 0.5
    assert res["p_two_sided"] < 0.1
    same = sharpe_difference_test(
        weak, weak.sample(frac=1.0, random_state=1).set_axis(idx), n_boot=500
    )
    assert same["p_two_sided"] > 0.05


def test_newey_west_alpha_recovers_truth() -> None:
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2015-01-01", periods=2000)
    market = pd.Series(rng.normal(0.0004, 0.01, 2000), index=idx)
    strategy = 0.0002 + 0.8 * market + rng.normal(0, 0.002, 2000)
    res = newey_west_alpha(strategy, market)
    assert res["beta"] == pytest.approx(0.8, abs=0.05)
    assert res["alpha_ann"] == pytest.approx(0.0002 * 252, rel=0.5)
    assert res["alpha_t"] > 3


def test_cgs_hurdles() -> None:
    assert passes_cgs_hurdle(3.5, cross_sectional=True)
    assert not passes_cgs_hurdle(3.5, cross_sectional=False)
    assert passes_cgs_hurdle(-4.0, cross_sectional=False)


# --- distillation ---------------------------------------------------------------


def test_distill_policy_learns_and_reports_distiller() -> None:
    rng = np.random.default_rng(5)
    n = 5000
    good = rng.uniform(0, 1, n) < 0.3
    other = rng.uniform(0, 1, n) < 0.5
    indicators = pd.DataFrame({"good_rule|+": good, "noise_rule|+": other})
    y = np.where(good, 0.01, -0.002) + rng.normal(0, 0.004, n)
    policy = distill_policy(indicators, y, seed=0)
    assert policy.distiller in ("gosdt", "corels", "greedy_rule_list", "sklearn_tree_d3")
    tiers = policy.tier(indicators)
    assert set(np.unique(tiers)) <= {0.0, 0.5, 1.0}
    # The policy must allocate more when the informative rule fires.
    assert tiers[good].mean() > tiers[~good].mean()
