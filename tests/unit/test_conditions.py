"""Condition-AST evaluation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edgestack.discovery.conditions import evaluate_condition
from edgestack.exceptions import LeakageError, ValidationError
from edgestack.features.binning import QuantileBinner
from edgestack.types import AllOf, Predicate


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rsi": [10.0, 50.0, 90.0, np.nan],
            "flag": [1.0, 0.0, 1.0, 1.0],
        }
    )


def test_value_predicate() -> None:
    mask = evaluate_condition(Predicate(feature="flag", op="==", value=1.0), _df())
    assert mask.tolist() == [True, False, True, True]


def test_nan_never_satisfies() -> None:
    mask = evaluate_condition(Predicate(feature="rsi", op=">", value=0.0), _df())
    assert mask.tolist() == [True, True, True, False]


def test_quantile_predicate_requires_binner() -> None:
    with pytest.raises(LeakageError, match="train-fitted binner"):
        evaluate_condition(Predicate(feature="rsi", op="<", quantile=0.5), _df())


def test_quantile_predicate_uses_train_threshold() -> None:
    train = pd.DataFrame({"rsi": np.arange(101, dtype=float)})
    binner = QuantileBinner(quantiles=(0.1, 0.5, 0.9)).fit(train, ("rsi",))
    mask = evaluate_condition(Predicate(feature="rsi", op="<", quantile=0.1), _df(), binner)
    # Train q10 threshold = 10.0; only the 10.0 row is NOT < 10.0 ... none pass
    # except values strictly below 10. Row values: 10, 50, 90, NaN -> all False.
    assert mask.tolist() == [False, False, False, False]
    mask_hi = evaluate_condition(Predicate(feature="rsi", op=">", quantile=0.5), _df(), binner)
    assert mask_hi.tolist() == [False, False, True, False]


def test_allof_conjunction() -> None:
    cond = AllOf(
        conditions=(
            Predicate(feature="flag", op="==", value=1.0),
            Predicate(feature="rsi", op="<", value=60.0),
        )
    )
    assert evaluate_condition(cond, _df()).tolist() == [True, False, False, False]


def test_unknown_feature_raises() -> None:
    with pytest.raises(ValidationError, match="unknown feature"):
        evaluate_condition(Predicate(feature="ghost", op=">", value=0), _df())
