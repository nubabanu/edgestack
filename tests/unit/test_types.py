"""Domain model tests: rule AST serialization and invariants."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from edgestack.types import (
    AllOf,
    Condition,
    Predicate,
    SignalCandidate,
    condition_depth,
    condition_features,
)

_CONDITION = TypeAdapter(Condition)


def _sample_condition() -> AllOf:
    return AllOf(
        conditions=(
            Predicate(feature="rsi_14_pctile", op="<", quantile=0.1),
            Predicate(feature="trend_sma200_above", op="==", value=True),
            Predicate(feature="rel_volume_20", op=">", quantile=0.9),
        )
    )


def test_condition_json_round_trip() -> None:
    cond = _sample_condition()
    payload = cond.model_dump_json()
    back = _CONDITION.validate_json(payload)
    assert back == cond


def test_condition_depth_and_features() -> None:
    cond = _sample_condition()
    assert condition_depth(cond) == 3
    assert condition_features(cond) == (
        "rsi_14_pctile",
        "trend_sma200_above",
        "rel_volume_20",
    )


def test_condition_describe_is_readable() -> None:
    text = _sample_condition().describe()
    assert "rsi_14_pctile < q0.1(train)" in text
    assert " AND " in text


def test_conviction_score_bounds_enforced() -> None:
    with pytest.raises(PydanticValidationError):
        SignalCandidate.model_validate({"conviction_score": 101})  # plus missing fields


def test_predicate_requires_op() -> None:
    with pytest.raises(PydanticValidationError):
        Predicate.model_validate({"feature": "x", "op": "~", "value": 1})
