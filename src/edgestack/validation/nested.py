"""Annual nested walk-forward folds with training-only decision freezing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from edgestack.exceptions import ValidationError
from edgestack.recommendation.hashing import stable_hash
from edgestack.recommendation.manifests import OuterFoldV2


@dataclass(frozen=True)
class AnnualOuterFold:
    definition: OuterFoldV2
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class FrozenOuterDecision[T]:
    fold_id: str
    decision: T
    decision_hash: str


def annual_outer_folds(
    dates: pd.Series,
    label_end: pd.Series | None = None,
    *,
    min_train_sessions: int = 756,
    min_test_sessions: int = 200,
) -> tuple[AnnualOuterFold, ...]:
    """Create expanding annual outer folds and purge unresolved train labels."""
    d = pd.to_datetime(dates).reset_index(drop=True)
    ends = pd.to_datetime(label_end).reset_index(drop=True) if label_end is not None else d
    if len(d) != len(ends):
        raise ValidationError("dates and label_end must align")
    if (ends < d).any():
        raise ValidationError("label_end cannot precede the signal date")
    sessions = pd.DatetimeIndex(sorted(d.unique()))
    if len(sessions) < min_train_sessions + min_test_sessions:
        raise ValidationError("not enough sessions for annual nested walk-forward")

    folds: list[AnnualOuterFold] = []
    years = sorted({int(x.year) for x in sessions})
    for year in years:
        test_sessions = sessions[sessions.year == year]
        if test_sessions.empty:
            continue
        test_start = test_sessions[0]
        train_session_values = sessions[sessions < test_start]
        if len(train_session_values) < min_train_sessions:
            continue
        test_mask = (d >= test_sessions[0]) & (d <= test_sessions[-1])
        train_mask = (d < test_start) & (ends < test_start)
        valid = len(test_sessions) >= min_test_sessions and bool(test_mask.any())
        reason = None if valid else f"test year has {len(test_sessions)} sessions"
        definition = OuterFoldV2(
            fold_id=str(year),
            train_start=train_session_values[0].date(),
            train_end=train_session_values[-1].date(),
            test_start=test_sessions[0].date(),
            test_end=test_sessions[-1].date(),
            train_sessions=len(train_session_values),
            test_sessions=len(test_sessions),
            valid=valid,
            invalid_reason=reason,
        )
        folds.append(
            AnnualOuterFold(
                definition=definition,
                train_idx=np.flatnonzero(train_mask.to_numpy()),
                test_idx=np.flatnonzero(test_mask.to_numpy()),
            )
        )
    return tuple(folds)


def freeze_outer_decisions[T](
    frame: pd.DataFrame,
    folds: tuple[AnnualOuterFold, ...],
    selector: Callable[[pd.DataFrame, OuterFoldV2], T],
) -> tuple[FrozenOuterDecision[T], ...]:
    """Call ``selector`` with outer-training rows only and hash each decision."""
    decisions = []
    for fold in folds:
        if not fold.definition.valid:
            continue
        train = frame.iloc[fold.train_idx].copy(deep=True)
        decision = selector(train, fold.definition)
        decisions.append(
            FrozenOuterDecision(
                fold_id=fold.definition.fold_id,
                decision=decision,
                decision_hash=stable_hash(decision),
            )
        )
    return tuple(decisions)
