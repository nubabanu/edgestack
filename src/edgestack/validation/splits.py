"""Purged walk-forward splitting for overlapping labels.

Purging is interval-based, not offset-based: a training row is dropped when
its outcome interval ``[date, label_end]`` overlaps the test window. This is
exact for variable-length labels (triple-barrier exits) where a fixed offset
would under-purge. An additional embargo drops training rows whose signal
date falls within ``embargo_sessions`` after a test window — relevant when a
fold's training data lies after an earlier fold's test window (rolling mode)
or when callers reuse folds for cross-fitting.

Two facades over one implementation:
- :meth:`split_frame` — rich ``Fold`` objects for event studies and edge
  validation (row indices into an arbitrary long frame);
- :meth:`split` — sklearn-compatible ``(train_idx, test_idx)`` pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.exceptions import ValidationError


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purged_count: int


class PurgedWalkForwardSplitter:
    def __init__(
        self,
        *,
        n_folds: int,
        test_sessions: int,
        train_min_sessions: int,
        embargo_sessions: int,
        expanding: bool = True,
    ) -> None:
        if n_folds < 1:
            raise ValidationError("n_folds must be >= 1")
        self.n_folds = n_folds
        self.test_sessions = test_sessions
        self.train_min_sessions = train_min_sessions
        self.embargo_sessions = embargo_sessions
        self.expanding = expanding

    @classmethod
    def from_config(cls, cfg: EdgeStackConfig) -> PurgedWalkForwardSplitter:
        v = cfg.validation
        return cls(
            n_folds=v.n_folds,
            test_sessions=v.test_sessions,
            train_min_sessions=v.train_min_sessions,
            embargo_sessions=v.embargo_sessions,
            expanding=v.method == "expanding_walk_forward",
        )

    # -- core ---------------------------------------------------------------

    def split_frame(self, dates: pd.Series, label_end: pd.Series) -> list[Fold]:
        """Split rows of a long frame by signal date with interval purging."""
        if len(dates) != len(label_end):
            raise ValidationError("dates and label_end must align")
        date_values = pd.to_datetime(dates).to_numpy()
        end_values = pd.to_datetime(label_end).to_numpy()
        if (end_values < date_values).any():
            raise ValidationError("label_end before signal date — corrupted labels")

        sessions = np.unique(date_values)
        needed = self.train_min_sessions + self.n_folds * self.test_sessions
        if len(sessions) < needed:
            raise ValidationError(
                f"not enough sessions for validation: have {len(sessions)}, "
                f"need {needed} (train_min {self.train_min_sessions} + "
                f"{self.n_folds}x{self.test_sessions} test)"
            )

        folds: list[Fold] = []
        first_test_pos = len(sessions) - self.n_folds * self.test_sessions
        for k in range(self.n_folds):
            test_lo = first_test_pos + k * self.test_sessions
            test_hi = test_lo + self.test_sessions - 1
            test_start, test_end = sessions[test_lo], sessions[test_hi]

            if self.expanding:
                train_lo_pos = 0
            else:
                train_lo_pos = max(0, test_lo - self.train_min_sessions)
            train_start = sessions[train_lo_pos]
            train_end = sessions[test_lo - 1]

            in_train_window = (date_values >= train_start) & (date_values <= train_end)
            # Interval purge: outcome must be fully resolved before the test
            # window opens.
            overlaps_test = end_values >= test_start
            # Embargo: signal dates too close after ANY earlier test window.
            embargoed = np.zeros(len(date_values), dtype=bool)
            for prev in folds:
                prev_end_pos = int(np.searchsorted(sessions, prev.test_end.to_numpy()))
                emb_hi_pos = min(len(sessions) - 1, prev_end_pos + self.embargo_sessions)
                embargoed |= (date_values > prev.test_end.to_numpy()) & (
                    date_values <= sessions[emb_hi_pos]
                )

            train_mask = in_train_window & ~overlaps_test & ~embargoed
            test_mask = (date_values >= test_start) & (date_values <= test_end)
            purged = int(in_train_window.sum() - train_mask.sum())

            folds.append(
                Fold(
                    fold_id=k,
                    train_idx=np.flatnonzero(train_mask),
                    test_idx=np.flatnonzero(test_mask),
                    train_start=pd.Timestamp(train_start),
                    train_end=pd.Timestamp(train_end),
                    test_start=pd.Timestamp(test_start),
                    test_end=pd.Timestamp(test_end),
                    purged_count=purged,
                )
            )
        return folds

    # -- sklearn facade -------------------------------------------------------

    def split(self, X, y=None, groups=None):
        """Yield (train_idx, test_idx); X must carry `date` and `label_end` columns."""
        if not isinstance(X, pd.DataFrame) or "date" not in X or "label_end" not in X:
            raise ValidationError("sklearn facade requires a DataFrame with `date` and `label_end`")
        for fold in self.split_frame(X["date"], X["label_end"]):
            yield fold.train_idx, fold.test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_folds
