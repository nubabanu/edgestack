"""Purged walk-forward splitter tests with hand-drawn overlap fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edgestack.exceptions import ValidationError
from edgestack.validation.splits import PurgedWalkForwardSplitter


def _frame(n_sessions: int, horizon: int) -> pd.DataFrame:
    """One row per session; label resolves `horizon` sessions later."""
    dates = pd.bdate_range("2015-01-01", periods=n_sessions)
    label_end_pos = np.minimum(np.arange(n_sessions) + horizon, n_sessions - 1)
    return pd.DataFrame({"date": dates, "label_end": dates[label_end_pos]})


def test_purge_removes_overlapping_train_rows() -> None:
    df = _frame(300, horizon=5)
    splitter = PurgedWalkForwardSplitter(
        n_folds=2, test_sessions=50, train_min_sessions=100, embargo_sessions=5
    )
    folds = splitter.split_frame(df["date"], df["label_end"])
    assert len(folds) == 2

    f0 = folds[0]
    # Test window: sessions 200..249. Train window 0..199, but rows 195..199
    # have label_end >= session 200 and must be purged.
    assert f0.purged_count == 5
    assert f0.train_idx.max() == 194
    assert list(f0.test_idx) == list(range(200, 250))
    # No train row's outcome interval may overlap the test window.
    train_end = df["label_end"].iloc[f0.train_idx]
    assert (train_end < f0.test_start).all()


def test_embargo_bites_when_train_follows_earlier_test() -> None:
    df = _frame(350, horizon=5)
    splitter = PurgedWalkForwardSplitter(
        n_folds=3, test_sessions=50, train_min_sessions=100, embargo_sessions=5
    )
    folds = splitter.split_frame(df["date"], df["label_end"])
    f2 = folds[2]
    # Fold 2 trains on 0..299, which includes fold 0's test window (200..249).
    # The 5 sessions right after fold 0's test end (250..254) are embargoed,
    # plus rows 295..299 are purged for overlap: 10 rows removed in total.
    assert f2.purged_count == 10
    train_positions = set(f2.train_idx)
    assert not train_positions.intersection(range(250, 255))
    assert not train_positions.intersection(range(295, 300))
    assert 255 in train_positions


def test_variable_length_labels_are_purged_by_interval() -> None:
    # Two rows share a signal date but one resolves much later: only the
    # long-lived one must be purged.
    dates = pd.bdate_range("2015-01-01", periods=200)
    df = pd.DataFrame(
        {
            "date": [*dates[:150], dates[100]],
            "label_end": [*dates[np.minimum(np.arange(150) + 2, 199)], dates[160]],
        }
    )
    splitter = PurgedWalkForwardSplitter(
        n_folds=1, test_sessions=30, train_min_sessions=100, embargo_sessions=0
    )
    fold = splitter.split_frame(df["date"], df["label_end"])[0]
    # Test window: sessions 120..149. The duplicate row at position 150
    # (signal date = session 100, label_end = session 160) overlaps and is gone.
    assert 150 not in set(fold.train_idx)
    # Its short-lived twin at position 100 (resolves at 102) survives.
    assert 100 in set(fold.train_idx)


def test_insufficient_history_raises() -> None:
    df = _frame(120, horizon=5)
    splitter = PurgedWalkForwardSplitter(
        n_folds=3, test_sessions=50, train_min_sessions=100, embargo_sessions=5
    )
    with pytest.raises(ValidationError, match="not enough sessions"):
        splitter.split_frame(df["date"], df["label_end"])


def test_corrupted_labels_rejected() -> None:
    dates = pd.bdate_range("2015-01-01", periods=10)
    df = pd.DataFrame({"date": dates, "label_end": dates - pd.Timedelta(days=5)})
    splitter = PurgedWalkForwardSplitter(
        n_folds=1, test_sessions=2, train_min_sessions=2, embargo_sessions=0
    )
    with pytest.raises(ValidationError, match="corrupted"):
        splitter.split_frame(df["date"], df["label_end"])


def test_sklearn_facade_yields_index_pairs() -> None:
    df = _frame(300, horizon=5)
    splitter = PurgedWalkForwardSplitter(
        n_folds=2, test_sessions=50, train_min_sessions=100, embargo_sessions=5
    )
    pairs = list(splitter.split(df))
    assert len(pairs) == splitter.get_n_splits() == 2
    train_idx, test_idx = pairs[0]
    assert len(np.intersect1d(train_idx, test_idx)) == 0
