"""Selection-aware tests that require an explicitly complete searched family."""

from __future__ import annotations

import pandas as pd

from edgestack.exceptions import ValidationError
from edgestack.validation.advanced_tests import spa_test, stepm_superior


def complete_family_tests(
    benchmark_returns: pd.Series,
    trial_returns: pd.DataFrame,
    *,
    complete_trial_ids: tuple[str, ...],
    reps: int = 2_000,
    block_size: int = 20,
    seed: int = 42,
) -> dict:
    expected = tuple(sorted(complete_trial_ids))
    actual = tuple(sorted(str(c) for c in trial_returns.columns))
    if len(expected) != len(set(expected)):
        raise ValidationError("complete trial family contains duplicate ids")
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValidationError(f"searched-family mismatch; missing={missing}, extra={extra}")
    spa = spa_test(
        benchmark_returns,
        trial_returns,
        reps=reps,
        block_size=block_size,
        seed=seed,
    )
    superior = stepm_superior(
        benchmark_returns,
        trial_returns,
        reps=reps,
        block_size=block_size,
        seed=seed,
    )
    return {"spa": spa, "stepm_superior": tuple(sorted(superior))}
