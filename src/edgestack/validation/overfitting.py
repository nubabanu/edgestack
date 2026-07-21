"""Probability of backtest overfitting via CSCV.

Bailey, Borwein, Lopez de Prado & Zhu (2017): split the sample into S
contiguous blocks, enumerate every way to pick S/2 blocks as in-sample (IS),
find the IS-best trial, and ask where it ranks out-of-sample (OOS). PBO is
the fraction of combinations where the IS winner lands in the bottom half
OOS (logit <= 0). CSCV respects the time ordering inside blocks, is
model-free and non-parametric, and — because the enumeration is exhaustive —
fully deterministic: no RNG, replicable to the bit.

Implementation: per-block sufficient statistics (count / sum / sum-of-
squares), then chunked matrix products over the C(S, S/2) combination
matrix, so no combination ever materializes its raw return rows.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from edgestack.exceptions import ValidationError

_CHUNK = 2048
_MAX_PARTITIONS = 20  # C(20,10)=184_756 combinations — the enumeration cap


def _sharpe_from_stats(n: np.ndarray, total: np.ndarray, sumsq: np.ndarray) -> np.ndarray:
    """Vectorized per-period Sharpe from sufficient statistics (ddof=1)."""
    mean = total / n
    var = np.maximum(sumsq - n * mean**2, 0.0) / (n - 1.0)
    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mean / sd, 0.0)
    return sr


def pbo_cscv(trial_returns: pd.DataFrame, *, n_partitions: int = 16) -> dict:
    """Probability of backtest overfitting for a T x k trial-return matrix.

    ``n_partitions`` (S) must be even, 4..20. Blocks should hold >= 10 rows
    each for the Sharpe estimates to mean anything; < 2 rows per block is a
    hard error. Ties: the IS-best is the lowest column index (deterministic),
    the OOS rank uses mid-ranks, so the PBO value is column-order invariant.
    """
    if n_partitions % 2 or not 4 <= n_partitions <= _MAX_PARTITIONS:
        raise ValidationError(f"n_partitions must be even and in [4, {_MAX_PARTITIONS}]")
    cleaned = trial_returns.dropna()
    n_periods, n_trials = cleaned.shape
    if n_trials < 2:
        raise ValidationError("PBO needs at least 2 trials")
    if n_periods < 2 * n_partitions:
        raise ValidationError(
            f"need >= 2 rows per block: {n_periods} rows across {n_partitions} blocks"
        )

    x = cleaned.to_numpy(dtype=float)
    blocks = np.array_split(np.arange(n_periods), n_partitions)
    counts = np.array([len(b) for b in blocks], dtype=float)  # (S,)
    sums = np.stack([x[b].sum(axis=0) for b in blocks])  # (S, k)
    sumsqs = np.stack([(x[b] ** 2).sum(axis=0) for b in blocks])  # (S, k)
    total_count = counts.sum()
    total_sum = sums.sum(axis=0)
    total_sumsq = sumsqs.sum(axis=0)

    combos = np.array(
        list(itertools.combinations(range(n_partitions), n_partitions // 2)), dtype=np.intp
    )
    n_combos = len(combos)
    selector = np.zeros((n_combos, n_partitions))
    selector[np.arange(n_combos)[:, None], combos] = 1.0

    logits = np.empty(n_combos)
    oos_sr_best = np.empty(n_combos)
    for lo in range(0, n_combos, _CHUNK):
        sel = selector[lo : lo + _CHUNK]  # (m, S)
        is_n = (sel @ counts)[:, None]  # (m, 1)
        is_sum = sel @ sums  # (m, k)
        is_sq = sel @ sumsqs
        oos_n = total_count - is_n
        oos_sum = total_sum[None, :] - is_sum
        oos_sq = total_sumsq[None, :] - is_sq

        is_sr = _sharpe_from_stats(is_n, is_sum, is_sq)
        oos_sr = _sharpe_from_stats(oos_n, oos_sum, oos_sq)

        best = np.argmax(is_sr, axis=1)  # ties -> lowest index
        rows = np.arange(len(sel))
        best_oos = oos_sr[rows, best]
        # Mid-rank of the IS winner among all OOS Sharpes of the combination.
        less = (oos_sr < best_oos[:, None]).sum(axis=1)
        equal = (oos_sr == best_oos[:, None]).sum(axis=1)
        rank = less + (equal + 1.0) / 2.0
        omega = rank / (n_trials + 1.0)
        logits[lo : lo + len(sel)] = np.log(omega / (1.0 - omega))
        oos_sr_best[lo : lo + len(sel)] = best_oos

    return {
        "pbo": float((logits <= 0).mean()),
        "n_partitions": n_partitions,
        "n_combinations": int(n_combos),
        "n_trials": int(n_trials),
        "n_periods": int(n_periods),
        "logit_mean": float(logits.mean()),
        "logit_median": float(np.median(logits)),
        "prob_oos_loss": float((oos_sr_best <= 0).mean()),
    }


def _n_combinations(n_partitions: int) -> int:
    """Exposed for tests: C(S, S/2)."""
    return math.comb(n_partitions, n_partitions // 2)
