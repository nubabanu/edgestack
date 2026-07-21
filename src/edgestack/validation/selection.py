"""Selection-aware tests that require an explicitly complete searched family."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from edgestack.exceptions import ValidationError
from edgestack.validation.advanced_tests import model_confidence_set, spa_test, stepm_superior
from edgestack.validation.bootstrap import suggested_block_length
from edgestack.validation.overfitting import pbo_cscv

BENCHMARK_COLUMN = "__benchmark__"


def _guarded(fn, *args, **kwargs) -> dict:
    """Diagnostics never veto the canonical SPA/StepM path: degrade to an envelope."""
    try:
        return fn(*args, **kwargs)
    except ValidationError as exc:
        return {"status": "UNAVAILABLE", "reason": str(exc)}


def persist_trial_returns(
    benchmark_returns: pd.Series,
    trial_returns: pd.DataFrame,
    *,
    persist_dir: Path,
    batch_id: str,
) -> dict:
    """Persist the aligned T x k family matrix (plus benchmark) for replay.

    float32 parquet + sha256 sidecar. Canonical statistics are computed on
    the float64 in-memory frame; replayed MCS/PBO p-values can differ in low
    decimals — the sidecar says so.
    """
    from edgestack.data.catalog import atomic_write_bytes, safe_child_path

    aligned = pd.concat(
        [benchmark_returns.rename(BENCHMARK_COLUMN), trial_returns], axis=1
    ).dropna()
    ordered = aligned[[BENCHMARK_COLUMN, *sorted(str(c) for c in trial_returns.columns)]]
    parquet_path = safe_child_path(persist_dir, f"{batch_id}.parquet")
    buffer = io.BytesIO()
    ordered.astype("float32").to_parquet(buffer, index=True, compression="zstd")
    payload = buffer.getvalue()
    atomic_write_bytes(parquet_path, payload)
    sha256 = hashlib.sha256(payload).hexdigest()
    meta = {
        "batch_id": batch_id,
        "created_at": datetime.now(UTC).isoformat(),
        "sha256": sha256,
        "n_rows": len(ordered),
        "n_trials": int(ordered.shape[1] - 1),
        "dtype": "float32",
        "note": (
            "diagnostic replay matrix; canonical statistics were computed on float64 in-memory"
        ),
    }
    atomic_write_bytes(
        safe_child_path(persist_dir, f"{batch_id}.meta.json"),
        json.dumps(meta, indent=2).encode("utf-8"),
    )
    return {"path": str(parquet_path), **meta}


def complete_family_tests(
    benchmark_returns: pd.Series,
    trial_returns: pd.DataFrame,
    *,
    complete_trial_ids: tuple[str, ...],
    reps: int = 2_000,
    block_size: int = 20,
    seed: int = 42,
    mcs_size: float = 0.05,
    pbo_partitions: int = 16,
    persist_dir: Path | None = None,
    batch_id: str | None = None,
) -> dict:
    if (persist_dir is None) != (batch_id is None):
        raise ValidationError("persist_dir and batch_id must be passed together")
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
    mcs = _guarded(
        model_confidence_set,
        trial_returns,
        size=mcs_size,
        reps=min(reps, 1_000),
        block_size=block_size,
        seed=seed,
    )
    pbo = _guarded(pbo_cscv, trial_returns, n_partitions=pbo_partitions)
    pooled = trial_returns.mean(axis=1).dropna().to_numpy()
    diagnostic = suggested_block_length(pooled, fallback=block_size)
    trial_returns_path: str | None = None
    if persist_dir is not None and batch_id is not None:
        artifact = persist_trial_returns(
            benchmark_returns, trial_returns, persist_dir=persist_dir, batch_id=batch_id
        )
        trial_returns_path = artifact["path"]
    return {
        "spa": spa,
        "stepm_superior": tuple(sorted(superior)),
        "mcs": mcs,
        "pbo": pbo,
        "block_length_diagnostic": diagnostic,
        "trial_returns_path": trial_returns_path,
    }
