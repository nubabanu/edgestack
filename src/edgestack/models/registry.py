"""Model artifact persistence with integrity checks.

sklearn estimators serialize via pickle, which can execute code on load.
Mitigations (spec section 41): artifacts are only ever loaded from the
project's own artifacts directory, every artifact carries a SHA-256 checksum
plus a JSON metadata sidecar, and loading verifies both BEFORE unpickling.
A tampered or foreign file is refused.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path

from edgestack.data.catalog import atomic_write_bytes, safe_child_path
from edgestack.exceptions import DataError
from edgestack.models.base import TrainedModel

_META_KEYS = ("name", "horizon", "side", "featureset_id", "config_hash", "seed")


def _artifact_name(model: TrainedModel) -> str:
    return f"model_h{model.horizon}_{model.side.lower()}"


def save_models(models_dir: Path, models: list[TrainedModel]) -> list[Path]:
    """Persist each model as .pkl + .sha256 + .meta.json (atomic writes)."""
    paths = []
    for model in models:
        base = _artifact_name(model)
        payload = pickle.dumps(model)
        digest = hashlib.sha256(payload).hexdigest()
        meta = {k: v for k, v in asdict(model).items()
                if k in _META_KEYS or k in ("metrics", "reliability_bins")}
        pkl_path = safe_child_path(models_dir, f"{base}.pkl")
        atomic_write_bytes(pkl_path, payload)
        atomic_write_bytes(safe_child_path(models_dir, f"{base}.sha256"), digest.encode())
        atomic_write_bytes(
            safe_child_path(models_dir, f"{base}.meta.json"),
            json.dumps(meta, indent=2, default=str).encode(),
        )
        paths.append(pkl_path)
    return paths


def load_models(models_dir: Path, *, expected_config_hash: str | None = None,
                ) -> list[TrainedModel]:
    """Load all model artifacts, verifying checksum and metadata first."""
    if not models_dir.exists():
        raise DataError(f"no model artifacts at {models_dir}; run `edgestack models train`")
    models = []
    for pkl_path in sorted(models_dir.glob("model_*.pkl")):
        payload = pkl_path.read_bytes()
        digest_path = pkl_path.with_suffix(".sha256")
        meta_path = pkl_path.with_suffix(".meta.json")
        if not digest_path.exists() or not meta_path.exists():
            raise DataError(f"{pkl_path.name}: missing checksum or metadata sidecar")
        expected = digest_path.read_text().strip()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise DataError(
                f"{pkl_path.name}: checksum mismatch — refusing to unpickle a "
                "tampered or foreign artifact"
            )
        meta = json.loads(meta_path.read_text())
        if any(k not in meta for k in _META_KEYS):
            raise DataError(f"{pkl_path.name}: metadata sidecar missing required keys")
        model: TrainedModel = pickle.loads(payload)
        if expected_config_hash and model.config_hash != expected_config_hash:
            # Config drift is a warning-level mismatch, not corruption; callers
            # decide. We refuse only when integrity is in question.
            model.metrics["config_hash_mismatch"] = 1.0
        models.append(model)
    if not models:
        raise DataError(f"no model artifacts found in {models_dir}")
    return models
