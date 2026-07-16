"""Deterministic serialization and hashing for research artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def canonical_value(value: Any) -> Any:
    """Return JSON-compatible data with deterministic ordering and types."""
    if isinstance(value, BaseModel):
        return canonical_value(value.model_dump(mode="json", exclude_none=False))
    if isinstance(value, dict):
        return {
            str(k): canonical_value(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonical_value(v) for v in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value.as_posix())
    if isinstance(value, Enum):
        return canonical_value(value.value)
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        canonical_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
