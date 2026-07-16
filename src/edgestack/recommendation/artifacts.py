"""Content-addressed storage for frozen research artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from edgestack.data.catalog import atomic_write_bytes
from edgestack.exceptions import DataError
from edgestack.recommendation.hashing import canonical_json, file_sha256, stable_hash
from edgestack.recommendation.manifests import FrozenArtifactV2


class FrozenArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def freeze_json(
        self,
        *,
        artifact_type: str,
        artifact_version: str,
        manifest_hash: str,
        data_version: str,
        feature_version: str,
        policy_version: str,
        payload: Any,
        horizon_sessions: int | None = None,
    ) -> FrozenArtifactV2:
        payload_bytes = canonical_json(payload)
        payload_hash = stable_hash(payload)
        metadata_seed = {
            "artifact_type": artifact_type,
            "artifact_version": artifact_version,
            "manifest_hash": manifest_hash,
            "data_version": data_version,
            "feature_version": feature_version,
            "policy_version": policy_version,
            "horizon_sessions": horizon_sessions,
            "payload_hash": payload_hash,
        }
        content_hash = stable_hash(metadata_seed)
        directory = self.root / content_hash
        payload_file = "payload.json"
        metadata = FrozenArtifactV2(
            artifact_type=artifact_type,
            artifact_version=artifact_version,
            manifest_hash=manifest_hash,
            data_version=data_version,
            feature_version=feature_version,
            policy_version=policy_version,
            horizon_sessions=horizon_sessions,
            payload_hash=payload_hash,
            content_hash=content_hash,
            payload_file=payload_file,
        )
        if directory.exists():
            existing = self.load(content_hash)
            if existing != metadata:
                raise DataError(f"artifact hash collision or incompatible artifact: {content_hash}")
            return existing
        directory.mkdir(parents=True, exist_ok=False)
        atomic_write_bytes(directory / payload_file, payload_bytes)
        atomic_write_bytes(
            directory / "artifact.json", metadata.model_dump_json(indent=2).encode("utf-8")
        )
        return metadata

    def load(
        self,
        content_hash: str,
        *,
        manifest_hash: str | None = None,
        data_version: str | None = None,
        feature_version: str | None = None,
        policy_version: str | None = None,
        horizon_sessions: int | None = None,
    ) -> FrozenArtifactV2:
        directory = self.root / content_hash
        metadata_path = directory / "artifact.json"
        if not metadata_path.exists():
            raise DataError(f"unknown frozen artifact: {content_hash}")
        metadata = FrozenArtifactV2.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        payload_path = directory / metadata.payload_file
        if metadata.content_hash != content_hash:
            raise DataError("artifact directory and metadata hash disagree")
        if not payload_path.exists() or file_sha256(payload_path) != metadata.payload_hash:
            # payload_hash is SHA-256 of canonical JSON and therefore equals the file hash.
            raise DataError(f"frozen artifact payload checksum failed: {content_hash}")
        expected = {
            "manifest_hash": manifest_hash,
            "data_version": data_version,
            "feature_version": feature_version,
            "policy_version": policy_version,
            "horizon_sessions": horizon_sessions,
        }
        for field, wanted in expected.items():
            if wanted is not None and getattr(metadata, field) != wanted:
                raise DataError(f"artifact {field} mismatch: expected {wanted!r}")
        return metadata

    def payload(self, content_hash: str, **expected: Any) -> Any:
        metadata = self.load(content_hash, **expected)
        path = self.root / content_hash / metadata.payload_file
        return json.loads(path.read_text(encoding="utf-8"))
