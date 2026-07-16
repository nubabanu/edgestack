"""Fail-fast, content-addressed, atomic recommendation publication."""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from edgestack.data.catalog import atomic_write_bytes
from edgestack.exceptions import DataError
from edgestack.recommendation.compatibility import (
    board_projection,
    master_projection,
    picks_projection,
    signals_projection,
)
from edgestack.recommendation.hashing import canonical_json, file_sha256, stable_hash
from edgestack.recommendation.instrument_schemas import FrozenTimingArtifactV2, NewsEvidenceV2
from edgestack.recommendation.manifests import PublicationRecordV2
from edgestack.recommendation.risk import RiskInputsV2
from edgestack.recommendation.schemas import CanonicalRecommendationBundleV2
from edgestack.recommendation.service import (
    CurrentBundlePointerV2,
    risk_inputs_json,
)


class AtomicRecommendationPublisher:
    """Publish a complete run and expose it with one pointer replacement."""

    def __init__(self, artifacts_dir: Path):
        self.root = artifacts_dir / "recommendations"
        self.runs = self.root / "runs"
        self.staging = self.root / ".staging"

    def publish(
        self,
        *,
        bundle: CanonicalRecommendationBundleV2,
        risk_inputs: RiskInputsV2,
        paper_state: dict[str, Any],
        monitoring: dict[str, Any],
        timing_artifacts: tuple[FrozenTimingArtifactV2, ...] = (),
        news_evidence: tuple[NewsEvidenceV2, ...] = (),
        before_pointer_swap: Callable[[Path], None] | None = None,
    ) -> PublicationRecordV2:
        if risk_inputs.session != bundle.session:
            raise DataError("publication risk inputs and bundle sessions differ")
        expected_versions = (bundle.data_version, bundle.artifact_version, bundle.policy_version)
        seen_timing: set[tuple[str, object]] = set()
        for artifact in timing_artifacts:
            versions = (artifact.data_version, artifact.artifact_version, artifact.policy_version)
            if versions != expected_versions:
                raise DataError(
                    f"timing artifact version mismatch for {artifact.symbol}/{artifact.horizon}"
                )
            key = (artifact.symbol, artifact.horizon)
            if key in seen_timing:
                raise DataError(
                    f"duplicate timing artifact for {artifact.symbol}/{artifact.horizon}"
                )
            seen_timing.add(key)
        self.staging.mkdir(parents=True, exist_ok=True)
        temporary = self.staging / uuid.uuid4().hex
        temporary.mkdir()
        try:
            inputs_payload = risk_inputs_json(risk_inputs)
            version_set = {
                "schema_version": 2,
                "session": bundle.session.isoformat(),
                "as_of": bundle.as_of.isoformat(),
                "execution_at": bundle.execution_at.isoformat(),
                "data_version": bundle.data_version,
                "artifact_version": bundle.artifact_version,
                "policy_version": bundle.policy_version,
                "bundle_hash": bundle.bundle_hash,
                "risk_inputs_hash": stable_hash(inputs_payload),
            }
            documents: dict[str, Any] = {
                "recommendation.json": bundle.model_dump(mode="json"),
                "risk_inputs.json": inputs_payload,
                "paper_state.json": _versioned_payload(version_set, paper_state),
                "monitoring.json": _versioned_payload(version_set, monitoring),
                "instrument_timing.json": _versioned_payload(
                    version_set,
                    {"artifacts": [item.model_dump(mode="json") for item in timing_artifacts]},
                ),
                "news_context.json": _versioned_payload(
                    version_set,
                    {"items": [item.model_dump(mode="json") for item in news_evidence]},
                ),
                "compatibility/board.json": board_projection(bundle),
                "compatibility/picks.json": picks_projection(bundle),
                "compatibility/master.json": master_projection(bundle),
                "compatibility/signals.json": signals_projection(bundle),
                "versions.json": version_set,
            }
            for relative, payload in documents.items():
                _write_json(temporary / relative, payload)
            base_hashes = _hash_files(temporary)
            run_id = stable_hash(
                {
                    "bundle_hash": bundle.bundle_hash,
                    "risk_inputs_hash": stable_hash(inputs_payload),
                    "paper_state_hash": stable_hash(paper_state),
                    "monitoring_hash": stable_hash(monitoring),
                    "instrument_timing_hash": stable_hash(
                        [item.model_dump(mode="json") for item in timing_artifacts]
                    ),
                    "news_context_hash": stable_hash(
                        [item.model_dump(mode="json") for item in news_evidence]
                    ),
                }
            )
            run_dir = self.runs / run_id
            if run_dir.exists():
                _verify_checksums(run_dir)
                record = PublicationRecordV2.model_validate_json(
                    (run_dir / "publication.json").read_text(encoding="utf-8")
                )
                shutil.rmtree(temporary)
            else:
                record = PublicationRecordV2(
                    run_id=run_id,
                    published_at=datetime.now(UTC),
                    bundle_hash=bundle.bundle_hash,
                    data_version=bundle.data_version,
                    artifact_version=bundle.artifact_version,
                    policy_version=bundle.policy_version,
                    file_hashes=base_hashes,
                )
                _write_json(temporary / "publication.json", record.model_dump(mode="json"))
                _write_json(temporary / "checksums.json", _hash_files(temporary))
                self.runs.mkdir(parents=True, exist_ok=True)
                temporary.replace(run_dir)
                _verify_checksums(run_dir)

            pointer = CurrentBundlePointerV2(
                run_id=run_id,
                bundle_hash=bundle.bundle_hash,
                risk_inputs_hash=stable_hash(inputs_payload),
                session=bundle.session,
            )
            if before_pointer_swap is not None:
                before_pointer_swap(run_dir)
            atomic_write_bytes(self.root / "current.json", canonical_json(pointer))
            return record
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise


def _versioned_payload(version_set: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return {**version_set, "payload": payload}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(payload))


def _hash_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*.json"))
        if path.name != "checksums.json"
    }


def _verify_checksums(run_dir: Path) -> None:
    path = run_dir / "checksums.json"
    if not path.exists():
        raise DataError(f"published run has no checksums: {run_dir.name}")
    try:
        expected = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataError(f"invalid publication checksums: {exc}") from exc
    if not isinstance(expected, dict) or not expected:
        raise DataError("publication checksums must be a non-empty object")
    for relative, checksum in expected.items():
        candidate = run_dir / str(relative)
        if not candidate.exists() or file_sha256(candidate) != checksum:
            raise DataError(f"publication checksum mismatch: {relative}")


def verify_published_run(run_dir: Path) -> None:
    """Public verifier used by every canonical read surface."""
    _verify_checksums(run_dir)
