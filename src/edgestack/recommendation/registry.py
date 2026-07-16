"""DuckDB registry for immutable V2 research and publication records."""

from __future__ import annotations

from datetime import UTC, datetime

from edgestack.data.catalog import DataCatalog
from edgestack.exceptions import DataError
from edgestack.recommendation.manifests import (
    ExperimentManifestV2,
    FrozenArtifactV2,
    PromotionDecisionV2,
    ProspectiveEvidenceV2,
    PublicationRecordV2,
    TrialRecordV2,
)


class RecommendationRegistry:
    def __init__(self, catalog: DataCatalog) -> None:
        self.catalog = catalog

    def save_manifest(self, manifest: ExperimentManifestV2) -> None:
        with self.catalog.connect() as con:
            existing = con.execute(
                "SELECT payload FROM experiment_manifests_v2 WHERE manifest_hash = ?",
                [manifest.manifest_hash],
            ).fetchone()
            payload = manifest.model_dump_json()
            if existing is not None:
                if ExperimentManifestV2.model_validate_json(existing[0]) != manifest:
                    raise DataError("immutable manifest hash collision")
                return
            con.execute(
                "INSERT INTO experiment_manifests_v2 VALUES (?, ?, ?, ?)",
                [manifest.manifest_hash, manifest.experiment_id, datetime.now(UTC), payload],
            )

    def register_trial(self, trial: TrialRecordV2) -> None:
        """Register a candidate before evaluation; status updates append to audit."""
        with self.catalog.connect() as con:
            if con.execute(
                "SELECT 1 FROM trial_ledger_v2 WHERE trial_id = ?", [trial.trial_id]
            ).fetchone():
                raise DataError(f"trial already registered: {trial.trial_id}")
            con.execute(
                "INSERT INTO trial_ledger_v2 VALUES (?, ?, ?, ?, ?)",
                [
                    trial.trial_id,
                    trial.experiment_id,
                    datetime.now(UTC),
                    trial.status.value,
                    trial.model_dump_json(),
                ],
            )

    def update_trial(self, trial: TrialRecordV2) -> None:
        with self.catalog.connect() as con:
            row = con.execute(
                "SELECT experiment_id FROM trial_ledger_v2 WHERE trial_id = ?", [trial.trial_id]
            ).fetchone()
            if row is None:
                raise DataError("candidate must be registered before evaluation")
            if row[0] != trial.experiment_id:
                raise DataError("trial experiment cannot change")
            con.execute(
                "UPDATE trial_ledger_v2 SET status = ?, payload = ? WHERE trial_id = ?",
                [trial.status.value, trial.model_dump_json(), trial.trial_id],
            )

    def trials(self, experiment_id: str) -> tuple[TrialRecordV2, ...]:
        with self.catalog.connect() as con:
            rows = con.execute(
                "SELECT payload FROM trial_ledger_v2 WHERE experiment_id = ? ORDER BY trial_id",
                [experiment_id],
            ).fetchall()
        return tuple(TrialRecordV2.model_validate_json(row[0]) for row in rows)

    def save_artifact(self, artifact: FrozenArtifactV2) -> None:
        self._insert_immutable(
            "frozen_artifacts_v2",
            "content_hash",
            artifact.content_hash,
            "INSERT INTO frozen_artifacts_v2 VALUES (?, ?, ?, ?)",
            [
                artifact.content_hash,
                artifact.manifest_hash,
                datetime.now(UTC),
                artifact.model_dump_json(),
            ],
            artifact,
            FrozenArtifactV2,
        )

    def save_promotion(self, decision: PromotionDecisionV2) -> None:
        with self.catalog.connect() as con:
            row = con.execute(
                "SELECT payload FROM promotion_decisions_v2 "
                "WHERE sleeve_id = ? AND artifact_hash = ?",
                [decision.sleeve_id, decision.artifact_hash],
            ).fetchone()
            if row is not None:
                if PromotionDecisionV2.model_validate_json(row[0]) != decision:
                    raise DataError("promotion decision is immutable")
                return
            con.execute(
                "INSERT INTO promotion_decisions_v2 VALUES (?, ?, ?, ?, ?)",
                [
                    decision.sleeve_id,
                    decision.artifact_hash,
                    datetime.now(UTC),
                    decision.promoted,
                    decision.model_dump_json(),
                ],
            )

    def append_prospective_evidence(self, evidence: ProspectiveEvidenceV2) -> None:
        with self.catalog.connect() as con:
            con.execute(
                "INSERT INTO prospective_evidence_v2 VALUES (?, ?, ?, ?)",
                [
                    evidence.sleeve_id,
                    evidence.frozen_artifact_hash,
                    datetime.now(UTC),
                    evidence.model_dump_json(),
                ],
            )

    def save_publication(self, publication: PublicationRecordV2) -> None:
        self._insert_immutable(
            "publications_v2",
            "run_id",
            publication.run_id,
            "INSERT INTO publications_v2 VALUES (?, ?, ?, ?)",
            [
                publication.run_id,
                publication.published_at,
                publication.bundle_hash,
                publication.model_dump_json(),
            ],
            publication,
            PublicationRecordV2,
        )

    def _insert_immutable(
        self, table, key_name, key, insert_sql, values, model, model_type
    ) -> None:
        with self.catalog.connect() as con:
            row = con.execute(f"SELECT payload FROM {table} WHERE {key_name} = ?", [key]).fetchone()
            if row is not None:
                if model_type.model_validate_json(row[0]) != model:
                    raise DataError(f"immutable record differs for {table}:{key}")
                return
            con.execute(insert_sql, values)
