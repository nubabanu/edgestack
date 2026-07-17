"""Canonical contract, hashing, and immutable artifact tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog
from edgestack.exceptions import DataError
from edgestack.recommendation.artifacts import FrozenArtifactStore
from edgestack.recommendation.hashing import stable_hash
from edgestack.recommendation.manifests import (
    ExperimentManifestV2,
    OuterFoldV2,
    TrialKind,
    TrialRecordV2,
    TrialStatus,
)
from edgestack.recommendation.policy import load_baseline_policy
from edgestack.recommendation.registry import RecommendationRegistry
from edgestack.recommendation.schemas import RiskProfileV2, RiskStateV2


def _manifest(trial_ids: tuple[str, ...] = ("t1",)) -> ExperimentManifestV2:
    return ExperimentManifestV2(
        experiment_id="exp-v2",
        code_revision="66554e9",
        data_version="data-v1",
        data_hashes={"bars": "abc"},
        universe_definition={"name": "synthetic"},
        point_in_time_coverage={"overall": 1.0},
        feature_definitions=({"name": "momentum"},),
        labels=({"horizon": 5},),
        horizons=(5,),
        candidate_family=trial_ids,
        execution_policies=({"entry": "next_open"},),
        transaction_costs={"scenario": "CONSERVATIVE"},
        financing={"spread_bps": 200},
        calibration_method="cross_fitted_isotonic",
        outer_folds=(
            OuterFoldV2(
                fold_id="2024",
                train_start=date(2020, 1, 1),
                train_end=date(2023, 12, 31),
                test_start=date(2024, 1, 1),
                test_end=date(2024, 12, 31),
                train_sessions=1000,
                test_sessions=252,
                valid=True,
            ),
        ),
        complete_trial_ids=trial_ids,
        random_seeds=(42,),
        policy_version="baseline-diversified-v1",
    )


def test_baseline_policy_is_long_only_unit_weight() -> None:
    policy = load_baseline_policy()
    assert {w.symbol: w.weight for w in policy.weights} == {
        "SPY": 0.25,
        "TLT": 0.25,
        "SHY": 0.25,
        "GLD": 0.25,
    }
    assert policy.alpha_claimed is False


def test_hashes_are_order_independent_and_ignore_no_content() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert _manifest().manifest_hash == _manifest().manifest_hash
    assert _manifest(("t1", "t2")).manifest_hash != _manifest().manifest_hash


def test_frozen_artifact_rejects_tamper_and_incompatible_context(tmp_path: Path) -> None:
    store = FrozenArtifactStore(tmp_path / "frozen")
    artifact = store.freeze_json(
        artifact_type="edge",
        artifact_version="edge-v1",
        manifest_hash="manifest",
        data_version="data-v1",
        feature_version="features-v1",
        policy_version="baseline-diversified-v1",
        horizon_sessions=5,
        payload={"threshold": 0.4},
    )
    assert store.payload(artifact.content_hash, horizon_sessions=5) == {"threshold": 0.4}
    with pytest.raises(DataError, match="data_version mismatch"):
        store.load(artifact.content_hash, data_version="data-v2")
    payload_path = tmp_path / "frozen" / artifact.content_hash / "payload.json"
    payload_path.write_text('{"threshold":0.5}', encoding="utf-8")
    with pytest.raises(DataError, match="checksum"):
        store.load(artifact.content_hash)


def test_trial_must_be_registered_and_manifests_are_immutable(tmp_path: Path) -> None:
    cfg = EdgeStackConfig.model_validate(
        {"paths": {"data_dir": tmp_path / "data", "artifacts_dir": tmp_path / "artifacts"}}
    )
    registry = RecommendationRegistry(DataCatalog(cfg))
    manifest = _manifest()
    registry.save_manifest(manifest)
    registry.save_manifest(manifest)
    trial = TrialRecordV2(
        trial_id="t1",
        experiment_id="exp-v2",
        kind=TrialKind.STANDALONE,
        family="momentum",
        horizon_sessions=5,
    )
    registry.register_trial(trial)
    assert registry.trials("exp-v2") == (trial,)
    with pytest.raises(DataError, match="already registered"):
        registry.register_trial(trial)


def test_risk_contract_bounds_and_state_consistency() -> None:
    profile = RiskProfileV2(maximum_gross_leverage=5.0)
    assert profile.profile_hash == RiskProfileV2(maximum_gross_leverage=5.0).profile_hash
    with pytest.raises(ValueError):
        RiskProfileV2(maximum_gross_leverage=5.01)
    assert RiskStateV2.initial(100_000).current_drawdown == 0
    with pytest.raises(ValueError, match="current_drawdown"):
        RiskStateV2(peak_equity=100_000, current_equity=90_000, current_drawdown=0.05)


def test_failed_cached_and_compound_trials_remain_in_complete_family(tmp_path: Path) -> None:
    cfg = EdgeStackConfig.model_validate(
        {"paths": {"data_dir": tmp_path / "data", "artifacts_dir": tmp_path / "artifacts"}}
    )
    registry = RecommendationRegistry(DataCatalog(cfg))
    trials = (
        TrialRecordV2(
            trial_id="failed",
            experiment_id="complete-family",
            kind=TrialKind.STANDALONE,
            family="momentum",
            horizon_sessions=5,
        ),
        TrialRecordV2(
            trial_id="cached",
            experiment_id="complete-family",
            kind=TrialKind.EXECUTION_VARIANT,
            family="momentum",
            horizon_sessions=5,
        ),
        TrialRecordV2(
            trial_id="compound",
            experiment_id="complete-family",
            kind=TrialKind.INTERACTION,
            family="momentum_x_liquidity",
            horizon_sessions=5,
            parent_trial_ids=("failed", "cached"),
        ),
    )
    for trial in trials:
        registry.register_trial(trial)
    registry.update_trial(
        trials[0].model_copy(update={"status": TrialStatus.FAILED, "failure_reason": "fit error"})
    )
    registry.update_trial(
        trials[1].model_copy(
            update={"status": TrialStatus.CACHED, "cached_from_trial_id": "failed"}
        )
    )
    registry.update_trial(trials[2].model_copy(update={"status": TrialStatus.REJECTED}))

    recorded = registry.trials("complete-family")
    assert {trial.trial_id for trial in recorded} == {"failed", "cached", "compound"}
    assert {trial.status for trial in recorded} == {
        TrialStatus.FAILED,
        TrialStatus.CACHED,
        TrialStatus.REJECTED,
    }
