"""Verified canonical-bundle reads and stateless recommendation previews."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from edgestack.exceptions import DataError
from edgestack.recommendation.hashing import stable_hash
from edgestack.recommendation.instrument_schemas import FrozenTimingArtifactV2, NewsEvidenceV2
from edgestack.recommendation.risk import RiskInputsV2, size_recommendation
from edgestack.recommendation.schemas import (
    CanonicalRecommendationBundleV2,
    PortfolioRecommendationV2,
    RiskProfileV2,
    RiskStateV2,
)


class CurrentBundlePointerV2(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=2, ge=2, le=2)
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    risk_inputs_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    session: date


class CanonicalBundleRepository:
    def __init__(self, artifacts_dir: Path):
        self.root = artifacts_dir / "recommendations"

    @property
    def pointer_path(self) -> Path:
        return self.root / "current.json"

    def pointer(self) -> CurrentBundlePointerV2:
        if not self.pointer_path.exists():
            raise DataError("canonical recommendation has not been published")
        try:
            return CurrentBundlePointerV2.model_validate_json(
                self.pointer_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise DataError(f"invalid canonical recommendation pointer: {exc}") from exc

    def run_dir(self, pointer: CurrentBundlePointerV2 | None = None) -> Path:
        selected = pointer or self.pointer()
        return self.root / "runs" / selected.run_id

    def latest(self) -> CanonicalRecommendationBundleV2:
        pointer = self.pointer()
        run_dir = self.run_dir(pointer)
        _verify_run(run_dir)
        path = run_dir / "recommendation.json"
        if not path.exists():
            raise DataError(f"published recommendation is missing: {pointer.run_id}")
        try:
            bundle = CanonicalRecommendationBundleV2.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise DataError(f"invalid published recommendation: {exc}") from exc
        if bundle.bundle_hash != pointer.bundle_hash:
            raise DataError("published recommendation bundle hash mismatch")
        if bundle.session != pointer.session:
            raise DataError("published recommendation session mismatch")
        return bundle

    def risk_inputs(self) -> RiskInputsV2:
        pointer = self.pointer()
        run_dir = self.run_dir(pointer)
        _verify_run(run_dir)
        path = run_dir / "risk_inputs.json"
        if not path.exists():
            raise DataError(f"published risk inputs are missing: {pointer.run_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("risk inputs are not an object")
            if stable_hash(payload) != pointer.risk_inputs_hash:
                raise DataError("published risk-input hash mismatch")
            return _risk_inputs_from_json(payload)
        except DataError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DataError(f"invalid published risk inputs: {exc}") from exc

    def timing_artifacts(self) -> tuple[FrozenTimingArtifactV2, ...]:
        payload = self._optional_versioned_payload("instrument_timing.json")
        raw = payload.get("artifacts", [])
        if not isinstance(raw, list):
            raise DataError("published timing artifacts are not a list")
        try:
            items = tuple(FrozenTimingArtifactV2.model_validate(item) for item in raw)
            bundle = self.latest()
            expected = (bundle.data_version, bundle.artifact_version, bundle.policy_version)
            for item in items:
                versions = (item.data_version, item.artifact_version, item.policy_version)
                if versions != expected:
                    raise DataError(
                        f"published timing artifact version mismatch for "
                        f"{item.symbol}/{item.horizon}"
                    )
            return items
        except DataError:
            raise
        except ValueError as exc:
            raise DataError(f"invalid published timing artifact: {exc}") from exc

    def news_evidence(self, symbol: str | None = None) -> tuple[NewsEvidenceV2, ...]:
        payload = self._optional_versioned_payload("news_context.json")
        raw = payload.get("items", [])
        if not isinstance(raw, list):
            raise DataError("published news context is not a list")
        try:
            items = tuple(NewsEvidenceV2.model_validate(value) for value in raw)
            return (
                items if symbol is None else tuple(item for item in items if item.symbol == symbol)
            )
        except ValueError as exc:
            raise DataError(f"invalid published news evidence: {exc}") from exc

    def _optional_versioned_payload(self, filename: str) -> dict[str, Any]:
        pointer = self.pointer()
        run_dir = self.run_dir(pointer)
        _verify_run(run_dir)
        path = run_dir / filename
        # Compatibility for the first V2 runs published before instrument analysis.
        if not path.exists():
            return {}
        try:
            wrapper = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(wrapper, dict) or not isinstance(wrapper.get("payload"), dict):
                raise TypeError("versioned document is not an object")
            bundle = self.latest()
            expected = {
                "session": bundle.session.isoformat(),
                "data_version": bundle.data_version,
                "artifact_version": bundle.artifact_version,
                "policy_version": bundle.policy_version,
                "bundle_hash": bundle.bundle_hash,
            }
            for key, value in expected.items():
                if wrapper.get(key) != value:
                    raise DataError(f"published {filename} {key} mismatch")
            return dict(wrapper["payload"])
        except DataError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise DataError(f"invalid published {filename}: {exc}") from exc


class CanonicalRecommendationService:
    def __init__(self, repository: CanonicalBundleRepository):
        self.repository = repository

    def latest(self) -> CanonicalRecommendationBundleV2:
        return self.repository.latest()

    def preview(
        self,
        *,
        profile: RiskProfileV2,
        state: RiskStateV2 | None = None,
        equity_override: float | None = None,
        reset_requested: bool = False,
    ) -> PortfolioRecommendationV2:
        bundle = self.repository.latest()
        inputs = self.repository.risk_inputs()
        selected_profile = profile
        if equity_override is not None:
            selected_profile = RiskProfileV2.model_validate(
                {**profile.model_dump(), "account_equity": equity_override}
            )
        selected_state = state or bundle.default_recommendation.output_risk_state
        if inputs.session != bundle.session:
            raise DataError("risk inputs and canonical bundle sessions differ")
        return size_recommendation(
            base=bundle.base_recommendation,
            profile=selected_profile,
            state=selected_state,
            inputs=inputs,
            reset_requested=reset_requested,
        )


def risk_inputs_json(inputs: RiskInputsV2) -> dict[str, Any]:
    return {
        "session": inputs.session.isoformat(),
        "stressed_forecast_volatility": inputs.stressed_forecast_volatility,
        "parametric_995_one_day_loss": inputs.parametric_995_one_day_loss,
        "historical_995_one_day_loss": inputs.historical_995_one_day_loss,
        "bootstrapped_99_path_drawdown": inputs.bootstrapped_99_path_drawdown,
        "liquidity_position_limits": dict(sorted(inputs.liquidity_position_limits.items())),
        "funding_rate": inputs.funding_rate,
        "funding_rate_as_of": inputs.funding_rate_as_of.isoformat(),
        "monitoring_healthy": inputs.monitoring_healthy,
        "stress_acceptable": inputs.stress_acceptable,
        "data_fresh": inputs.data_fresh,
    }


def _risk_inputs_from_json(payload: dict[str, Any]) -> RiskInputsV2:
    liquidity = payload["liquidity_position_limits"]
    if not isinstance(liquidity, dict):
        raise TypeError("liquidity_position_limits is not an object")
    return RiskInputsV2(
        session=date.fromisoformat(str(payload["session"])),
        stressed_forecast_volatility=float(payload["stressed_forecast_volatility"]),
        parametric_995_one_day_loss=float(payload["parametric_995_one_day_loss"]),
        historical_995_one_day_loss=float(payload["historical_995_one_day_loss"]),
        bootstrapped_99_path_drawdown=float(payload["bootstrapped_99_path_drawdown"]),
        liquidity_position_limits={str(key): float(value) for key, value in liquidity.items()},
        funding_rate=float(payload["funding_rate"]),
        funding_rate_as_of=date.fromisoformat(str(payload["funding_rate_as_of"])),
        monitoring_healthy=bool(payload.get("monitoring_healthy", True)),
        stress_acceptable=bool(payload.get("stress_acceptable", True)),
        data_fresh=bool(payload.get("data_fresh", True)),
    )


def _verify_run(run_dir: Path) -> None:
    # Local import keeps the publisher/repository dependency acyclic at import time.
    from edgestack.recommendation.publication import verify_published_run

    verify_published_run(run_dir)
