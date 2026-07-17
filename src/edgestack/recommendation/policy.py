"""Tracked baseline-policy loading."""

from __future__ import annotations

from pathlib import Path

import yaml

from edgestack.recommendation.schemas import BaselinePolicyV2

DEFAULT_POLICY_PATH = Path("configs/policies/baseline-diversified-v1.yaml")


def load_baseline_policy(path: Path = DEFAULT_POLICY_PATH) -> BaselinePolicyV2:
    return BaselinePolicyV2.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
