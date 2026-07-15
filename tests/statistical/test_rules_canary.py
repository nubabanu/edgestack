"""The rules-campaign canary as a permanent statistical acceptance test."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

pytestmark = pytest.mark.statistical


def test_canary_rediscovers_injected_effect_and_rejects_noise() -> None:
    from rules_campaign import run_canary

    with_effect = run_canary(with_effect=True)
    assert with_effect["passing"] >= 1
    assert with_effect["tom_rule_found"], (
        "the injected turn-of-month structure must be rediscovered"
    )
    assert with_effect["tom_best"]["freq"] >= 0.7

    noise = run_canary(with_effect=False)
    assert noise["passing"] <= 1, (
        f"pure noise produced {noise['passing']} passing rules - "
        "stability selection or the CGS hurdle regressed"
    )
