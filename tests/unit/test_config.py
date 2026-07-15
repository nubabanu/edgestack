"""Configuration validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from edgestack.config import EdgeStackConfig, load_config
from edgestack.exceptions import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_defaults_are_valid() -> None:
    cfg = EdgeStackConfig()
    assert cfg.project.name == "EdgeStack"
    assert cfg.paper.live_trading_enabled is False


def test_default_yaml_loads_and_round_trips() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    dumped = cfg.model_dump(mode="json")
    again = EdgeStackConfig.model_validate(dumped)
    assert again.config_hash() == cfg.config_hash()


def test_config_hash_changes_with_content() -> None:
    a = EdgeStackConfig()
    b = EdgeStackConfig.model_validate({"project": {"random_seed": 43}})
    assert a.config_hash() != b.config_hash()


def test_missing_explicit_config_raises() -> None:
    with pytest.raises(ConfigError):
        load_config("does/not/exist.yaml")


def test_live_trading_cannot_be_enabled() -> None:
    with pytest.raises(ConfigError):
        _cfg_from({"paper": {"live_trading_enabled": True}})


def test_embargo_must_cover_longest_horizon() -> None:
    with pytest.raises(ConfigError):
        _cfg_from({"validation": {"embargo_sessions": 5}, "signals": {"horizons": [1, 60]}})


def test_horizons_must_be_sorted_unique() -> None:
    with pytest.raises(ConfigError):
        _cfg_from({"signals": {"horizons": [5, 1]}})


def test_invalid_symbol_rejected() -> None:
    with pytest.raises(ConfigError):
        _cfg_from({"universe": {"symbols": ["AAPL; DROP TABLE"]}})


def test_unknown_key_rejected() -> None:
    with pytest.raises(ConfigError):
        _cfg_from({"projectt": {"name": "typo"}})


def _cfg_from(overrides: dict) -> EdgeStackConfig:
    try:
        return EdgeStackConfig.model_validate(overrides)
    except Exception as exc:  # normalize pydantic errors to ConfigError for assertions
        raise ConfigError(str(exc)) from exc
