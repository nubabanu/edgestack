"""Shared fixtures: isolated config + catalog per test, synthetic panels."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from edgestack.config import EdgeStackConfig
from edgestack.data.providers.synthetic import SyntheticMarket


@pytest.fixture()
def cfg(tmp_path: Path) -> EdgeStackConfig:
    """Config whose data/artifact dirs live in an isolated temp directory."""
    return EdgeStackConfig.model_validate(
        {
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "artifacts_dir": str(tmp_path / "artifacts"),
            },
            "universe": {"symbols": ["AAA", "BBB"], "source": "synthetic"},
            "validation": {"final_test_start": "2020-01-01"},
        }
    )


@pytest.fixture(scope="session")
def synthetic_panel() -> pd.DataFrame:
    """Small deterministic two-symbol panel spanning the guard boundary."""
    market = SyntheticMarket(seed=42)
    return market.generate(("AAA", "BBB"), date(2017, 1, 1), date(2020, 12, 31))
