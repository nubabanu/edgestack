"""Shared integration fixtures: a fully prepared research environment.

One synthetic market with an injected Friday drift is pushed through the
whole research pipeline (catalog -> features -> discovery -> validation ->
edge store) once per test session; multiple test modules reuse it.
"""

from __future__ import annotations

from datetime import date

import pytest

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog
from edgestack.data.providers.synthetic import GBM, CalendarEffect, SyntheticMarket
from edgestack.discovery.candidate_generation import generate_candidates
from edgestack.discovery.edge_store import save_batch, save_edges
from edgestack.pipelines import _load_research_frames, run_features_build
from edgestack.types import Predicate
from edgestack.validation.walk_forward import validate_batch


@pytest.fixture(scope="session")
def signal_cfg(tmp_path_factory) -> EdgeStackConfig:
    tmp = tmp_path_factory.mktemp("research-env")
    return EdgeStackConfig.model_validate(
        {
            "paths": {"data_dir": str(tmp / "data"), "artifacts_dir": str(tmp / "artifacts")},
            "universe": {
                "symbols": ["AAA", "BBB", "CCC"], "benchmark_symbol": "AAA",
                "source": "synthetic", "min_price": 1.0,
                "min_median_dollar_volume": 1000000,
            },
            "signals": {
                "horizons": [1, 5],
                "min_effective_sample_size": 100,
                "min_probability_of_profit": 0.5,
                "min_expected_net_return": 0.001,
                "min_conviction_score": 45,
                "min_reward_to_risk": 1.0,
            },
            "validation": {
                "n_folds": 3, "test_sessions": 250, "train_min_sessions": 750,
                "embargo_sessions": 5, "final_test_start": "2022-01-01",
                "bootstrap_samples": 500,
            },
            "discovery": {"horizons": [1], "min_support": 50},
        }
    )


@pytest.fixture(scope="session")
def prepared(signal_cfg: EdgeStackConfig) -> EdgeStackConfig:
    """Run the whole research pipeline once for the test session."""
    market = SyntheticMarket(
        seed=42, base=GBM(mu=0.0, sigma=0.05),
        effects=(CalendarEffect(facts_column="is_friday", drift_bps=60.0),),
    )
    panel = market.generate(("AAA", "BBB", "CCC"), date(2010, 1, 1), date(2020, 12, 31))
    catalog = DataCatalog(signal_cfg)
    catalog.write_bars(panel, provider="synthetic")

    run_features_build(signal_cfg)

    catalog, features, labels = _load_research_frames(signal_cfg)
    experiment_id = catalog.record_experiment("discovery")
    conditions = [
        *[Predicate(feature="cal_weekday", op="==", value=float(wd)) for wd in range(5)],
        Predicate(feature="cal_turn_of_month", op="==", value=1.0),
    ]
    batch = generate_candidates(features, labels, signal_cfg, experiment_id,
                                conditions=conditions)
    save_batch(catalog, batch)
    edges = validate_batch(batch, features, labels, signal_cfg)
    save_edges(catalog, edges)
    return signal_cfg
