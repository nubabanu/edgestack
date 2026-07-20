"""Acceptance controls: noise, injected effects, OOS decay, and costs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from edgestack.research.opening_fade import (
    ValidationConfig,
    build_daily_context,
    build_session_features,
    load_campaign_config,
    occurrence_analysis,
    synthetic_intraday_fixture,
    validate_candidates,
)
from edgestack.types import CostScenario

ROOT = Path(__file__).resolve().parents[2]


def _trade_frame(candidate_id: str, returns: np.ndarray) -> pd.DataFrame:
    sessions = pd.bdate_range("2018-01-02", periods=len(returns))
    return pd.DataFrame(
        {
            "candidate_id": candidate_id,
            "family": "A",
            "symbol": "SPY",
            "session": sessions,
            "net_return": returns,
            "gross_return": returns + 0.0002,
            "cost_return": 0.0002,
            "mfe": np.maximum(returns, 0),
            "mae": np.minimum(returns, 0),
            "holding_minutes": 30,
            "exit_reason": "time_exit",
            "regime": "CALM",
        }
    )


def _scenario_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {scenario.value: frame.copy() for scenario in CostScenario}


def _small_validation_config():
    cfg = load_campaign_config(ROOT / "configs" / "opening_fade.yaml")
    return cfg.model_copy(
        update={
            "validation": ValidationConfig(
                train_sessions=60,
                test_sessions=20,
                n_folds=2,
                embargo_sessions=1,
                bootstrap_samples=100,
                block_length=5,
                minimum_effective_sample_size=20,
                minimum_positive_fold_fraction=0.70,
                fdr_alpha=0.05,
                random_seed=7,
            )
        }
    )


def test_shuffled_noise_does_not_produce_validated_edge() -> None:
    cfg = _small_validation_config()
    rng = np.random.default_rng(11)
    shuffled = rng.normal(0, 0.001, 100)
    shuffled = shuffled - shuffled.mean()
    frame = _trade_frame("A_VWAP_15", shuffled)
    ledger = [
        {
            "candidate_id": item.candidate_id,
            "status": "EVALUATED" if item.candidate_id == "A_VWAP_15" else "NO_TRADES",
        }
        for item in cfg.candidates
    ]
    result = validate_candidates(
        cfg,
        _scenario_frames(frame),
        ledger,
        pd.DatetimeIndex(frame["session"]),
    )
    assert result["paper_observation_candidates"] == []


def test_injected_opening_fade_effect_is_detected_descriptively() -> None:
    cfg = _small_validation_config()
    intraday, daily = synthetic_intraday_fixture(sessions=100, injected_fade=0.004, seed=9)
    features, _ = build_session_features(
        intraday,
        daily_context=build_daily_context(daily),
        definition=cfg.definition,
    )
    occurrence = occurrence_analysis(features, cfg.definition)
    retraced = next(item for item in occurrence["summary"] if item["definition"] == "retraced_50")
    assert retraced["frequency"] > 0.75


def test_training_only_effect_that_disappears_oos_is_rejected() -> None:
    cfg = _small_validation_config()
    returns = np.r_[np.full(60, 0.002), np.full(40, -0.001)]
    frame = _trade_frame("A_VWAP_15", returns)
    ledger = [
        {
            "candidate_id": item.candidate_id,
            "status": "EVALUATED" if item.candidate_id == "A_VWAP_15" else "NO_TRADES",
        }
        for item in cfg.candidates
    ]
    result = validate_candidates(
        cfg, _scenario_frames(frame), ledger, pd.DatetimeIndex(frame["session"])
    )
    candidate = next(item for item in result["candidates"] if item["candidate_id"] == "A_VWAP_15")
    assert "FOLD_SIGN_INCONSISTENCY" in candidate["failure_reasons"]
    assert not candidate["paper_observation_candidate"]


def test_effect_smaller_than_conservative_costs_is_rejected() -> None:
    cfg = _small_validation_config()
    frame = _trade_frame("A_VWAP_15", np.full(100, -0.0001))
    frame["gross_return"] = 0.0001
    frame["cost_return"] = 0.0002
    ledger = [
        {
            "candidate_id": item.candidate_id,
            "status": "EVALUATED" if item.candidate_id == "A_VWAP_15" else "NO_TRADES",
        }
        for item in cfg.candidates
    ]
    result = validate_candidates(
        cfg, _scenario_frames(frame), ledger, pd.DatetimeIndex(frame["session"])
    )
    candidate = next(item for item in result["candidates"] if item["candidate_id"] == "A_VWAP_15")
    assert "NONPOSITIVE_CONSERVATIVE_EXPECTANCY" in candidate["failure_reasons"]
    assert "DID_NOT_SURVIVE_STRESS_COSTS" in candidate["failure_reasons"]
