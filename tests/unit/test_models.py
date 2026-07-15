"""Model training, selection, calibration and artifact-registry tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edgestack.config import EdgeStackConfig
from edgestack.exceptions import DataError
from edgestack.models.calibration import (
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
    fit_isotonic,
    log_loss_score,
    reliability_bins,
)
from edgestack.models.registry import load_models, save_models
from edgestack.models.training import train_models
from edgestack.types import Side


def _model_cfg() -> EdgeStackConfig:
    return EdgeStackConfig.model_validate(
        {
            "signals": {"horizons": [5]},
            "validation": {
                "n_folds": 3, "test_sessions": 200, "train_min_sessions": 400,
                "embargo_sessions": 5, "final_test_start": "2022-01-01",
            },
        }
    )


def _research_frames(signal_strength: float, seed: int = 0):
    """Synthetic features/labels where `x1` (optionally) predicts the outcome."""
    rng = np.random.default_rng(seed)
    n = 1200
    dates = pd.bdate_range("2015-01-01", periods=n)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.02, n)
    gross = signal_strength * 0.02 * x1 + noise
    features = pd.DataFrame({"symbol": "TST", "date": dates, "x1": x1, "x2": x2})
    labels = pd.DataFrame(
        {
            "symbol": "TST",
            "date": dates,
            "horizon": 5,
            "label_end": dates[np.minimum(np.arange(n) + 5, n - 1)],
            "gross_ret": gross,
        }
    )
    return features, labels


def test_calibration_metrics_known_values() -> None:
    y = np.array([1, 0, 1, 0])
    perfect = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y, perfect) == 0.0
    assert log_loss_score(y, perfect) < 1e-9
    uniform = np.full(4, 0.5)
    assert brier_score(y, uniform) == pytest.approx(0.25)
    # ECE of a perfectly calibrated constant predictor is 0.
    assert expected_calibration_error(np.array([1, 0, 1, 0]), uniform) == pytest.approx(0.0)
    bins = reliability_bins(y, np.array([0.9, 0.1, 0.8, 0.2]), n_bins=2)
    assert bins[0]["observed_frequency"] == 0.0 and bins[1]["observed_frequency"] == 1.0


def test_isotonic_and_platt_improve_miscalibrated_scores() -> None:
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.1, 0.9, 3000)
    y = (rng.uniform(0, 1, 3000) < true_p).astype(int)
    overconfident = np.clip(true_p + 0.6 * (true_p - 0.5), 0.01, 0.99)
    raw_ece = expected_calibration_error(y, overconfident)
    iso = fit_isotonic(overconfident, y)
    iso_ece = expected_calibration_error(y, np.clip(iso.predict(overconfident), 0, 1))
    platt = PlattCalibrator().fit(overconfident, y)
    platt_ece = expected_calibration_error(y, platt.predict(overconfident))
    assert iso_ece < raw_ece
    assert platt_ece < raw_ece


def test_noise_selects_base_rate_model() -> None:
    features, labels = _research_frames(signal_strength=0.0)
    models = train_models(features, labels, _model_cfg(), sides=(Side.LONG,))
    assert len(models) == 1
    assert models[0].name == "base_rate"  # complexity did not pay for itself
    p = models[0].predict_probability(features.head(50))
    assert ((p >= 0) & (p <= 1)).all()


def test_real_signal_upgrades_model_and_beats_base_rate() -> None:
    features, labels = _research_frames(signal_strength=1.0)
    models = train_models(features, labels, _model_cfg(), sides=(Side.LONG,))
    model = models[0]
    assert model.name != "base_rate"
    assert model.metrics["oof_log_loss"] < model.metrics["base_rate_log_loss"]
    assert model.is_calibrated
    # Probabilities must track the signal direction.
    hi = features.assign(x1=2.0)
    lo = features.assign(x1=-2.0)
    assert model.predict_probability(hi.head(20)).mean() > (
        model.predict_probability(lo.head(20)).mean()
    )


def test_model_artifact_round_trip_and_tamper_detection(tmp_path) -> None:
    features, labels = _research_frames(signal_strength=1.0)
    models = train_models(features, labels, _model_cfg(), sides=(Side.LONG,))
    save_models(tmp_path, models)

    loaded = load_models(tmp_path)
    assert loaded[0].name == models[0].name
    p_before = models[0].predict_probability(features.head(10))
    p_after = loaded[0].predict_probability(features.head(10))
    np.testing.assert_allclose(p_before, p_after)

    # Tamper with the pickle: loading must refuse before unpickling.
    pkl = next(tmp_path.glob("*.pkl"))
    pkl.write_bytes(pkl.read_bytes() + b"tampered")
    with pytest.raises(DataError, match="checksum mismatch"):
        load_models(tmp_path)
