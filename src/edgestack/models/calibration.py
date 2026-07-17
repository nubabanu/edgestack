"""Probability calibration and its diagnostics.

Calibration is fit on OUT-OF-FOLD predictions only: each fold's test-window
predictions come from a model that never saw that window. Fitting a
calibrator on in-sample predictions would just launder overconfidence.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from edgestack.exceptions import ValidationError


@dataclass(frozen=True)
class CrossFittedCalibration:
    final_calibrator: IsotonicRegression
    calibrated_oof: np.ndarray
    fold_assignments: np.ndarray
    metrics: dict[str, float]


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((np.asarray(p) - y) ** 2))


def log_loss_score(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    y = np.asarray(y_true, dtype=float)
    q = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE: |mean predicted - observed frequency| weighted by bin population."""
    y = np.asarray(y_true, dtype=float)
    q = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (q >= lo) & (q < hi) if hi < 1.0 else (q >= lo) & (q <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(q[mask].mean() - y[mask].mean())
    return float(ece)


def reliability_bins(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Reliability-diagram data: per bin, predicted vs observed frequency."""
    y = np.asarray(y_true, dtype=float)
    q = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in itertools.pairwise(edges):
        mask = (q >= lo) & (q < hi) if hi < 1.0 else (q >= lo) & (q <= hi)
        if mask.sum() == 0:
            continue
        out.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "count": float(mask.sum()),
                "mean_predicted": float(q[mask].mean()),
                "observed_frequency": float(y[mask].mean()),
            }
        )
    return out


def fit_isotonic(oof_pred: np.ndarray, oof_true: np.ndarray) -> IsotonicRegression:
    if len(oof_pred) < 50:
        raise ValidationError("need at least 50 out-of-fold predictions to calibrate")
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(oof_pred, dtype=float), np.asarray(oof_true, dtype=float))
    return iso


def cross_fit_isotonic(
    oof_pred: np.ndarray,
    oof_true: np.ndarray,
    fold_assignments: np.ndarray,
) -> CrossFittedCalibration:
    """Evaluate calibration only on rows excluded from calibrator fitting."""
    raw = np.asarray(oof_pred, dtype=float)
    true = np.asarray(oof_true, dtype=int)
    folds = np.asarray(fold_assignments, dtype=int)
    if not (len(raw) == len(true) == len(folds)):
        raise ValidationError("predictions, outcomes, and calibration folds must align")
    unique = np.unique(folds)
    if len(unique) < 2:
        raise ValidationError("cross-fitted calibration needs at least two folds")
    calibrated = np.full(len(raw), np.nan)
    for fold in unique:
        evaluation = folds == fold
        training = ~evaluation
        calibrator = fit_isotonic(raw[training], true[training])
        calibrated[evaluation] = np.clip(calibrator.predict(raw[evaluation]), 0.0, 1.0)
    if not np.isfinite(calibrated).all():
        raise ValidationError("cross-fitted calibration did not cover every row")
    final = fit_isotonic(raw, true)
    return CrossFittedCalibration(
        final_calibrator=final,
        calibrated_oof=calibrated,
        fold_assignments=folds,
        metrics={
            "oof_brier": brier_score(true, calibrated),
            "oof_log_loss": log_loss_score(true, calibrated),
            "oof_ece": expected_calibration_error(true, calibrated),
        },
    )


class PlattCalibrator:
    """Platt scaling: logistic fit on the raw score (log-odds transformed)."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(max_iter=1000)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        q = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
        return np.log(q / (1.0 - q)).reshape(-1, 1)

    def fit(self, oof_pred: np.ndarray, oof_true: np.ndarray) -> PlattCalibrator:
        if len(oof_pred) < 50:
            raise ValidationError("need at least 50 out-of-fold predictions to calibrate")
        self._lr.fit(self._logit(oof_pred), np.asarray(oof_true, dtype=int))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(self._logit(p))[:, 1]
