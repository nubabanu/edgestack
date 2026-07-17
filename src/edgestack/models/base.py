"""Model interfaces and factories.

Every model estimates P(net trade return > 0) for one (horizon, side) target.
Complex models must EARN their place: selection keeps the simplest model
whose out-of-fold log loss is not materially beaten by a more complex one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BaseRateClassifier(BaseEstimator, ClassifierMixin):
    """Predicts the training-set base rate for every row — the floor to beat."""

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        self.base_rate_ = float(y_arr.mean()) if len(y_arr) else 0.5
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.base_rate_)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_base_rate(seed: int) -> BaseEstimator:
    return BaseRateClassifier()


def make_logistic(seed: int) -> BaseEstimator:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=2000, random_state=seed)),
        ]
    )


def make_gradient_boosting(seed: int) -> BaseEstimator:
    # Shallow and regularized; handles NaN natively.
    return HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=150,
        learning_rate=0.05,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=seed,
    )


#: Ordered simplest -> most complex; selection walks this order.
MODEL_FACTORIES: tuple[tuple[str, Callable[[int], BaseEstimator]], ...] = (
    ("base_rate", make_base_rate),
    ("logistic", make_logistic),
    ("gradient_boosting", make_gradient_boosting),
)


@dataclass
class TrainedModel:
    """A fitted estimator + OOF-fitted calibrator for one (horizon, side)."""

    name: str
    horizon: int
    side: str
    feature_columns: tuple[str, ...]
    estimator: BaseEstimator
    calibrator: Any | None
    metrics: dict[str, float]
    reliability_bins: list[dict[str, float]]
    featureset_id: str
    config_hash: str
    seed: int
    calibration_fold_assignments: tuple[int, ...] = ()
    cross_fitted_calibrated_predictions: tuple[float, ...] = ()

    def predict_probability(self, features: pd.DataFrame) -> np.ndarray:
        """Calibrated P(net return > 0) for feature rows."""
        X = features[list(self.feature_columns)]
        raw = self.estimator.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            raw = np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return raw

    @property
    def is_calibrated(self) -> bool:
        return self.calibrator is not None
