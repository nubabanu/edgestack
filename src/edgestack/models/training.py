"""Model training with walk-forward OOF evaluation, selection and calibration.

For each (horizon, side):

1. build the binary target: net-of-CONSERVATIVE-cost return > 0;
2. per purged fold: fit each candidate model on train, predict the test fold
   -> pooled out-of-fold predictions;
3. select the SIMPLEST model whose OOF log loss is within tolerance of the
   best (complexity must pay for itself out of sample);
4. fit an isotonic calibrator on the winner's OOF predictions;
5. refit the winner on all (guard-truncated) data and attach the calibrator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.exceptions import ValidationError
from edgestack.execution.costs import CostModel
from edgestack.features.registry import featureset_id
from edgestack.logging import get_logger, log_event
from edgestack.models.base import MODEL_FACTORIES, TrainedModel
from edgestack.models.calibration import (
    brier_score,
    expected_calibration_error,
    fit_isotonic,
    log_loss_score,
    reliability_bins,
)
from edgestack.types import Side
from edgestack.validation.splits import PurgedWalkForwardSplitter

log = get_logger("models")

#: A more complex model must improve OOF log loss by at least this relative
#: margin over a simpler one to be selected.
COMPLEXITY_TOLERANCE = 0.01


def feature_matrix_columns(features: pd.DataFrame) -> tuple[str, ...]:
    return tuple(c for c in features.columns if c not in ("symbol", "date"))


def train_models(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cfg: EdgeStackConfig,
    *,
    horizons: tuple[int, ...] | None = None,
    sides: tuple[Side, ...] = (Side.LONG, Side.SHORT),
) -> list[TrainedModel]:
    horizons = horizons or cfg.signals.horizons
    splitter = PurgedWalkForwardSplitter.from_config(cfg)
    cost_model = CostModel.from_config(cfg)
    columns = feature_matrix_columns(features)
    fsid = featureset_id() if set(columns) >= {"rsi_14"} else "custom"
    models: list[TrainedModel] = []

    for horizon in horizons:
        h_labels = labels.loc[labels["horizon"] == horizon]
        merged = features.merge(
            h_labels[["symbol", "date", "label_end", "gross_ret"]],
            on=["symbol", "date"],
            how="inner",
        ).reset_index(drop=True)
        if merged.empty:
            raise ValidationError(f"no rows for horizon {horizon}")
        folds = splitter.split_frame(merged["date"], merged["label_end"])
        X = merged[list(columns)]

        for side in sides:
            gross = merged["gross_ret"].to_numpy()
            directional = gross if side is Side.LONG else -gross
            net = directional - cost_model.roundtrip_cost(side, horizon)
            y = (net > 0).astype(int)

            oof_by_model: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for name, factory in MODEL_FACTORIES:
                preds: list[np.ndarray] = []
                trues: list[np.ndarray] = []
                for fold in folds:
                    est = factory(cfg.project.random_seed)
                    est.fit(X.iloc[fold.train_idx], y[fold.train_idx])
                    preds.append(est.predict_proba(X.iloc[fold.test_idx])[:, 1])
                    trues.append(y[fold.test_idx])
                oof_by_model[name] = (np.concatenate(preds), np.concatenate(trues))

            selected = _select_simplest(oof_by_model)
            oof_pred, oof_true = oof_by_model[selected]
            calibrator = None
            if selected != "base_rate":
                calibrator = fit_isotonic(oof_pred, oof_true)
                calibrated = np.clip(calibrator.predict(oof_pred), 0.0, 1.0)
            else:
                calibrated = oof_pred  # a constant base rate is calibrated by construction

            base_pred, base_true = oof_by_model["base_rate"]
            metrics = {
                "oof_brier": brier_score(oof_true, calibrated),
                "oof_log_loss": log_loss_score(oof_true, calibrated),
                "oof_ece": expected_calibration_error(oof_true, calibrated),
                "oof_n": float(len(oof_true)),
                "base_rate_log_loss": log_loss_score(base_true, base_pred),
            }

            factory = dict(MODEL_FACTORIES)[selected]
            final = factory(cfg.project.random_seed)
            final.fit(X, y)
            models.append(
                TrainedModel(
                    name=selected,
                    horizon=horizon,
                    side=side.value,
                    feature_columns=columns,
                    estimator=final,
                    calibrator=calibrator,
                    metrics=metrics,
                    reliability_bins=reliability_bins(oof_true, calibrated),
                    featureset_id=fsid,
                    config_hash=cfg.config_hash(),
                    seed=cfg.project.random_seed,
                )
            )
            log_event(log, 20, "model trained", horizon=horizon, side=side.value,
                      selected=selected, oof_log_loss=round(metrics["oof_log_loss"], 4))
    return models


def _select_simplest(oof_by_model: dict[str, tuple[np.ndarray, np.ndarray]]) -> str:
    """Walk simplest -> complex; upgrade only on a material OOF improvement."""
    losses = {
        name: log_loss_score(true, pred)
        for name, (pred, true) in oof_by_model.items()
    }
    ordered = [name for name, _ in MODEL_FACTORIES if name in losses]
    selected = ordered[0]
    for challenger in ordered[1:]:
        if losses[challenger] < losses[selected] * (1.0 - COMPLEXITY_TOLERANCE):
            selected = challenger
    return selected
