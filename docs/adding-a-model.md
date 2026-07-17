# Adding a model

> Models added through this legacy guide remain research-only until registered in a V2 manifest/trial family and passed through the frozen nested promotion protocol.

Models estimate `P(net trade return > 0)` for one (horizon, side) target and
plug into the ladder in `src/edgestack/models/base.py`:

```python
def make_my_model(seed: int) -> BaseEstimator:
    return Pipeline([...])   # anything with fit / predict_proba

MODEL_FACTORIES = (
    ("base_rate", make_base_rate),
    ("logistic", make_logistic),
    ("gradient_boosting", make_gradient_boosting),
    ("my_model", make_my_model),          # position = complexity rank
)
```

## The rules the ladder enforces

1. **Order = complexity.** `_select_simplest` walks the tuple from simplest
   to most complex and upgrades only when the challenger improves
   out-of-fold log loss by at least `COMPLEXITY_TOLERANCE` (1%). If your
   model cannot beat logistic out of sample, it will not be selected — that
   is the point, not a bug.
2. **Handle the data honestly.** Feature matrices contain NaNs (warmup
   windows). Either handle NaN natively (HistGradientBoosting does) or start
   your pipeline with an imputer. Never fit preprocessing outside the
   pipeline — each fold refits everything on its training slice only.
3. **Determinism.** Accept the seed argument and pass it to every stochastic
   component.
4. **Calibration is added for you**: the winner gets an isotonic calibrator
   fit on its out-of-fold predictions; Brier/log-loss/ECE and reliability
   bins are stored in the artifact metadata.
5. **Artifacts**: models are pickled with a SHA-256 checksum and a JSON
   metadata sidecar; loading verifies both before unpickling. Optional heavy
   dependencies (e.g. LightGBM) must sit behind an extra and degrade
   gracefully when absent.

## Deep learning?

The default stance is interpretable tabular models. A neural model is
welcome only as the last rung of the ladder, behind an extra, subject to the
same 1%-better-OOF rule — on a few thousand daily observations it rarely
earns its place.
