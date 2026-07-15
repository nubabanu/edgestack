# Adding a feature

Features live in `src/edgestack/features/` and register themselves:

```python
from edgestack.features.registry import feature
from edgestack.types import Family


@feature("my_pullback_depth", Family.MEAN_REVERSION, inputs=("close",),
         min_history=21)
def my_pullback_depth(df: pd.DataFrame) -> pd.Series:
    """Close relative to its trailing 20-session high (depth of pullback)."""
    return df["close"] / df["close"].rolling(20).max() - 1.0
```

The function receives **one symbol's history** indexed by session date, with
raw bars (`open, high, low, close, volume, adj_close`), calendar facts
(`is_friday, is_turn_of_month, ...`) and the broadcast benchmark close
(`bench_close`). Return a Series aligned to the input index.

## Rules of the road

1. **Be causal**: the value stored at date D may use data up to D's close.
   Rolling/shift operations with positive lags are naturally causal.
2. **Need future bars to confirm?** Declare `availability_lag=k` and look
   ahead at most `k` bars — the engine shifts your output by `k` so the
   stored value is observable when it appears. See `pivot_high_2` in
   `structure.py`.
3. **Cross-sectional features** (`cross_sectional=True`) receive one
   date-slice at a time (indexed by symbol) — normalizing over history is
   structurally impossible. They may consume previously computed time-series
   features; declare them in `inputs`.
4. **No fitted state inside features.** Quantile thresholds, scaler moments
   etc. belong in `TrainFitted` transformers (`features/binning.py`) fit per
   fold — a feature that fits on its full input leaks.
5. Booleans should be returned as floats (0.0/1.0); outputs are coerced to
   float64.
6. Add the feature to `CONTINUOUS_RULE_FEATURES` or `BINARY_RULE_FEATURES` in
   `discovery/candidate_generation.py` if discovery should search it.

## Verification (automatic)

`tests/regression/test_lookahead.py` iterates the ENTIRE registry: it builds
features, mutates every bar after a cutoff date, rebuilds, and requires all
values at or before the cutoff to be bit-identical — and separately checks
truncation invariance. Register the feature and it is covered; a leaking
feature fails with its name in the assertion message. Add a value-correctness
spot check in `tests/unit/test_features.py` for the math itself.
