# Data model

> `EdgeStatus.VALIDATED` is a retained legacy lifecycle/database value, not authorization for V2 portfolio weight. V2 promotion is represented separately by immutable `PromotionDecisionV2` records.

## Canonical daily bar (`edgestack.data.schemas`)

One row per (symbol, session): `symbol, date, open, high, low, close, volume,
adj_close`. Raw prices are used for fills; `adj_close` (split+dividend
adjusted, NaN when the provider cannot supply it) for total-return math.
Structural validation rejects duplicate (symbol, date) rows, non-positive
prices, negative volume and OHLC violations (`high >= max(open, close, low)`,
`low <= min(open, close, high)`).

## Labels (`edgestack.labels`)

Forward-return label for horizon `h`, signal at close of session `T`,
execution delay `d` (default 1):

```
entry  = open(T + d)
exit   = open(T + d + h)
gross  = exit / entry - 1
label_end = session date of the exit
```

Triple-barrier labels place `up = entry + k_up * ATR(T)` and
`down = entry - k_dn * ATR(T)` (ATR measured at signal time, causal), walk the
window bar by bar, fill gaps at the open, resolve double touches to the stop
(conservative for longs), and fall back to a time exit. `label_end` is the
actual exit session — this is what makes interval purging exact.

## Edge (`edgestack.types.Edge`)

Frozen Pydantic model, persisted per version in DuckDB with scalar stats as
native columns and the full payload as JSON:

- **identity**: id, name, rule AST (JSON-serializable conditions over
  registered features), family, direction, holding horizon;
- **stats**: n, effective n, gross/net means, hit rates, VaR/ES, drawdown,
  Sharpe/Sortino, p/q values, shrinkage posterior + credible interval,
  block-bootstrap CI, deflated Sharpe;
- **robustness**: fold stability, regime breakdown, cost-scenario survival,
  decay and OOS scores;
- **lifecycle**: status, discovery/validation windows, failure reasons, batch
  and experiment references.

Statuses: `CANDIDATE, RESEARCH_ONLY, VALIDATED, ACTIVE, DEGRADED, SUSPENDED,
RETIRED, REJECTED`. Transitions are appended to `edge_events`; the current
status is a view over the latest event.

## Rule AST (`edgestack.types.Condition`)

```json
{"kind": "all", "conditions": [
  {"kind": "predicate", "feature": "rsi_14_pctile", "op": "<", "quantile": 0.1},
  {"kind": "predicate", "feature": "above_sma200", "op": "==", "value": 1.0}
]}
```

`quantile` predicates never carry a concrete threshold: the threshold is
resolved through a `QuantileBinner` fitted on the training slice in play
(fold train window during validation, full pre-date history at signal time),
which is what makes one rule definition re-evaluable anywhere without leakage.

## Signal report (`edgestack.types.SignalReport`)

Per analysis date: ranked `long_candidates` and `short_candidates`
(conviction, calibrated probability + calibration flag, expected gross/net,
holding horizon, entry plan with earliest timestamp and limit zone, stop and
targets, regime context, evidence items with per-edge contributions and
q-values, warnings, plain-English explanation), plus `abstentions` — one per
non-emitted (symbol, side) with every failed gate — and universe warnings.
Persisted as JSON on disk and in the `signal_reports` table.

## Experiments and audit

`experiments`: id, kind, config hash (SHA-256 of the canonicalized resolved
config), git commit, data-manifest hash, seed, ranges, feature-set version,
cost scenario, trial count. `audit_log`: every download, feature build,
signal run, backtest, paper session — and every `test_set_accessed` unlock.
