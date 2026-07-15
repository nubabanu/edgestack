# Architecture

## Layers and data flow

```
providers/          synthetic | local CSV/Parquet | yahoo | stooq
   |                (ProviderMetadata declares adjustment, PIT capability,
   v                 publication delay and honest limitations)
data/catalog.py     per-symbol Parquet (atomic writes) + one DuckDB file with
                    experiments, candidates, edges, edge_events, backtests,
                    signal_reports, audit_log. ALL modeling reads go through
                    DataCatalog.load_panel(), which truncates at the final-test
                    boundary (TestPeriodGuard) unless an audited unlock key is
                    presented. There is no off switch.
   v
features/           registry of ~60 features with metadata (family, inputs,
                    min_history, availability_lag, cross_sectional). The ENGINE
                    enforces causality: availability lags are applied by
                    shifting, cross-sectional features see one date-slice at a
                    time, fitted state (quantile bins) goes through per-fold
                    TrainFitted transformers. A generic mutation test iterates
                    the whole registry.
   v
labels/             open-to-open forward returns per horizon and triple-barrier
                    outcomes. Every row carries entry_date and label_end;
                    execution_delay >= 1 is enforced (no same-close fills).
   v
discovery/          rule enumeration (quantile predicates over continuous
                    features, event predicates over binary ones, depth-2
                    context conjunctions). EVERY evaluated rule is persisted
                    with a batch id — the honest trial count for FDR/DSR.
   v
validation/         purged walk-forward folds (interval-overlap purging,
                    embargo), per-fold train-fitted binners, OOS net returns
                    under all four cost scenarios, batch FDR, deflated Sharpe,
                    stability gates -> Edge records (VALIDATED or REJECTED with
                    reasons).
   v
models/             base-rate -> logistic -> HistGradientBoosting ladder; a
                    complex model must beat the simpler one by >=1% OOF log
                    loss. Isotonic calibration on out-of-fold predictions.
                    Artifacts: pickle + SHA-256 + metadata, verified pre-load.
   v
scoring/            signal engine: match edges on the analysis date, aggregate
                    with family caps, model probability + disagreement,
                    conviction, gates -> SignalCandidate or Abstention.
   v
backtest/ paper/    session-driven engine and paper sessions share the same
api/ dashboard/     FillSimulator; read-only FastAPI; minimal Streamlit page.
```

## Key structural safety decisions

1. **TestPeriodGuard** (`data/catalog.py`): the final test period is
   unreachable without `catalog.guard.unlock(reason=...)`, which writes a
   `test_set_accessed` audit row *before* yielding a single-use key. The
   integration suite asserts a full pipeline run produces zero such rows.
2. **Feature causality is engine-enforced** (`features/registry.py`), not a
   convention: see [adding-a-feature.md](adding-a-feature.md).
3. **Trial-count honesty** (`discovery/candidate_generation.py`): validation
   refuses candidates without a batch id; the batch is the trial count.
4. **Event-sourced lifecycle** (`discovery/edge_store.py`): edge status is the
   latest `edge_events` row; nothing is updated in place, so the audit
   history of every edge — including retired ones — is complete.
5. **Execution invariants live in the code path** (`execution/fills.py`):
   a fill outside the bar range raises, in production and in tests alike.

## Storage layout

```
data/curated/prices/<SYMBOL>.parquet    canonical bars (raw + adj_close)
data/features/features.parquet          feature dataset (symbol, date, ...)
data/reports/signals_<date>.json        machine-readable signal reports
data/reports/backtests/<run_id>/        report.json + report.html
data/cache/<provider>/                  provider response caches
artifacts/edgestack.duckdb              metadata + audit tables
artifacts/models/model_h<h>_<side>.*    model pickle + sha256 + meta.json
artifacts/paper/state.json              paper-trading state
```

Every experiment row records config hash, git commit, data-manifest hash,
seed, date ranges, cost scenario and trial count — reruns are reproducible
and attributable.
