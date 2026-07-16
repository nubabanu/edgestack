# EdgeStack Recommendation Engine V2

EdgeStack is a research and paper-trading system. It does not place live orders and its output is not investment advice. Leverage can cause losses greater than invested capital.

The authoritative output is one immutable `CanonicalRecommendationBundleV2`. The API, Android app, dashboard, paper simulator, monitoring view, and one-release compatibility endpoints read or transform that bundle. Legacy `VALIDATED` records and the previously accessed 2024–2026 period are not V2 promotion evidence.

## Initial behavior

Until a sleeve passes every V2 promotion gate, the actionable unlevered portfolio is the tracked `baseline-diversified-v1` policy:

- SPY 25%
- TLT 25%
- SHY 25%
- GLD 25%
- quarterly next-open rebalance, or earlier at five-percentage-point drift
- adjusted-open total-return research, explicit dividends/splits, 5 bps one-way baseline costs, and participation impact

Old stock ideas are zero-weight watchlist records. A stock sleeve additionally needs 252 frozen prospective sessions and effective sample size of at least 100 before it can receive portfolio weight.

## Install and verify

Requires Python 3.12 and, for Android, JDK 17.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,api,research]"
.venv/Scripts/python -m ruff check src tests scripts
.venv/Scripts/python -m ruff format --check src tests scripts
.venv/Scripts/python -m mypy src/edgestack
.venv/Scripts/python -m pytest
.venv/Scripts/python -m build
.venv/Scripts/python -m twine check dist/*
```

Android:

```bash
cd android
./gradlew testDebugUnitTest assembleDebug
```

On Windows, point `JAVA_HOME` to a JDK 17 installation before invoking `gradlew.bat`.

## Run the nightly publication

```bash
.venv/Scripts/python scripts/nightly.py --config configs/live.yaml
```

For a data-prepared offline check:

```bash
.venv/Scripts/python scripts/nightly.py --config configs/live.yaml --skip-data-update --skip-features
```

The orchestrator fails on data-quality, artifact, schema, or checksum errors. It writes all outputs to a temporary directory, renames the completed run to `artifacts/recommendations/runs/<content-hash>`, and then atomically replaces `artifacts/recommendations/current.json`. Failure before the pointer swap leaves the prior publication current.

## Public interface

- `GET /recommendations/latest` returns the verified canonical bundle.
- `POST /recommendations/preview` accepts `RiskProfileV2`, optional `RiskStateV2`/equity override, and an optional reset request. It only recalculates sizing and stress; it cannot research, promote, select, or persist.
- `/board`, `/picks`, `/master`, `/signals/*`, and `/candidates/*` are deprecated compatibility projections. Board rows and picks are empty; master contains canonical weights only.

Run the API with:

```bash
edgestack api serve --config configs/live.yaml
```

## User risk sizing

Alpha selection is profile-independent. The user can set target volatility, maximum drawdown, maximum gross leverage from 0–5×, funding spread, per-stock cap, and sector cap. Effective leverage is the minimum of independent volatility, one-day loss, path drawdown, liquidity, concentration, freshness, funding, drawdown-state, user, and system limits.

Exposure increases by no more than 0.25× per distinct session and reductions are immediate. Market staleness cannot introduce or enlarge a position. DGS3MO older than five US business days disables increases above 1×. At the drawdown limit the system targets cash and latches until 20 sessions, recovery, fresh compatible data, acceptable stress, healthy monitoring, and an explicit reset.

```bash
edgestack risk reset-latch --config configs/live.yaml
```

## Research promotion

Promotion uses expanding annual outer folds with inner-only selection, cross-fitted calibration, session-aggregated returns, 20-session stationary blocks, 2,000 deterministic bootstrap replications, complete-family SPA, and StepM. At least five valid outer folds are required.

A sleeve must have positive fully costed compounded return in at least 70% of valid folds, SPA-consistent `p ≤ 0.05`, StepM inclusion, positive one-sided 95% paired block-bootstrap Sharpe lower bounds versus training-risk-matched SPY and the diversified baseline, and survival under conservative/stress execution, delayed fills, liquidity, participation, and 200/400/800 bps financing scenarios.

Different horizons have separate labels, artifacts, calibrators, statistics, and sleeve identities. They meet only as realized daily exposures in the portfolio covariance and optimizer.

## Documentation

- [V2 architecture and operations](docs/recommendation-engine-v2.md)
- [Migration and claim withdrawal](docs/v2-migration.md)
- [Statistical validation](docs/statistical-validation.md)
- [Backtesting and execution](docs/backtesting.md)
- [Data model](docs/data-model.md)

Historical campaign reports are retained as research provenance. Their 2024–2026 results are previously accessed, their manual search histories may be incomplete, and none can promote a V2 sleeve without a new frozen manifest and prospective evidence.
