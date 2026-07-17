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

Requires Python 3.12 and, for Android, JDK 17 or 21. JDK 24 breaks Gradle's
test task ("Type T not present"); point `JAVA_HOME` at 17/21 before building.

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

On Windows, point `JAVA_HOME` to a JDK 17/21 installation before invoking `gradlew.bat`.

## Run the nightly publication

```bash
.venv/Scripts/python scripts/nightly.py --config configs/live.yaml
```

For a data-prepared offline check:

```bash
.venv/Scripts/python scripts/nightly.py --config configs/live.yaml --skip-data-update --skip-features
```

The orchestrator fails on data-quality, artifact, schema, or checksum errors. It writes all outputs to a temporary directory, renames the completed run to `artifacts/recommendations/runs/<content-hash>`, and then atomically replaces `artifacts/recommendations/current.json`. Failure before the pointer swap leaves the prior publication current.

Fail-fast applies to the symbols tonight's publication consumes (baseline policy plus configured universe). Delisted point-in-time catalog members that return empty series are skipped with a warning, and the pre-flight quality gate covers the required symbols; the full research catalog remains covered by `edgestack data validate`. The publication step separately re-validates its own panel and requires 252 aligned return sessions for the policy ETFs — backfill them once with:

```bash
edgestack data download --symbols "SPY,TLT,SHY,GLD" --start 2015-01-02 --config configs/live.yaml
```

## Public interface

- `GET /recommendations/latest` returns the verified canonical bundle.
- `POST /recommendations/preview` accepts `RiskProfileV2`, optional `RiskStateV2`/equity override, and an optional reset request. It only recalculates sizing and stress; it cannot research, promote, select, or persist.
- `POST /instruments/analyze` accepts a stock/ETF ticker or commodity name/proxy and an optional intended-entry timestamp. It returns day/week/month/year best and worst historical windows, tailwinds, headwinds, counter-effects, news context, current-year observations, and explicit abstentions. Only compatible frozen promoted timing artifacts can make a window actionable or create a directional rating.
- `POST /instruments/recheck` compares a prior analysis with the latest canonical inputs and reports whether the selected timing still holds or a higher-ranked alternative emerged.
- `POST /instruments/pattern-leaders` scans 1–50 explicitly supplied symbols and returns a research-only ranking. It never promotes the symbol/slot search.
- `GET /sniper/latest` returns the version-bound staged sniper shadow plan. `POST /sniper/preview` changes only account equity, modeled loss budget, and the approved diversified vehicle; it cannot mutate canonical weights or promotion state. C1/C2 and Santa are Stage 1 shadow candidates, C3/C4 are non-initiating overlays, and Stage 2 remains blocked until its data and promotion prerequisites exist.
- `GET /edges` lists the validated edge catalog with lifecycle status, net mean return, q-value, deflated Sharpe, and sample size; `GET /edges/{edge_id}` returns one full record.
- `GET /monitoring/edges` returns the canonical monitoring payload; `GET /backtests` and `GET /backtests/{run_id}` list and fetch stored backtest runs.
- `/board`, `/picks`, `/master`, `/signals/*`, and `/candidates/*` are deprecated compatibility projections. Board rows and picks are empty; master contains canonical weights only.

Run the API with:

```bash
edgestack api serve --config configs/live.yaml
```

Hourly and 15-minute analysis are opt-in because daily bars cannot identify an intraday slot:

```bash
edgestack data intraday-download --symbols GLD,AAPL --interval 60m --start 2025-07-16 --end 2026-07-16 --config configs/live.yaml
edgestack data intraday-download --symbols GLD,AAPL --interval 15m --start 2026-05-20 --end 2026-07-16 --config configs/live.yaml
python scripts/nightly.py --config configs/live.yaml --skip-data-update --skip-features
```

Commodity words resolve to disclosed tradable proxies (`GOLD → GLD`, `OIL/WTI → USO`, `BRENT → BNO`, `SILVER → SLV`). Proxy fees, tracking error, roll yield, and trading hours remain visible warnings.

When an intended entry includes a date and time, EdgeStack separately rates its 15-minute slot, hour, weekday across every holding horizon, position within the month, and month of the year. It displays rank, cost-adjusted historical win score, a better compatible slot if available, and conditional day/week/month/year exits. Win score is a shrunk historical frequency with an ESS confidence penalty—not a promised probability of profit. Rechecks tighten from daily to six-hourly, hourly, and finally 15-minute cadence as entry approaches.

## Android companion app

The app in `android/` (package `com.edgestack.app`) is a read-only client for the PC-hosted API. It never places orders and never computes signals on the device; offline it displays the last server result or the bundled canonical seed.

Tabs:

- **Portfolio** — canonical base weights, promoted sleeves, zero-weight watchlist, freshness.
- **Calendar** — NYSE month grid with turn-of-month windows, session math, and per-day historical tailwind shading projected from the last instrument analysis (weekday, week-of-month, and month slots; research only).
- **Risk** — server-calculated leverage, binding constraint, stress and financing, canonical targets.
- **Analyze** — instrument timing: best/worst windows per horizon, chosen-time ratings, exit maps, tailwinds/headwinds, news context, automatic rechecks.
- **Sniper** — staged shadow plan with equity/loss-budget preview; paper only.
- **Edges** — validated edge catalog with search and status filters, monitoring health, backtest runs.
- **Trades** — canonical paper account state, plus a personal position tracker with delayed device quotes and unrealized P&L (device-side display only; never feeds signals).
- **Settings** — server URL, risk-sizing profile, connection test, sync.

Build and install:

```bash
cd android
./gradlew assembleDebug     # output: app/build/outputs/apk/debug/app-debug.apk
./gradlew assembleRelease   # R8-minified and release-signed (see below)
```

Release signing reads `RELEASE_STORE_FILE`, `RELEASE_STORE_PASSWORD`, `RELEASE_KEY_ALIAS`, and `RELEASE_KEY_PASSWORD` from the gitignored `android/local.properties`; generate a keystore once with `keytool -genkeypair -keystore keystore/edgestack-release.jks -alias edgestack -keyalg RSA -validity 10950`. Without these properties the release build is unsigned. Debug and release signatures differ — switching between them on a device requires uninstalling first.

Connect a phone (same Wi-Fi as the PC):

1. Allow the port once, in an elevated shell: `New-NetFirewallRule -DisplayName "EdgeStack API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`
2. Serve the API on all interfaces: `edgestack api serve --host 0.0.0.0 --port 8000 --config configs/live.yaml`
3. In the app's Settings tab set `http://<PC-LAN-IP>:8000`, then Test and Sync. The Android emulator reaches the host at `http://10.0.2.2:8000`.

Away from home, install Tailscale on both devices and use the PC's Tailscale IP instead; the app permits cleartext HTTP for LAN/Tailscale use.

Refresh the bundled offline seed after a new publication:

```bash
.venv/Scripts/python scripts/export_mobile_bundle.py
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
