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

## Run the continuous Edge Factory

The factory turns missing evidence into resumable work instead of treating
`INSUFFICIENT` as a terminal conclusion. Copy `.env.example`, add any free Alpaca
and FRED credentials you own, declare a real SEC user-agent, then run:

```bash
edgestack research status --config configs/live.yaml
edgestack research queue --config configs/live.yaml
edgestack research run --config configs/live.yaml
edgestack research run --once --config configs/live.yaml
edgestack research pause --config configs/live.yaml
edgestack research resume --config configs/live.yaml
edgestack research proposal-import configs/proposal.example.json --config configs/live.yaml
edgestack research proposals --config configs/live.yaml
edgestack research factor-audit factor_panel.parquet --factor-name breadth_trend
edgestack research strange-edges --config configs/default.yaml
```

It runs below normal priority with four isolated worker processes and a 75 GB combined data/artifact
cap. Priority is PIT universe repair, prospective shadows, evidence acquisition,
then frozen evaluation. Monthly cohorts are content-addressed and capped at 1,500
pre-registered trials. Long free-tier intraday gaps become `BLOCKED_FREE_TIER`
when delayed Alpaca SIP is not configured; Yahoo is never misrepresented as a
source of decade-long minute history. Other viable campaigns continue.

The monthly universe reverse-walks the public S&P 500 change log, ranks members
on trailing 60-session dollar volume, keeps one-minute bars for ETFs and the top
25 stocks, and five-minute bars for ranks 26–100. Fifteen- and sixty-minute bars
are deterministic derivatives. Earnings, ALFRED vintages, and FINRA off-exchange
short-sale volume are stored with raw hashes, normalized Parquet, provenance,
coverage, quality, and limitations. FINRA volume is not short interest.

Candidate returns include cash yield and every waiting day. Promotion additionally
requires positive lower-confidence net log-growth versus buy-now SPY, risk-matched
SPY, the diversified baseline, and financed SPY at the same realized risk. Sizing
uses a shrinkage quarter-Kelly ceiling inside the existing minimum-of-independent-
constraints overlay; 5× remains a hard ceiling, never a target. Only a healthy
promoted sleeve whose frozen next-open signal is currently active can affect
canonical paper. This repository still ships no live adapter.

Fail-fast applies to the symbols tonight's publication consumes (baseline policy plus configured universe). Delisted point-in-time catalog members that return empty series are skipped with a warning, and the pre-flight quality gate covers the required symbols; the full research catalog remains covered by `edgestack data validate`. The publication step separately re-validates its own panel and requires 252 aligned return sessions for the policy ETFs — backfill them once with:

```bash
edgestack data download --symbols "SPY,TLT,SHY,GLD" --start 2015-01-02 --config configs/live.yaml
```

## Public interface

- `GET /recommendations/latest` returns the verified canonical bundle.
- `GET /research/overview`, `/research/campaigns`, and
  `/research/campaigns/{campaign_id}` return worker, queue, evidence-gap, and
  frozen campaign state.
- `GET /research/proposals`, `/research/proposals/{proposal_id}`, and its
  `/attempts` and `/audit` subresources expose finite external hypotheses and
  their complete search history without executing agent-generated code.
- `GET /data/coverage` reports source/feed coverage, quality, hashes, and limitations.
- `GET /paper/strategies` returns independent shadow books and benchmark-relative results.
- `GET /recommendations/growth-diagnostics` returns the canonical action, log-growth
  range, recommended leverage, and every binding constraint. Non-promoted evidence
  cannot produce an entry action.
- `GET /market/quotes?symbols=SPY,ACN` returns normalized indicative snapshots from the configured live-quote chain. `GET /market/providers/health` reports configuration, cooldown, freshness, and cache state without exposing credentials. These endpoints cannot feed canonical signals or paper fills.
- `POST /recommendations/preview` accepts `RiskProfileV2`, optional `RiskStateV2`/equity override, and an optional reset request. It only recalculates sizing and stress; it cannot research, promote, select, or persist.
- `POST /instruments/analyze` accepts a stock/ETF ticker or commodity name/proxy and an optional intended-entry timestamp. It returns day/week/month/year best and worst historical windows, tailwinds, headwinds, counter-effects, news context, current-year observations, and explicit abstentions. Only compatible frozen promoted timing artifacts can make a window actionable or create a directional rating.
- `POST /instruments/recheck` compares a prior analysis with the latest canonical inputs and reports whether the selected timing still holds or a higher-ranked alternative emerged.
- `POST /instruments/pattern-leaders` scans 1–50 explicitly supplied symbols and returns a research-only ranking. It never promotes the symbol/slot search.
- `POST /oil/decision` evaluates a timestamped manual eToro `OIL` bid/ask observation against the canonical bundle, free oil-market context, three friction assumptions, event vetoes, and normalized leverage stress. Its schema is paper-only: `actionable=false`, canonical weight is zero, and order, notional, and quantity fields do not exist.
- `GET /sniper/latest` returns the version-bound staged sniper shadow plan. `POST /sniper/preview` changes only account equity, modeled loss budget, and the approved diversified vehicle; it cannot mutate canonical weights or promotion state. C1/C2 and Santa are Stage 1 shadow candidates, C3/C4 are non-initiating overlays, and Stage 2 remains blocked until its data and promotion prerequisites exist.
- `GET /edges` lists the validated edge catalog with lifecycle status, net mean return, q-value, deflated Sharpe, and sample size; `GET /edges/{edge_id}` returns one full record.
- `GET /monitoring/edges` returns the canonical monitoring payload; `GET /backtests` and `GET /backtests/{run_id}` list and fetch stored backtest runs.
- `/board`, `/picks`, `/master`, `/signals/*`, and `/candidates/*` are deprecated compatibility projections. Board rows and picks are empty; master contains canonical weights only.

Run the API with:

```bash
edgestack api serve --config configs/live.yaml
```

### Indicative live quotes

Copy `.env.example` to the gitignored `.env` and add any tokens you own. The
default REST snapshot order is Finnhub, Twelve Data, Alpaca, then the no-key
Yahoo fallback; unconfigured providers are skipped. Accounts and keys must be
created with the vendors themselves:

- [Finnhub API documentation](https://finnhub.io/docs/api)
- [Twelve Data API documentation](https://twelvedata.com/docs)
- [Alpaca Market Data documentation](https://docs.alpaca.markets/us/docs/about-market-data-api)

```bash
python scripts/agent_toolkit.py quote SPY,ACN
python scripts/agent_toolkit.py quote SPY --provider alpaca
curl "http://127.0.0.1:8000/market/quotes?symbols=SPY,ACN"
curl "http://127.0.0.1:8000/market/providers/health"
```

The gateway caches quotes for five seconds and marks active-session observations
older than 120 seconds as stale. Alpaca free data is labeled `IEX_ONLY`; Yahoo is
labeled unofficial with unknown latency. Provider plan limits can change, so the
gateway reacts to authentication errors and HTTP 429/5xx responses instead of
encoding marketing allowances. Alpha Vantage is not in the live chain because
its free US `GLOBAL_QUOTE` defaults to end-of-day data.

Hourly and 15-minute analysis are opt-in because daily bars cannot identify an intraday slot:

```bash
edgestack data intraday-download --symbols GLD,AAPL --interval 60m --start 2025-07-16 --end 2026-07-16 --config configs/live.yaml
edgestack data intraday-download --symbols GLD,AAPL --interval 15m --start 2026-05-20 --end 2026-07-16 --config configs/live.yaml
python scripts/nightly.py --config configs/live.yaml --skip-data-update --skip-features
```

Commodity words resolve to disclosed tradable proxies (`GOLD → GLD`, `OIL/WTI → USO`, `BRENT → BNO`, `SILVER → SLV`). Proxy fees, tracking error, roll yield, and trading hours remain visible warnings.

When an intended entry includes a date and time, EdgeStack separately rates its 15-minute slot, hour, weekday across every holding horizon, position within the month, and month of the year. It displays rank, cost-adjusted historical win score, a better compatible slot if available, and conditional day/week/month/year exits. Win score is a shrunk historical frequency with an ESS confidence penalty—not a promised probability of profit. Rechecks tighten from daily to six-hourly, hourly, and finally 15-minute cadence as entry approaches.

### eToro OIL paper gate

Refresh the oil research inputs within Yahoo's free-data limits, then republish the
canonical bundle so its data hash matches the catalog:

```bash
edgestack oil refresh --through 2026-07-18 --config configs/live.yaml
python scripts/nightly.py --config configs/live.yaml --skip-data-update --skip-features
```

The refresh uses `CL=F` as primary WTI history, `USO` as the tradable historical
proxy, and BNO/XLE/UUP/`^OVX` as non-directional context. Yahoo 15-minute history
is explicitly capped at 59 calendar days and is never eligible to promote an edge.

After eToro displays a quote, run the same calculation used by the API and Android
card. All timestamps must include an offset:

```bash
edgestack oil check \
  --intended-entry-at 2026-07-20T09:30:00-04:00 \
  --observed-at 2026-07-20T15:29:30+02:00 \
  --bid 67.90 --ask 68.10 \
  --offered-leverage 10 --modeled-leverage 10 \
  --config configs/live.yaml
```

Add `--event-flags WEEKEND_SUPPLY_ESCALATION,SHIPPING_DISRUPTION` (or
`WTI_ROLLOVER_EXPIRY`/`BROKER_MAINTENANCE`) when applicable. Crossed, stale, or
incomplete quotes fail closed. Wednesday's standard EIA petroleum-release window,
abnormal gaps, source/version staleness, and the manual flags are hard vetoes.
VWAP, opening range, momentum, trend, news, and cross-market alignment stay
descriptive unless they separately pass the repository's promotion process.

The three cost rows use the greater of the observed spread and 10/25/50 bp. The
stress grid shows 1%, 5%, and 10% adverse moves at 1×, 5×, and 10×. The 100% row
is a catastrophic scenario only; the service cannot turn it into position sizing
or an order. CLI snapshots are written immutably under
`artifacts/oil_decisions/snapshots/<content-hash>.json`.

Monday paper runbook:

1. After the broker opens, capture the displayed gap and spread; do not enter.
2. Refresh the oil inputs and republish the canonical bundle. Any source/hash mismatch blocks.
3. At 15:30 CEST (09:30 New York), record the opening observation. The regression fixture classifies this slot as weak after 25 bp costs.
4. Recheck with a newly timestamped quote every 15 minutes.
5. At 19:45 CEST (13:45 New York), a passing gate may record a prospective shadow observation and its one-hour paper exit at 20:45 CEST. It remains non-actionable and never creates a live order.

The next scheduled EIA Weekly Petroleum Status Report is Wednesday, 22 July
2026 at 16:30 CEST; the service blocks the 15-minute window on either side of
the standard 10:30 New York release time.

## Android companion app

The app in `android/` (package `com.edgestack.app`) is a read-only client for the PC-hosted API. It never places orders and never computes signals on the device; offline it displays the last server result or the bundled canonical seed.

Tabs:

- **Portfolio** — canonical base weights, promoted sleeves, zero-weight watchlist, freshness.
- **Calendar** — NYSE month grid with turn-of-month windows, session math, and per-day historical tailwind shading projected from the last instrument analysis (weekday, week-of-month, and month slots; research only).
- **Risk** — server-calculated leverage, binding constraint, stress and financing, canonical targets.
- **Analyze** — instrument timing plus a manual eToro OIL paper gate with a prominent `BLOCKED`, `OBSERVE`, or `PAPER_ONLY` card; best/worst windows, exit maps, context, and automatic rechecks remain server-derived.
- **Sniper** — staged shadow plan with equity/loss-budget preview; paper only.
- **Edge Lab** — worker/provider/storage health, exact acquisition gaps, campaign
  funnel and trials, shadow-versus-benchmark paper results, canonical growth and
  leverage diagnostics, plus the validated catalog. Server results are cached offline.
- **Trades** — canonical paper account state, plus a personal position tracker with backend-first indicative quotes, a no-key Yahoo device fallback, source/freshness labels, and unrealized P&L. Stale or market-closed quotes cannot fire stop/target notifications and never feed signals.
- **Settings** — server URL, risk-sizing profile, connection test, sync.

Build and install:

```bash
cd android
./gradlew assembleDebug     # output: app/build/outputs/apk/debug/app-debug.apk
./gradlew assembleRelease   # R8-minified and release-signed (see below)
```

Current local release: `releases/EdgeStack-v1.7.apk` (version code 8), SHA-256
`98c08f577b98a5849aa6c26c107c948a80c9f67aacbdbe0738c035bea4d56b80`.
The APK is RSA-signed with Android APK Signature Scheme v2.

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

- [Upstream research systems: adopted ideas and data boundaries](docs/upstream-research-systems.md)
- [V2 architecture and operations](docs/recommendation-engine-v2.md)
- [Opening spike/fade research campaign](docs/opening-fade-study.md)
- [Strange-edge campaign: data feasibility and frozen results](docs/strange-edges-study.md)
- [Migration and claim withdrawal](docs/v2-migration.md)
- [Statistical validation](docs/statistical-validation.md)
- [Backtesting and execution](docs/backtesting.md)
- [Data model](docs/data-model.md)

Historical campaign reports are retained as research provenance. Their 2024–2026 results are previously accessed, their manual search histories may be incomplete, and none can promote a V2 sleeve without a new frozen manifest and prospective evidence.
