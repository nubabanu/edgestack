# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

EdgeStack is a research and paper-trading platform for statistical edges in daily US
equities. It never places live orders and its output is not investment advice.

## Agent toolkit — use this before reading internals

`scripts/agent_toolkit.py` wraps the whole platform as a JSON CLI: every subcommand
prints one machine-parseable JSON object (errors as `{"error": ...}`, exit 1). Start
any working session with it instead of re-deriving things from source:

```bash
.venv/Scripts/python scripts/agent_toolkit.py manual            # command reference
.venv/Scripts/python scripts/agent_toolkit.py status            # data/publication/watcher freshness
.venv/Scripts/python scripts/agent_toolkit.py snapshot ACN      # current indicator state
.venv/Scripts/python scripts/agent_toolkit.py analyze CTSH --entry-date 2026-11-02
.venv/Scripts/python scripts/agent_toolkit.py edge-check CTSH   # zoo rule library + survivor bar
.venv/Scripts/python scripts/agent_toolkit.py earnings ACN      # EDGAR-stamped events
.venv/Scripts/python scripts/agent_toolkit.py universe --as-of 2020-03-31
.venv/Scripts/python scripts/agent_toolkit.py research-summary  # every campaign's verdict
.venv/Scripts/python scripts/agent_toolkit.py watch             # tranche trigger status
```

`edge-check` applies the corrupted-series screen and the exact survivor bar for you;
`analyze` goes through the canonical bundle with an audited guard unlock. Relay the
embedded caveats whenever you surface results.

## Commands

Windows; the venv interpreter is `.venv/Scripts/python.exe` (Python 3.12 required).

```bash
.venv/Scripts/python -m pip install -e ".[dev,api,research]"   # install
.venv/Scripts/python -m ruff check src tests scripts            # lint
.venv/Scripts/python -m ruff format --check src tests scripts   # format check
.venv/Scripts/python -m mypy src/edgestack                      # types
.venv/Scripts/python -m pytest                                  # tests (network tests excluded by default)
.venv/Scripts/python -m pytest tests/unit/test_x.py::test_y     # single test
.venv/Scripts/python -m pytest -m statistical                   # slower statistical acceptance tests
.venv/Scripts/python -m pytest -m network                       # tests needing internet
```

The `Makefile` mirrors these as `install / lint / format-check / type / test / package /
test-all` (POSIX make; on Windows run the commands directly). Tests live under
`tests/{unit,integration,regression,statistical}`.

Other entry points:

```bash
edgestack api serve --config configs/live.yaml                  # FastAPI (CLI installed via pyproject [project.scripts])
.venv/Scripts/python scripts/nightly.py --config configs/live.yaml            # nightly publication
.venv/Scripts/python scripts/nightly.py --config configs/live.yaml --skip-data-update --skip-features   # offline re-publish
cd android && ./gradlew testDebugUnitTest assembleDebug         # Android app (JDK 17/21 only — JDK 24 breaks Gradle)
```

## Big-picture architecture

Two worlds coexist and must not be confused:

1. **The canonical V2 pipeline** (`src/edgestack/`): providers → `DataCatalog` (per-symbol
   Parquet in `data/curated/prices/` + one DuckDB for experiments/edges/audit) → feature
   engine → labels → rule discovery → validation → models → scoring → one immutable
   `CanonicalRecommendationBundleV2`. The API, Android app, dashboard, and paper simulator
   only ever read that bundle. Until a sleeve passes every promotion gate the actionable
   portfolio is the fixed `baseline-diversified-v1` policy (SPY/TLT/SHY/GLD 25% each).
   `docs/architecture.md` and `docs/recommendation-engine-v2.md` are the authoritative
   references.

2. **Standalone research campaigns** (`scripts/`): strategy_zoo, seasonality_scan,
   allocation_zoo, unified_book, execution_sensitivity, rules_campaign, tranche_*, etc.
   These are self-contained studies whose findings are summarized in
   `docs/strategy-zoo.md`. Their results — including everything touching 2024–2026 — are
   **previously-accessed history and can never promote a V2 sleeve**.

Structural safety rails you must not weaken:

- **TestPeriodGuard** (`src/edgestack/data/catalog.py`): the final test period is
  unreachable without an audited `catalog.guard.unlock(reason=...)` key; every unlock is
  logged. The integration suite asserts a full pipeline run performs zero unlocks.
- **Feature causality is engine-enforced** (`features/registry.py`): availability lags are
  applied by shifting; no same-close fills anywhere (`execution_delay >= 1`).
- Every evaluated rule is persisted with a batch id so multiplicity corrections (FDR,
  deflated Sharpe) use the honest trial count.
- Publications are atomic: outputs are written to a temp dir, renamed to
  `artifacts/recommendations/runs/<content-hash>`, then `current.json` is pointer-swapped.
  Use `edgestack.data.catalog.atomic_write_bytes` for any artifact writes.

## Research-script conventions (scripts/)

- Signal at close *t* earns session *t+1*; 2 bps per unit of exposure change (5 bps for
  monthly/allocation studies).
- Splits: dev 1999–2015 (or 2011–2017 for single stocks), val 2016–2023, holdout 2024+ —
  the holdout is previously accessed; say so in any new writeup.
- Survivor bar: Sharpe ≥ buy-and-hold in **all** splits AND pooled Newey–West alpha t ≥ 2
  (`edgestack.validation.advanced_tests.newey_west_alpha`,
  `edgestack.validation.metrics.sharpe_ratio`).
- Universe scans over `data/curated/prices/*.parquet` MUST screen out corrupted series
  first: some legacy tickers (CBE, TIE…) contain $0.01 shell quotes or two instruments
  mixed in one file. Minimum screen: drop any symbol whose max single-day |return| > 200%,
  plus per-window price ≥ $1 and median dollar-volume ≥ $1M filters. Yahoo coverage is
  survivorship-biased (no delisted names) — say so when quoting recovery statistics.
- Curated bar schema: `symbol, date, open, high, low, close, volume, adj_close`
  (OHLC split-adjusted; `adj_close` split+dividend adjusted).

## Nightly automation

Windows Task Scheduler job **"EdgeStack Nightly"** runs `scripts/nightly.bat` daily at
23:30, which runs `scripts/nightly.py` (price refresh for the whole catalog → quality gates
→ feature build → news → canonical publication) and then **always** runs
`scripts/tranche_watch.py`, even if nightly failed. Logs: `logs/nightly_YYYYMMDD.log` and
`logs/tranche_watch.log`.

Reliability rails (added 2026-07-21 after the 2026-07-20 outage, which was a double
fault: the intermittent fatal `PyEval_SaveThread` GIL crash in the Yahoo fetch, plus
cache poisoning — a pre-close daytime catch-up run cached the `..end=today` price range
without the close bar, and the 30-day TTL served that stale range to every retry.
`range_cache_fresh` in `data/providers/base.py` now refuses any cached daily range not
written after the end date's US close): the price refresh uses the `yahoo_stooq` failover provider
(`data/providers/failover.py`; Stooq fills per-symbol gaps with `adj_close=close` until
Yahoo restates), each 25-symbol batch runs in a subprocess via
`edgestack.data.fetch_worker` with one fresh-process retry (`--no-fetch-isolation` to
debug in-process), and every chain stage is recorded through `scripts/run_stage.py` into
`artifacts/nightly_status.json`. `scripts/nightly_status_report.py` ends the chain: it
prints the stage summary, alerts via Telegram/toast on failure, and exits 0 (ok) /
1 (auxiliary stage failed) / 2 (core stage failed) so Task Scheduler sees the truth.

### Tranche watch program (ACN / CTSH / EPAM / SPY)

`scripts/tranche_watch.py` monitors staged buy-entry triggers from the 2026-07 buy-timing
research: T1 dip, T2 repair (alert carries a 61%-whipsaw caution), T3 200-DMA trend, REL
peer-basket relative-strength recross, SECTOR breadth, CAL seasonal windows, WINDOW
(post-earnings window open — the positive counterpart of the pre-earnings blackout),
plus earnings-blackout suppression (Yahoo quoteSummary needs the cookie+crumb flow —
already implemented there) and REVIEW/exit checks for positions recorded in
`artifacts/tranche_positions.json`. Alerts go to BOTH channels: Windows toast
(`scripts/notify_toast.ps1`, requires powershell.exe 5.1, not pwsh) and Telegram
(`scripts/notify_telegram.py`; config in gitignored `artifacts/telegram_config.json`,
setup steps in its docstring; graceful no-op when unconfigured). Other outputs:
`artifacts/tranche_status.html` dashboard, `artifacts/tranche_watch.json`, dedupe state
in `artifacts/tranche_watch_state.json` (delete to re-arm alerts).

A 0-100 **GO score** (hand-set weights, `go_score()`) is computed per symbol and shown
in the dashboard/log — **display-only**: `agent_toolkit.py go-backtest SYM` FAILED its
validation gate on all three names (2026-07-19: the 60+ bucket does not beat
unconditional forward returns), so `GO_ALERTS_ENABLED = False`. Do not enable without
a fresh go-backtest PASS; the individual triggers remain the actionable alerts.

A market-level **WIND score** (`market_wind_score()`, audit:
`scripts/confluence_timing.py`) is logged and published in the status JSON as a V2
`DESCRIPTIVE_ONLY` / `NO_ACTION` payload. Its five neutral price/calendar components
are not observed mechanism flows. Corrected verdict: strict monotonicity and selective
entry FAILED; the old QQQ daily switching veto passed historically (t=3.13) but does
not validate delaying a planned purchase. WIND never fires alerts, creates tickets,
sizes exposure, or gates a buy. Full source-claim audit:
`docs/market-cycle-claim-audit.md`.

The former 3x and 20x WIND paper experiments are RETIRED. `scripts/wind_paper_book.py`
is now an idempotent retirement handler: it may settle only the position already
pending at retirement, opens nothing new, then writes a content-addressed archive.
`scripts/wind_execution_shadow.py` replaces it as the evidence collector: an unlevered,
fixed-notional paired-fill audit comparing a negative-score session's open with a
mandatory fill exactly one XNYS session later. It is reviewable only after 252
prospective sessions and 30 completed events and can never promote automatically.

`scripts/intraday_precheck.py` (Task Scheduler "EdgeStack Precheck", weekdays 21:45
local via `scripts/precheck.bat`, logs to `logs/precheck.log`) fetches today's partial
bar 15 min before the US close and sends a provisional-T1 heads-up so entries can be
planned the same evening; the nightly watcher on the real close stays canonical.

### Oil surge watch (CL=F shock reaction, added 2026-07-23)

`scripts/oil_surge_watch.py` REACTS to crude price shocks — it predicts no events.
SHOCK alerts (CL=F >= +4% day or >= +8%/5 sessions, descriptive facts, geopolitical
headlines quoted as corroboration only) fire live; while a 10-session surge regime is
armed, pullbacks >= 1.5% from the post-shock closing high fire DIP alerts. Two runs:
Task Scheduler **"EdgeStack Oil Surge Watch"** polls `--intraday` every 30 min
(08:00-23:00 via `scripts/oil_watch.bat`, provisional live-bar alerts), and
`nightly.bat` runs `--eod` on curated closes (canonical; state in
`artifacts/oil_surge_state.json`, log `logs/oil_surge_watch.log`, status via
`agent_toolkit.py oil-surge`). DIP alerts are **DISPLAY-ONLY**:
`scripts/oil_shock_study.py` (24 trials, 2000-2026) FAILED — best cell E2/R3@20d
t=1.84 < 2 (docs/strategy-zoo.md) — so `DIP_TICKETS_ENABLED = False`; enabling
requires a fresh study PASS naming R3 in `artifacts/oil_shock_study_verdict.json`
AND a human flip, at which point paper tickets would size from tranche_plan.json key
`"BNO": {"OILDIP": eur}` into `artifacts/paper_oil.json`. BNO is a data proxy only —
PRIIPs blocks US ETFs for EU retail; a real purchase is a UCITS Brent ETC. The
SHIPPING_DISRUPTION hard veto in `edgestack oil check` (fresh CFD entries) is
intentionally unchanged and opposite in polarity to this dip program.

SPY is watched too (T1 = calm regime AND any dip — the best-compounding entry in the
research; T3 alerts on the reclaim cross only, REL/earnings skipped for the index).

**Order tickets + paper autopilot**: `artifacts/tranche_plan.json` (auto-created
template, gitignored) maps (symbol, trigger) -> euro size; alerts append
"-> ORDER: buy ~EUR X at next open" tickets (ASCII only — the cp1252 console dies on
Unicode arrows), and `paper_autopilot()` executes every ticketed alert ON PAPER at the
actual next-open adjusted price into `artifacts/paper_tranche.json`, with nightly P&L
lines and a dashboard section. Live orders remain human — repo covenant; German retail
brokers have no official trading APIs, and the paper book exists to measure what full
automation would have done.

Related one-offs on this machine: Windows Startup launcher
(`%APPDATA%\...\Startup\EdgeStackAPI.bat`) keeps `scripts/api_serve.bat` (watchdog,
0.0.0.0:8000) alive for the Android app; firewall rule "EdgeStack API" exists;
`releases/EdgeStack-v1.8.apk` is the current companion build.

`scripts/tranche_policy_backtest.py` is the companion cohort study; its headline:
staged entry is insurance, not alpha, and lump-sum recovery odds depend heavily on
whether the market itself was crashed at episode start.

## Data

`data/`, `artifacts/`, `logs/`, and `.venv/` are gitignored and regenerable. Rebuild
prices via the nightly run or
`edgestack data download --symbols "SPY,TLT,SHY,GLD" --start 2015-01-02 --config configs/live.yaml`.
Yahoo free-data limits apply (60m bars ≤ 729 days, 15m ≤ 59 days).

Free point-in-time sources wired into the nightly (added 2026-07):

- **EDGAR earnings events** — `scripts/edgar_earnings.py` ingests 8-K item-2.02
  acceptance datetimes plus XBRL quarterly EPS (keyless, ~10 req/s fair use, declared
  User-Agent) into `data/curated/events/{SYM}.parquet`. `scripts/pead_study.py` computes
  seasonal-random-walk SUE and post-announcement drift from them; the 2026-07 mega-cap
  run found **no PEAD edge** (t ≈ 0.5) — the honest test needs small/mid caps, whose
  delisted price history has no free source.
- **Universe forward snapshots** — `scripts/universe_snapshot.py` freezes today's exact
  S&P membership (via `data/universe_pit.py`) daily into
  `data/curated/universe_snapshots/`; forward panels are survivorship-free by
  construction, the bias only lives in the past.
- **Intraday forward archive** — `scripts/intraday_collector.py` pulls trailing 60m/15m
  bars nightly for the watch set into `data/curated/intraday/`; free backfill no longer
  exists at scale, so the archive grows forward from 2026-07.

## Style

Ruff (line length 100, py312, rules E/F/W/I/UP/B/C4/SIM/RUF) and mypy are CI gates —
run both before committing. Scripts follow a common shape: module docstring stating
purpose *and honest caveats*, constants, small pure functions, `main() -> int`,
`raise SystemExit(main())`. Keep the caveat discipline: any new result that touches
2024–2026 must be labeled previously-accessed, and any bounded scan should log what it
dropped.
