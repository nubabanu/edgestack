# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

EdgeStack is a research and paper-trading platform for statistical edges in daily US
equities. It never places live orders and its output is not investment advice.

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
`logs/tranche_watch.log`. Known issue: nightly's Yahoo fetch intermittently dies with a
fatal `PyEval_SaveThread` GIL error; the watcher tolerates this and flags stale data.

### Tranche watch program (ACN / CTSH / EPAM)

`scripts/tranche_watch.py` monitors staged buy-entry triggers from the 2026-07 buy-timing
research: T1 dip, T2 repair (alert carries a 61%-whipsaw caution), T3 200-DMA trend, REL
peer-basket relative-strength recross, SECTOR breadth, CAL seasonal windows, plus
earnings-blackout suppression (Yahoo quoteSummary needs the cookie+crumb flow — already
implemented there) and REVIEW/exit checks for positions recorded in
`artifacts/tranche_positions.json`. Outputs: Windows toast (`scripts/notify_toast.ps1`,
requires powershell.exe 5.1, not pwsh), `artifacts/tranche_status.html` dashboard,
`artifacts/tranche_watch.json`, dedupe state in `artifacts/tranche_watch_state.json`
(delete to re-arm alerts). `scripts/tranche_policy_backtest.py` is the companion cohort
study; its headline: staged entry is insurance, not alpha, and lump-sum recovery odds
depend heavily on whether the market itself was crashed at episode start.

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
