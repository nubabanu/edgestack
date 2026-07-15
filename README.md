# EdgeStack

EdgeStack is a research platform that automatically **discovers, validates,
scores, monitors and ranks statistical trading advantages ("edges") in daily
US equities** — and explicitly abstains when the evidence is insufficient.

> **Research / paper trading only.** EdgeStack produces research output, not
> investment advice. No signal it emits is guaranteed, certain or risk-free.
> The bundled free data sources carry survivorship bias and other limitations
> that are flagged in every report. **There is no live-trading implementation
> in this repository**, and the configuration refuses `live_trading_enabled: true`.

## What EdgeStack does

- searches historical data for conditional patterns (calendar effects,
  momentum/mean-reversion states, breakouts, candlestick and structure
  events, volatility regimes, cross-sectional ranks, and conjunctions of
  these) — nothing is hardcoded as true; everything is tested;
- validates every candidate **out of sample** with purged walk-forward splits,
  Benjamini-Hochberg FDR control over the full trial count, deflated Sharpe
  ratios, cost-scenario survival and fold-stability gates;
- prices trades realistically: next-open execution, spread/slippage/impact/
  borrow costs under four scenarios (conservative by default), stops that gap
  through at the open;
- emits ranked long and short candidates with calibrated probabilities, a
  0–100 conviction score, entry zones, stops/targets, holding horizons,
  contributing evidence and plain-English explanations — or an explicit
  NO-TRADE with every failed gate recorded;
- monitors validated edges for decay and drift and retires them through an
  event-sourced lifecycle (`VALIDATED → ACTIVE → DEGRADED → SUSPENDED → RETIRED`).

## What EdgeStack does NOT do

- predict the future or guarantee returns;
- trade real money (no broker integration exists; paper trading only);
- provide point-in-time universes, earnings timestamps, borrow data or
  fundamentals — the free providers cannot; the gaps are labeled, not hidden;
- treat an indicator, a candlestick or a calendar day as truth without
  out-of-sample statistical evidence.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows (source .venv/bin/activate on unix)
pip install -e .[dev,api]              # dashboard extra: .[dashboard]
```

Requires Python 3.12+. Copy `.env.example` to `.env` if you want environment
overrides; no API keys are needed for any bundled provider.

## Quick start

Offline (synthetic market, fully reproducible):

```bash
python scripts/demo.py --offline
```

Real data (free Yahoo endpoint, ~10 symbols, a few minutes):

```bash
python scripts/demo.py
```

Or step by step with the CLI:

```bash
edgestack data download --start 2012-01-01 --config configs/demo.yaml
edgestack data validate --config configs/demo.yaml
edgestack features build --config configs/demo.yaml
edgestack edges discover --config configs/demo.yaml
edgestack edges validate --config configs/demo.yaml
edgestack models train --config configs/demo.yaml
edgestack signals generate --config configs/demo.yaml
edgestack signals rank --top 10 --config configs/demo.yaml
edgestack backtest run --config configs/demo.yaml
edgestack report backtest --config configs/demo.yaml
edgestack monitor run --config configs/demo.yaml
edgestack paper run --config configs/demo.yaml
edgestack api serve --config configs/demo.yaml        # read-only FastAPI
edgestack dashboard serve --config configs/demo.yaml  # Streamlit ([dashboard] extra)
```

## Architecture (short version)

```
data providers (synthetic | local | yahoo | stooq)
  -> DataCatalog (Parquet + DuckDB, TestPeriodGuard, audit log)
  -> feature registry (~60 features, structural point-in-time safety)
  -> labels (open-to-open forward returns, triple barrier; explicit label_end)
  -> discovery (rule enumeration; EVERY evaluated rule counts as a trial)
  -> walk-forward validation (purge+embargo, FDR, deflated Sharpe, cost gates)
  -> Edge store (event-sourced lifecycle)
  -> models (base-rate -> logistic -> boosting; isotonic calibration on OOF)
  -> signal engine (family-capped evidence, conviction, abstention, plans)
  -> backtester (session loop, realistic fills) / paper trading / API / dashboard
```

Full details: [docs/architecture.md](docs/architecture.md),
[docs/data-model.md](docs/data-model.md),
[docs/statistical-validation.md](docs/statistical-validation.md),
[docs/confidence-score.md](docs/confidence-score.md),
[docs/backtesting.md](docs/backtesting.md),
[docs/paper-trading.md](docs/paper-trading.md), and the extension guides
([feature](docs/adding-a-feature.md), [provider](docs/adding-a-provider.md),
[model](docs/adding-a-model.md)).

## Execution assumptions

- Daily bars. A signal is computed from information available at session T's
  close and is executable **no earlier than session T+1's open** — same-close
  execution is structurally blocked in labels, the signal engine, the
  backtester and paper trading.
- Fills always lie within the session's [low, high]; stops gapped through
  fill at the open; limits never fill unless touched; one order may take at
  most 10% of a session's volume.
- Costs default to the CONSERVATIVE scenario (1.5× spread/slippage/impact,
  scenario-scaled short borrow). An edge that only survives OPTIMISTIC costs
  is rejected.

## The two numbers on every signal

1. **Calibrated probability** — `P(net return > 0)` from a model calibrated
   with isotonic regression on out-of-fold predictions (Brier/log-loss/ECE
   reported). When no model applies, the shrinkage posterior is used and the
   signal is explicitly flagged as *not independently calibrated*.
2. **Conviction score (0–100)** — *not* a probability. It compounds economic
   edge, statistical reliability, regime applicability and tradability, then
   subtracts uncertainty/tail/disagreement penalties and shrinks toward the
   neutral 50 for small samples. See
   [docs/confidence-score.md](docs/confidence-score.md) for the exact formula.

## Interpreting backtests

Backtests replay validated edges **only inside walk-forward test windows**
(thresholds fitted on each fold's training window), under conservative costs,
with block-bootstrap confidence intervals on the headline metrics. Even so:
the universe is survivor-biased, the sample is one historical path, and a
positive backtest is *evidence*, never proof. See
[docs/backtesting.md](docs/backtesting.md).

## Known limitations

- No point-in-time index membership → survivorship bias (warned everywhere).
- Yahoo/Stooq are unofficial feeds: throttling, gaps and restatements happen.
- No earnings timestamps, borrow data, fundamentals, options or sentiment —
  the provider interfaces exist, implementations await licensed data.
- Deterministic regimes only (fixed thresholds); learned regimes (HMM/GMM)
  are on the roadmap.
- Parameter-robustness scoring uses a subperiod-consistency proxy rather than
  full parameter perturbation.
- The DSR uses a documented 1/n approximation for cross-trial SR variance.

## Roadmap

Chart-pattern detectors (13.8), fuller candlestick library, earnings/macro/
short/options/sentiment providers, learned regime models, stacking ensembles,
intraday frequencies, task-queued research API, Docker packaging.

## Risk disclaimer

EdgeStack is educational research software. Markets involve substantial risk
of loss. Historical patterns — even rigorously validated ones — can and do
stop working. Nothing produced by this software is investment advice, and it
must not be used to place real orders.
