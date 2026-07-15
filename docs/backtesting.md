# Backtesting

## What gets replayed

`edgestack backtest run` replays validated edges **out of sample only**:
signals are generated inside walk-forward test windows with quantile
thresholds fitted on each fold's training window. The engine never trades a
rule on the data that shaped it.

## Session loop

For each session, in order: (1) exits — time exit at the open when the
holding horizon has elapsed, then stop, then target (when a bar touches both
stop and target, the STOP is assumed to hit first — conservative); (2)
entries — market-on-open orders created from the previous session's signals;
(3) short-borrow accrual and mark-to-market; (4) tonight's signals become
tomorrow's entry orders. Signals can never execute at the close that
produced them.

## Fill realism (asserted in code, not just tested)

- every fill price lies within the session's `[low, high]`;
- a stop gapped through at the open fills at the open (worse than the stop);
- a limit never fills unless the bar touched it, and never at a worse price
  than the limit;
- one order consumes at most 10% of the session's volume (partial fills);
- friction (commission + scenario-scaled half-spread + slippage +
  participation-scaled impact + SEC fee on sells) is charged as explicit
  cash per leg, so fill prices remain verifiable against the bars;
- shorts accrue an annualized borrow fee daily, scenario-scaled.

## Cost scenarios

`OPTIMISTIC (0.5×) | BASE (1×) | CONSERVATIVE (1.5×, default) | STRESS (3×)`
on the friction components. A property test runs identical intents through
all four and asserts final equity is non-increasing — costs can never help.

## Metrics

Equity curve → cumulative return, CAGR, annualized volatility, Sharpe,
Sortino, Calmar, max drawdown, daily VaR/ES; trade ledger → hit rate, profit
factor, expectancy, average winner/loser, holding period, exit-reason
breakdown, long/short split. The mean daily return and daily Sharpe carry
**circular block-bootstrap 95% intervals** — when the Sharpe CI includes
zero, the report says so plainly.

## How to read the output

- The universe is a static, survivor-biased list: absolute levels are
  optimistic by construction (each report repeats this).
- One historical path, overlapping signals, regime dependence: a positive
  backtest is evidence, not proof, and never a promise.
- Compare CONSERVATIVE vs OPTIMISTIC runs: a strategy whose edge exists only
  under optimistic costs is already rejected at validation, but portfolio
  interactions can still differ.
- `report.json` (machine-readable, includes the full trade list) and
  `report.html` (self-contained) live under
  `data/reports/backtests/<run_id>/`; runs are registered in DuckDB.
