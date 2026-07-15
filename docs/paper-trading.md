# Paper trading

Paper trading simulates the daily operational loop with the SAME fill
simulator and cost model as the backtester — no real orders, ever.

## Hard safety rails

- `EDGESTACK_LIVE_TRADING_ENABLED=false` is the shipped default, and the
  configuration **refuses** `paper.live_trading_enabled: true` outright:
  this repository contains no live-trading implementation.
- Any future broker integration must implement the `BrokerAdapter` ABC
  (`src/edgestack/paper/broker.py`) and be explicitly and deliberately
  enabled. The only bundled adapter is `SimulatedBroker`.
- Short candidates are **never** paper-executed: without borrow data they
  are research output (`SHORT_RESEARCH_CANDIDATE`), and the session log
  records how many were skipped.

## Daily flow

```
edgestack signals generate --date <T>     # after T's close
edgestack paper run --date <T+1>          # next session
```

For session S the paper engine: (1) checks exits for open positions against
S's actual bar — time exit at the open once the holding horizon has elapsed,
else stop (gaps fill at the open), else target; (2) enters long candidates
from the report generated at the PREVIOUS session's close, at S's open,
sized by the score-weighted allocator under all portfolio limits; (3) marks
to market, persists state and prints a daily report.

## State and accountability

State lives in `artifacts/paper/state.json` (atomic writes): cash, open
positions, and every completed trade with **predicted probability and
expected net return recorded next to the realized outcome** — the raw
material for calibration monitoring. Sessions cannot be processed twice, and
every session writes an audit row.

## Interpreting results

Paper results inherit every backtest caveat (survivorship-biased universe,
free-data quirks) plus one more: fills are simulated against historical
bars, so they are optimistic about queue position and liquidity. Treat paper
P&L as a plumbing check and a forecast-quality tracker, not as evidence of
future returns.
