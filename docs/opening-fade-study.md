The local archive is not sufficient to establish an opening-fade frequency or a
tradable edge. The first frozen run used nine eligible SPY/QQQ symbol-sessions
from five July 2026 sessions at coarse 15-minute resolution. Its statistics are
descriptive only, all 40 strategy trials failed the research gates, and no
prospective monitor, canonical weight, ticket, or order was created.

# Opening spike and fade research campaign

## Design and safety boundary

This is a standalone campaign in `edgestack.research.opening_fade`, invoked by
`scripts/opening_fade_study.py`. It is intentionally disconnected from the V2
recommendation bundle, Android actionable views, sleeve promotion, portfolio
weights, the tranche watcher, and every broker/order interface.

The campaign freezes its occurrence definition, bounded A-H candidate family,
costs, validation plan, symbols, provider, and random seed in
`configs/opening_fade.yaml`. Unsupported and zero-trade variants remain in the
40-trial family. Outputs are content-addressed and atomically published under
`artifacts/research/opening_fade/runs/<content-hash>/`; `current.json` is only a
pointer. A successful-looking historical result can at most become a separate
paper-observation candidate after every gate passes.

Signals use completed bars and enter at the next bar open. Stop wins when stop
and target share a bar; a gap through a stop fills at the worse open. Daily
features are lagged. Cross-series signals use inner timestamp joins without
forward-filling. Simultaneous instruments collapse to one equal-weight session
return for inference.

## Data-availability audit (2026-07-20)

| Evidence | Local state | Classification |
|---|---|---|
| SPY 15m | 131 bars, 5 sessions, 2026-07-13 through 2026-07-17 | `INSUFFICIENT` |
| QQQ 15m | 131 bars, 5 sessions, same window | `INSUFFICIENT` |
| IWM/DIA 15m | absent | `INSUFFICIENT` |
| Sector/style ETFs 15m | absent except context ETFs TLT/GLD | `INSUFFICIENT` |
| Optional liquid stocks 15m | absent | `INSUFFICIENT` |
| SPY/QQQ 60m | 5 sessions | unusable for opening sequence |
| 1m and 5m | absent at audit time | `INSUFFICIENT` |
| Premarket | absent from existing files | unavailable |
| After-hours | one 16:00 ET boundary timestamp per SPY/QQQ file; not a usable series | `INSUFFICIENT` |
| Bid/ask and auction imbalance | absent; OHLCV trades only | unavailable |
| Futures | no licensed ES/NQ/RTY intraday series | unavailable |
| Economic calendar | no point-in-time `known_at` calendar | unavailable |
| Historical breadth | no timestamp-aligned point-in-time component panel | unavailable |

Canonical intraday timestamps are timezone-aware UTC and are converted to
`America/New_York` for session logic. The implementation never hard-codes a
German open time and includes a US/EU DST-mismatch test. Volume is present, but
venue/auction decomposition is not. VWAP is an OHLCV typical-price
approximation, not a consolidated trade-tape or executable quote VWAP.

The bundled Yahoo provider now accepts 1m, 5m, 15m, and 60m requests and can
request extended-hours trades. Its conservative per-request limits are 7 days
for 1m, 59 days for 5m/15m, and 729 days for 60m. Yahoo is unofficial,
throttle-prone, survivorship-biased, and may restate history. A licensed source
can implement the existing `IntradayDataProvider` interface or supply canonical
UTC OHLCV files; credentials must remain outside Git. The study does not scrape
eToro and does not equate broker CFDs with ETFs, futures, or cash indices.

Data labels are assigned as follows:

- `INSUFFICIENT`: under 60 sessions or resolution coarser than 15m.
- `EXPLORATORY`: at least 60 but fewer than 252 sessions, or only 15m bars.
- `RESEARCH_USABLE`: at least 252 complete 1m/5m sessions for bounded
  walk-forward research.
- `PROMOTION_ELIGIBLE`: never awarded by this campaign; independent V2 and
  prospective requirements would still have to pass.

## Frozen definitions and strategy family

The primary occurrence definition is a positive true-premarket gap, or clearly
labeled official-open gap when premarket is absent, of at least 0.20%; a
first-15-minute spike of at least 0.15 lagged daily ATR above the open; and at
least 50% retracement by 10:30 ET. It also reports 25%, 50%, 75%, full/open,
half-gap, and full-gap retracements with Wilson confidence intervals, counts,
and effective sample sizes. At 15m, high/low ordering and morning-high timing
are labeled ambiguous rather than guessed.

The 40 preregistered trials cover:

- A: VWAP, opening-low, and lower-high confirmed fades.
- B: 5m/15m/30m opening-range false breakouts and a buffered variant.
- C: true overnight-high touch, basis-point, and ATR-normalized overshoot
  rejections; unsupported without
  premarket data.
- D: naive gap shorts versus VWAP/opening-low confirmed gap fills.
- E: synchronized SPY/QQQ/SMH failures, SPY/QQQ divergence, and a point-in-time
  breadth variant.
- F: separate event population, wait-until-after-event, and exclude-event
  approaches; unsupported without `event_time_utc` plus `known_at_utc`.
- G: fixed calm/stressed volatility, trend, gap-size, opening-range-width, and
  recent-drawdown filters; no full-sample cutoff optimization.
- H: negative-gap long mirror, continuation, 09:45/10:00 naive shorts,
  deterministic random-time, unconditional-short, fixed opening-range, and
  no-trade controls.

Costs use the repository's optimistic/base/conservative/stress multipliers over
commission, modeled half-spread, slippage, participation impact, sell fees, and
short borrow. Individual-stock sessions must also clear frozen $5 price and
$1 million opening-dollar-volume floors. Requested participation above the 5%
per-bar cap is explicitly recorded as a partial fill. Reported performance is
unlevered. A separate paper-only risk
illustration caps hypothetical risk at 0.25% of example equity, forbids
averaging down, treats cross-index signals as one correlated position, includes
a daily loss limit and consecutive-loss pause, and creates no order.

Validation requires complete out-of-sample folds, purging/one-session embargo,
session ESS, circular block-bootstrap intervals, complete-family BH correction,
deflated Sharpe, SPA/StepM where at least 60 aligned sessions exist, paired
controls, cost monotonicity, parameter-neighborhood consistency, extreme-session
exclusions, multi-instrument and multi-year consistency, delayed-entry survival,
and negative/injected/train-only/cost-submerged acceptance tests. A candidate
also fails if it depends on one instrument, cannot beat matched random-time and
continuation controls, or breaches the frozen drawdown bound.

## First local run: descriptive result, not evidence of an edge

Command:

```powershell
.venv/Scripts/python scripts/opening_fade_study.py `
  --config configs/opening_fade.yaml `
  --symbols SPY,QQQ,IWM,DIA --interval 15m `
  --run-strategies --cost-sensitivity --render-report --json
```

Only nine symbol-sessions were eligible: five SPY and four QQQ (QQQ's first
session lacked a preceding close). The coarse observations were:

- 50% spike retracement by a post-opening bar low: 9/9, 100%, Wilson 95% CI
  70.1%-100%.
- Full retracement to the open by a post-opening bar low: 9/9, 100%, CI
  70.1%-100%.
- Full previous-close gap fill: 2/9, 22.2%, CI 6.3%-54.7%.
- Subsequent new high/continuation trap: 3/9, 33.3%, CI 12.1%-64.6%.
- Preregistered full opening fade: 0/9, CI 0%-29.9%.
- QQQ had the larger observed opening movement, but its sample was four and
  lacked daily ATR context; this is not a tradability ranking.
- A few two-to-four-trade naive variants had positive descriptive expectancy,
  while most evaluated variants were negative. No candidate had a valid
  walk-forward fold, all q-values were 1.0, and none cleared the research gates;
  selecting the tiny-sample winner would itself be data snooping.

The apparent 9/9 all-session and 4/4 positive-gap retracement rates are
especially uninformative: their Wilson lower bounds are only about 70% and 51%,
respectively, observations are correlated, the dates were previously accessed,
and 15-minute paths remain coarse. Neither rate is a probability of future
profit; the stricter ATR-and-gap preregistered full-fade event occurred 0/9.

## Commands and evidence milestones

```powershell
# Machine-readable audit only
.venv/Scripts/python scripts/opening_fade_study.py --audit-data --json

# Deterministic synthetic smoke run
.venv/Scripts/python scripts/opening_fade_study.py --smoke --json

# Forward collection; extended hours are requested and writes are idempotent
.venv/Scripts/python scripts/opening_fade_study.py --collect-forward `
  --symbols SPY,QQQ,IWM,DIA,XLK,SMH,XLF,XLE,XLY,XLP `
  --interval 5m --json
```

- 3 months: validate coverage, timestamps, missing bars, and descriptive
  examples only.
- 6 months: exploratory primary-ETF frequency intervals; still weak for
  regimes and multiplicity.
- 12 months: a first frozen bounded walk-forward run may become possible if
  1m/5m coverage is continuous.
- 24 months: multi-regime research may become `RESEARCH_USABLE`; promotion and
  trading authorization still do not follow.

Exact next step: continue extended-hours 1m/5m collection for the frozen liquid
universe and rerun the unchanged campaign after at least 252 complete sessions.
Because no candidate survived now, the prospective shadow monitor is
intentionally not implemented.
