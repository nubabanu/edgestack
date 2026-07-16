# Strategy Zoo — every daily-testable family from the retail catalogue

> Research output only. Not investment advice. 462 (rule, instrument) trials;
> at |t|>2 ~12 false positives are expected by chance. The survivor bar is
> stricter (Sharpe >= buy-and-hold in all three splits AND pooled NW alpha
> t >= 2), and the headline rule survives on three instruments independently.
> Caveat: survivor selection saw all three splits, so the 2024-26 holdout is
> no longer "untouched" for these specific rules; the rules themselves are
> textbook definitions with no parameters tuned on our data.

Data: SPY, QQQ + 11 SPDR sector ETFs, daily bars 1999->2026. Signals at close
t earn session t+1; 2 bps per unit exposure change. Splits: dev 1999-2015,
val 2016-2023, holdout 2024+.

## Survivors (6 of 462)

| Rule | Instrument | Pooled alpha | t | What it is |
|---|---|---|---|---|
| trend OR RSI2-dip | SPY | +5.2%/yr | 3.02 | hold above 200-DMA; below it, hold only RSI(2)<10 panic days |
| trend OR RSI2-dip | QQQ | +7.7%/yr | 3.15 | same |
| trend OR RSI2-dip | XLK | +8.3%/yr | 3.51 | same — 3 instruments = cross-confirmation |
| trend AND dip-buy | QQQ | +5.6%/yr | 3.27 | above 200-DMA and (RSI2<25 or IBS<0.3) — 23% exposure |
| trend AND dip-buy | XLK | +5.1%/yr | 3.19 | same |
| 20d-high breakout, hold 10 | QQQ | +5.5%/yr | 2.49 | classic breakout-momentum |

## Combinations (requested): all PASS all splits

| Combo (QQQ) | dev Sharpe | val | holdout | alpha t |
|---|---|---|---|---|
| buy-and-hold reference | 0.32 | 0.86 | 1.19 | — |
| equal-weight avg of 4 families | 0.72 | 1.27 | 1.49 | **4.85** |
| max(trend-dip, 20d-breakout) | 0.57 | 1.10 | 1.43 | 3.20 |
| trend-dip x vol-target gate | 0.56 | 1.12 | 1.22 | 3.28 |

The equal-weight combination (trend/dip composite + 20d-high breakout +
10%-vol targeting + 3-down-days reversion) is the best risk-adjusted package:
it beats buy-and-hold Sharpe in every split on both SPY and QQQ and carries
the highest alpha t-stat produced by this entire research program.

## Family verdicts

| Family | Verdict | Evidence |
|---|---|---|
| Short-term index mean reversion (RSI2, 3-down-days, IBS, RSI14) | **STRONGEST single family** | QQQ 3-down-days t=4.15; consistent on SPY/XLK; works via next-day bounce |
| Trend + dip composites | **Survivor** | the only rules passing all splits on 3 instruments |
| Volatility management (vol targeting, vol regime) | **Near-survivor, real** | QQQ vol-target t=3.28; same mechanism as the validated 200-DMA/vol gates |
| Breakout (20d high hold-10) | Survivor on QQQ only | t=2.49; 52w/NR7/inside-day/squeeze variants weaker |
| Time-series momentum (3/6/12m) | Positive, not significant | best QQQ 6m t=2.45, fails split consistency |
| MA crossovers (10/20, 20/50, 50/200, EMA, MACD) | No edge vs B&H | none survive; they are worse versions of price>200-DMA |
| Supertrend / PSAR / ADX / Donchian-turtle | No | below B&H Sharpe in most splits |
| Candlestick reversals (engulfing, hammer, doji, soldiers...) | **Dead** | pooled next-day excess within noise; hammer/-9bps, soldiers -7bps |
| Position-management stops (chandelier, -5% + reenter) | Hurts returns | drag vs B&H, consistent with earlier MAE findings |
| Calendar/seasonality | Validated earlier | ToM, Sep/Oct-Nov, overnight (see calendar_overlay + seasonality_scan) |
| Cross-sectional momentum/low-vol | Economically real, t<2 | see cross_sectional.json |
| Single-stock edge mining | Rejected at scale | rules campaign verdict (0 stable rules, canary-validated) |

## Untestable with this stack (no data — not faked)

Order-flow/tape/DOM/footprint/market-profile (no L2 or tick data); options
strategies incl. 0DTE (no options chain history); FX/futures/crypto sessions
(no data); earnings/analyst/M&A event-driven (no event feed); market internals
(TICK/TRIN/breadth); pairs/stat-arb at institutional latency; market making;
anything in the prohibited/manipulative list (would not test regardless).
