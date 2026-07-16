# The Sniper Playbook — loss-aversion-first trading

> Research output only. Not investment advice. Every number below is measured
> in this repo (see seasonality_scan, strategy_zoo, xmarket_replication,
> french_deep_history, execution_sensitivity artifacts).

## Design principle

Trade the probability of NOT losing, not expected profit — and trade rarely.
Every window below is a 1-4 session hold (a defined exit means a loss cannot
grow while you watch), always unlevered, always on index/defensive ETFs.
Single stocks run ~52-54% daily hit rates at best — psychologically wrong for
a loss-averse trader — with one measured exception noted below.

## The calendar (regenerate dates: `python scripts/sniper_calendar.py`)

| Window | Vehicle | Hold | Hit rate | Worst ever | p5 |
|---|---|---|---|---|---|
| Feb-1 (buy last Jan close) | XLV | 1 session | **89%** (28y) | −2.0% | −0.3% |
| Feb-1 companion | V (Visa) | 1 session | 80% (15y, MA-confirmed) | — | — |
| Wed before Thanksgiving (buy Tue close) | SPY | 1 session | 70% | −2.4% | −2.0% |
| Turn-of-month (buy last session close) | QQQ/SPY | 4 sessions | 62-64% | −7% | −2.5/−3.9% |
| 15th Dec trading day (buy td14 close) | SPY | 1 session | 73% | small | — |
| **Red close in calm uptrend (any day)** | SPY | overnight only | **60%, t=5.41** | — | **−0.8%** |

Confirmed cross-market: ToM positive in 12/12 foreign markets and 6/6 US eras
back to 1927; September weakest in 12/12 markets and 6/6 eras.

## Stand-aside rules (avoiding loss is also a trade)

- ALL of September: worst month everywhere; last August session = de-risk day.
- 7th November trading day: worst single day of the year (SPY & QQQ agree) —
  flat at the prior close.
- Never initiate before a >=4-day holiday closure (the only overnight span
  with negative average returns).
- VETO everything when SPY < 200-DMA or 20d vol > 20%: hit rates collapse
  and tails fatten (gap-down buys in high-vol sideways: 20% hit, −8.6% mean).

## Kind-of-vehicle rules (halves the loss SIZE)

Lowest-vol-quintile stocks: bad day p5 −1.8% vs −4.2% for the wildest
quintile — but even the calmest single stock is ~53% daily hit with
catastrophic idiosyncratic tails. Express "kind" through the vehicle:
XLV / XLP / USMV-style, or diversified low-vol baskets (p5 −1.5%, the
smallest tails measured). Single names additionally require: lowest-vol
quintile, beta < 1, clean fundamentals (no >10% short interest, no extreme
leverage, positive FCF), and NO earnings date inside the window.

## Gap rules at the open (for orders already planned)

- Moderate gap DOWN (−1..−3%): best entry cohort — proceed (t=16, 5d +59bps).
- Extreme gap DOWN (< −3%): cancel — dead-cat bounce, 5d −36bps.
- Moderate gap UP (+1..+3%): proceed, mild continuation.
- Extreme gap UP (> +3%): cancel — 47% hit buying that open. (Index gap-ups
  are the exception: SPY >+1% overnight continues, 60% same-day.)

## Sizing — where loss aversion is actually solved

position = (max € loss you can tolerate) / (worst-case % of the window).
Sized this way, the worst day in 30 years costs exactly your pre-agreed pain
limit. Never leverage: alignment grants permission and size, never aggression.

## Honest arithmetic

The best window loses 1 year in 9. The full calendar (~15 trades/yr at
65-89% hit) produces ~3-5 losing trades per year — small, capped, brief,
pre-agreed. No schedule has 100% hit; the design makes losses rare, small
and decided in advance, which is what makes them bearable.

Core for the untraded capital: permanent portfolio (25% each SPY/TLT/SHY/GLD,
5% bands) or the unified book (docs/strategy-zoo.md) — and check it rarely.

## App integration

The Android app fires all of these at 15:45 America/New_York: ToM window,
September de-risk, Feb-1 sniper, Nov worst-day, Thanksgiving Wednesday,
pre-Christmas, and the data-driven red-close-in-calm-uptrend trigger, plus
200-DMA breach and vol-gate transitions on the risk channel.
