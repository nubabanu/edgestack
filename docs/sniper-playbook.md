# Sniper V2 — staged loss-aversion-first shadow strategy

> Research and paper-trading shadow output only. No live orders and no investment advice. The
> earlier broad "all-stars" sniper hypothesis remains rejected. The narrower rules below use
> previously accessed history descriptively and receive zero canonical portfolio weight unless a
> frozen sleeve passes every Recommendation Engine V2 promotion gate.

## Ranked policy

| Rank | Rule | Current use |
|---:|---|---|
| 1 | C1/C2 RSI(2) or three-down dip | Stage 1 shadow engine |
| 2 | A1 Santa Claus window | Stage 1 scheduled shadow candidate |
| 3 | A2 defined-risk pre-FOMC | Blocked: options/FOMC/IV/fill data absent |
| 4 | E1 TLT month-end | Blocked until Stage 1 promotion and separate validation |
| 5 | C3 VIX contango re-cross | Veto/filter only; unavailable without point-in-time futures |
| 6 | C4 put/call and breadth washout | Confirmation only; unavailable without compatible inputs |
| 7 | A6 OpEx-week micro-tilt | Blocked; no incremental promoted artifact |
| 8 | E2 Gold January | Blocked low-conviction watchlist observation |

The hard-disabled set is A3 minor-holiday effects, A4 small-cap January, A5 weekend/Monday, B
naive overnight-only, and D short-volatility income. They cannot create a candidate, overlay, or
portfolio weight.

## C1/C2 primary engine

The engine evaluates SPY or an approved diversified low-volatility vehicle (`USMV`, `SPLV`, `XLV`,
or `XLP`) after the session close. It creates one combined candidate—not two positions—when either
RSI(2) is below 5 or adjusted closes have fallen for three consecutive sessions.

Eligibility requires both the vehicle and SPY to be above their 200-session moving averages and
both 20-session annualized volatilities to be at most 20%. September is a mandatory stand-aside.
A supplied VIX backwardation observation vetoes the candidate. The planned entry is the next
regular-session open; the descriptive exit is the first close above the then-known 5-DMA or the
fourth-session close.

Sizing is:

```text
shadow notional = maximum tolerable modeled loss / abs(prior-signal adverse-move p5)
```

The notional is capped at account equity, so the shadow preview cannot introduce leverage. At
least 30 resolved, non-overlapping in-regime historical signals are required before using the
descriptive adverse-tail estimate; otherwise the engine uses a conservative −4% fallback. The
5th percentile is not a maximum loss—gaps, execution, and regime change can lose more.

## A1 Santa window

The scheduled entry is the close of the fifth-to-last December exchange session and the exit is
the close of the second January session. It has its own descriptive outcome series and sizing. It
is not automatically sent to the canonical paper executor because close-auction execution differs
from the default next-open policy.

## Stages and overlays

Stage 2 cannot unlock merely because Stage 1 has a good recent result. It requires a compatible
promoted Stage 1 artifact, then independent data and validation for the Stage 2 family. A2 also
requires point-in-time FOMC dates, option chains, IV rank, spreads, and actual-fill assumptions;
shares are not substituted for the defined-risk option expression.

C3 and C4 have typed contracts with `can_initiate=false`. C3 can only pass or veto C1/C2. C4 can
only confirm an existing eligible C1/C2 candidate; it cannot override a trend, volatility,
September, freshness, or risk veto. A6 currently applies no tilt.

## API and Android

- `GET /sniper/latest` evaluates the canonical session with the default profile equity and a 0.25%
  modeled loss budget.
- `POST /sniper/preview` accepts account equity, maximum tolerable loss, and an approved vehicle.
  It is stateless and does not mutate the canonical recommendation.
- The Android **Sniper** tab consumes these server contracts, displays trigger components, entry
  and exit rules, modeled loss sizing, vetoes, evidence, blocked stages, overlays, and exclusions.
  Offline mode shows the last server result without calculating signals on-device.

Sniper output is version-bound to the current canonical data and artifact versions. If market data
changes after publication, the API fails closed until the canonical bundle is republished.
