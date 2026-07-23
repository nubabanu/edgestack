# Strategy Zoo — historical, non-promotable research

> V2 claim withdrawal: every result, including 2024–2026, has been previously accessed. Validation/holdout labels below describe historical experiment partitions only; they do not satisfy V2 promotion and none receives portfolio weight.

> Research output only. Not investment advice. 462 (rule, instrument) trials;
> at |t|>2 ~12 false positives are expected by chance. The survivor bar is
> stricter (Sharpe >= buy-and-hold in all three splits AND pooled NW alpha
> t >= 2), and the headline rule survives on three instruments independently.
> Caveat: survivor selection saw all three splits, so the 2024-26 holdout is
> previously accessed for these specific rules; the rules themselves are
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

---

# Allocation Zoo — monthly/yearly families (2008-2026, multi-asset ETFs)

Universe: SPY QQQ IWM EFA EEM TLT IEF SHY LQD HYG GLD DBC VNQ + 9 sectors.
Monthly signals, 5 bps costs. Survivor bar: Sharpe >= max(SPY, 60/40) in all
three splits (dev 08-15, val 16-23, holdout 24+) AND pooled NW alpha t >= 2.

## Result: ZERO survivors of 23 trials. Diversification wins, timing loses.

| Rank | Strategy | alpha/yr (t) | beta | Where it failed |
|---|---|---|---|---|
| 1 | Permanent portfolio (25% SPY/TLT/SHY/GLD) | +3.6% (2.20) | 0.25 | val Sharpe 0.77 vs 0.85 |
| 2 | Asset-class momentum top-3 + trend gate | +5.1% (2.15) | 0.35 | val 0.54 — momentum's lost decade |
| 3 | 60/40 with 5% rebal bands | +1.5% (2.00) | 0.59 | ties benchmark (it nearly IS it) |
| 4 | Faber GTAA-5 (10m SMA, 5 assets) | +2.5% (1.93) | 0.25 | val |
| — | Risk parity (SPY/TLT/GLD/DBC inv-vol) | +3.2% (1.79) | 0.36 | dev; best HOLDOUT Sharpe of all (2.09) |
| — | Dual momentum (GEM) | +0.5% (0.18) | 0.70 | everywhere post-publication |
| — | Halloween / Sell-in-May | +0.4% (0.18) | 0.49 | val 0.37 — effectively dead at monthly scale |
| — | SPY 10-month SMA (Faber timing) | +3.0% (1.30) | 0.47 | val — whipsaws 2016-2023 |
| — | Sector rotation (12-1 or 6m, top 3) | <= +0.7% (<=0.4) | ~0.85 | no edge after costs |
| — | Vol-target SPY 10% | negative alpha | 0.67 | drag in calm bull years |

## Family verdicts (monthly/yearly)

- Passive diversification (60/40, permanent, risk parity): the only positive-
  alpha family vs SPY — that's the diversification premium, earned at low
  beta, not timing skill. Permanent portfolio is the best all-era package.
- Rebalancing: threshold bands ~= monthly ~= annual (t 2.00 vs 1.84);
  never-rebalancing is worst risk-adjusted. Frequency barely matters; doing
  it at all matters.
- Trend/TAA (10m SMA, GTAA, TSMOM): real drawdown protection, but underper-
  formed 2016-2023; no timing family passed all eras.
- Dual momentum (GEM): t=0.18 since 2008 — post-publication decay in full.
- Seasonality at monthly granularity (Halloween, January barometer): dead or
  marginal; the intramonth effects (ToM, single days) from the daily scans
  remain the live ones.
- Untestable here (no data, marked honestly): fundamental value/quality/
  growth/income screens (no PIT fundamentals history), carry (FX/rates),
  options income/hedging, credit/duration beyond ETF proxies, crypto,
  tax strategies, event-driven (PEAD, M&A, spin-offs).

---

# Execution sensitivity + Unified book (Tier 3 refinements)

## Execution reality check (execution_sensitivity.py)

The zoo convention fills AT the signal close — impossible live (MOC cutoff
precedes the close). Under honest models at 2 bps:

| Vehicle | same-close (optimistic) | next-open (realistic) | next-close (worst) |
|---|---|---|---|
| ensemble4 SPY | t=4.04 | t=2.39 (marginal) | t=3.02 |
| ensemble4 QQQ | t=4.85 | **t=4.31 — survives intact** | t=2.65 |

Verdict: implement the satellite on QQQ with next-open (MOO) fills. At 10 bps
costs SPY dies (t=1.01); QQQ still stands (t=3.44).

## Unified book (unified_book.py) — the final portfolio

One jointly-constructed, vol-targeted book: permanent-portfolio core
(SPY/TLT/SHY/GLD, 5% bands, 5 bps) + ensemble4-on-QQQ satellite at
realistic next-open fills. Grid of 9 (split x vol-target) combos selected
on dev(2005-15)+val(2016-23) ONLY; holdout reported after selection.

SELECTED: **60% core / 40% satellite, 8% vol target** —
historical 2024–2026 result (previously accessed): Sharpe 1.70 vs SPY 1.31, CAGR +15.3%, maxDD **-5.9%**
vs SPY -18.8%; pooled alpha +5.9%/yr (t=4.88) at beta 0.29.

Robust: all 9 combos land holdout Sharpe 1.64-1.74 — the choice is not
fragile. Honest caveat: raw holdout CAGR trails SPY (15.3% vs ~21%); the
book wins on risk-adjusted terms and drawdown (one third of SPY's), not on
raw return in a bull market. It is the codified version of the final
recommendation: diversified core, validated satellite, vol governor.

## USO entry-timing transfer test (uso_entry_study.py, 2026-07-21)

Question: do the tranche T1/T2/T3 entry triggers transfer to oil (USO), i.e.
should tranche_watch cover it? Two independent tests, both negative:

- **Zoo edge-check** (35 rules): zero survivors. Best was mr_rsi2_dip_uptrend
  (dev/val beat B&H, holdout collapsed -0.52 vs 0.87; alpha t=1.69 < 2).
- **Entry-timing transfer** (go-backtest gate: conditional forward return
  must beat unconditional entry in all splits AND pooled non-overlapping
  event t >= 2): **0 of 16 trigger x horizon combos pass**; best pooled
  t = 0.57. Dev split is thin (curated USO starts 2015) — but no candidate
  even trends toward passing.

Verdict: **do NOT add USO to tranche_watch.** The equity dip/repair/trend
triggers carry no measurable entry edge on oil. Honest caveats: single
previously-accessed series, holdout previously accessed, USO roll-cost bleed
makes buy-and-hold a weak baseline (which flatters timing rules — and they
still failed).

## Breadth timing suite (breadth_timing.py, 2026-07-21)

First test of S&P breadth internals as a timing family — computed over
POINT-IN-TIME membership (universe_pit), not the survivorship-biased catalog
list. 11 pre-declared rules (textbook thresholds, no parameter search) x
SPY/QQQ = 22 trials (~0.6 false positives expected at t>=2): washouts
(%>20DMA < 15%), Zweig thrust (0.40 -> 0.615), participation gates
(%>200DMA, with and without hysteresis), McClellan oscillator, net new
highs, price+breadth confirm, dip-in-breadth-uptrend.

Result: **ZERO survivors, zero near-survivors.** Best was
dip_in_breadth_uptrend on SPY (pooled alpha t=1.98 — under the bar, and
Sharpe < B&H in dev AND val; its t is carried by the previously-accessed
2024+ window). Price+breadth confirm (t=1.63) adds nothing over the plain
200-DMA gate. McClellan-positive is actively bad (holdout Sharpe -0.08 vs
1.26).

Verdict: breadth internals, measured honestly, do not improve on the
already-validated price-based gates for SPY/QQQ exposure. Consistent with
the zoo's standing conclusion: diversification wins, timing loses. Caveats:
panel starts 2011 (dev split is 5y); PIT membership with free-feed price
coverage of 379-475 members (reported per split in
artifacts/breadth_timing.json).

## Modern-era re-score of the rejected graveyard (modern_era_rescore.py, 2026-07-21)

Hypothesis: "markets changed after 2020; rules rejected on full history may
be good in the modern era." Re-scored ALL 407 (rule, vehicle) trials — full
zoo library + breadth suite — on 2020-2022 and 2023-2026 windows only
(candidate bar: >= B&H Sharpe in both + pooled 2020+ alpha t >= 2).

Result: **8 candidates vs ~10 expected by chance** — and their composition
is the real finding. The top candidates are the ALREADY-VALIDATED families
re-emerging in the modern window: trend+dip composite (QQQ t=3.56), 3-down-
days mean reversion (XLK t=3.33, QQQ t=3.11), 20d-high breakout (QQQ/SPY
~t=2.1-2.2). The only true graveyard resurrection is PSAR on XLY (one
sector, t=2.24) — exactly the profile of the ~10 chance hits.

Verdict: the post-2020 market rewards the SAME edge families that worked
before 2020. No evidence of edge rotation; strong evidence of edge
stability. The "regimes changed" argument, tested against 407 rejected
trials, resurrects nothing beyond noise. All windows previously accessed;
hypothesis-generation only.

## Ensemble4 re-weighting test (ensemble_weights.py, 2026-07-21)

Prompted by the modern-era re-score ("the validated families lead post-2020
— weight them accordingly"). Five pre-declared schemes (equal incumbent,
alpha-t-weighted, Sharpe-weighted, inverse-vol, drop-worst), weight inputs
from dev+val only, next-open fills at 2 bps, selection rule: beat equal's
Sharpe in BOTH dev and val on QQQ; holdout reported after.

Result: **no challenger beats equal weight in both splits** (alpha-t and
inverse-vol win dev by +0.02 Sharpe, lose/tie val; drop-worst loses val
badly). The four families' dev+val evidence is nearly uniform (alpha t
2.18-3.08) — when evidence is this balanced, 1/N IS the evidence-based
weighting (DeMiguel et al. 2009 in miniature). Differences across schemes
are ±0.02 Sharpe: churn without signal.

Verdict: keep ensemble4 equal-weighted. Weights that "lean into" any family
are fitting noise in the seen window.

## Legacy confluence "WIND" audit (confluence_timing.py, 2026-07-21)

Five observable components retain the original arithmetic under neutral names:
`turn_of_month`, `post_down_month`, `weekday_reversal`, `trend_vol_regime`,
and `short_term_dip`. They are not treated as observed institutional flows or
independent mechanisms. The reproduction is frozen to its original
2026-07-15 cutoff with input hashes; all 2024+ data and source-agent results
were previously accessed and are promotion-ineligible.

Corrected findings:

- **Strict monotonicity fails** in every SPY and QQQ split. The old report
  clipped scores below -1 and above 4 into tail buckets, masking non-monotonic
  extremes. The audit reports every exact score.
- Selective entry still fails: score>=2 and score>=3 do not clear the old
  all-split survivor bar.
- The old daily switching rule that holds QQQ except on negative-score days
  retains its historical result (pooled alpha t=3.13 at 2 bps); SPY does not
  pass (t=1.43). A simple trend/volatility filter fails the all-split bar, but
  the expanded ablations and cost checks are post-hoc diagnostics, not new
  evidence.
- Most importantly, an all-in/all-out daily switching backtest does **not**
  validate delaying an already-planned purchase. WIND therefore has no
  actionable veto, green light, sizing, or leverage interpretation.

Operational verdict: the watcher publishes a V2 `DESCRIPTIVE_ONLY` / `NO_ACTION`
payload and never alerts or creates tickets. Both leveraged paper experiments
are retired. A new unlevered forward shadow compares a hypothetical purchase at
a negative-score session's open with a mandatory fill exactly one XNYS session
later. Review requires at least 252 prospective sessions and 30 completed
events, and cannot promote automatically. See `docs/market-cycle-claim-audit.md`
for the source-claim evidence ledger.

## Oil shock dip-entry study (oil_shock_study.py, 2026-07-23)

Question, asked after the Bab al-Mandab closure surge was missed: once CL=F
has already jumped (>= +4% day or >= +8% over 5 sessions), does buying the
subsequent pullback beat unconditional entry? The study question was selected
after observing a 2026 episode — selection bias is baked in, and the 2024+
holdout is previously-accessed twice over.

Setup: CL=F daily 2000-2026 (Yahoo silently dropped the negative-price rows
of April 2020; the 2020-04-15..2020-05-01 window is excised and logged).
Episode definitions E1 (raw threshold, 196 episodes) and E2 (E1 minus
crash-rebound days with trailing 20-session return <= -10%, 167 episodes).
Entry rules R1 immediate / R2 first down close / R3 pullback >= 1.5% from the
post-shock closing high / R4 pullback >= 2.5%; horizons 5/10/20 sessions;
fill at close t+1; 24 trials with a batch id, Bonferroni-deflated bar (~2.9)
reported alongside the repo t >= 2.

**Verdict: FAIL — no trial clears the event gate plus survivor bar.** The
directional picture is consistent but weak: E2/R3 (pullback-buy during a
non-crash surge) beats unconditional in all three splits at every horizon,
best cell E2/R3@20d at +2.2% pooled excess, t = 1.84. +4% oil days are
common (196 in 26 years), and their average follow-through is small relative
to oil's daily noise. Operational consequence: `scripts/oil_surge_watch.py`
SHOCK alerts are live (descriptive facts), DIP alerts stay DISPLAY-ONLY, and
`DIP_TICKETS_ENABLED` remains False. Re-run the study before ever proposing
to flip it; INSUFFICIENT/FAIL forever is a valid end state.
