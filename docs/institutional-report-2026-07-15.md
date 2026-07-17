# EdgeStack Institutional Research Report — historical snapshot, 2026-07-15

> Superseded by Recommendation Engine V2. The 2024–2026 period was previously accessed, manual search history may be incomplete, and historical confirmation/promotion claims are withdrawn. Values below are provenance only and cannot assign portfolio weight.

> **RESEARCH OUTPUT — NOT INVESTMENT ADVICE.** Every number below comes from the
> EdgeStack point-in-time research stack (Yahoo daily OHLCV, PIT S&P 500 membership,
> purged walk-forward validation, 2024–2026 historical period). Verified data and
> estimates are labeled. Nothing here is a guarantee of returns.

---

## Section 1 — Executive conclusion and regime

**Regime (verified from data through 2026-07-14 close):** UP-trend / medium-low
volatility. SPY is ~8% above its 200-day moving average; 20-day realized volatility
~12% annualized. Historically (2012–2023 playbook, measured): in this regime entry
*timing within the week is nearly irrelevant* — trend dominates; buying dips and
buying strength produced statistically indistinguishable outcomes.

**Core verdict after the full 10-hour campaign — the sniper hypothesis is REJECTED
as tested.** Across every method available to this stack:

| Method | Result | Evidence |
|---|---|---|
| Rule discovery (EBM → RuleFit → stability selection → distillation), fully nested, ~7,700 trials/window | **0 stable rules** on the broad PIT universe; apparatus itself validated (canary rediscovers an injected effect at 89% frequency, finds 0/9,312 on pure noise) | rules_campaign.json |
| Ensemble of near-miss rules | **Significantly negative**: NW alpha −12.2%/yr, t = −3.93 | ensemble_returns_*.parquet |
| Genetic/GP rule mining | In-sample t = 23.7 → **0 out-of-sample** | roadmap falsification |
| Confluence ("all stars aligned") + leverage | **−62%** — maximum-signal days are adverse-selection (distress) days | sniper campaign |
| ML at horizons 1d / 5d / 10d / 21d (85 PIT features, 250k rows) | **Base rate wins every horizon** — no model improved OOF log-loss | horizon_sweep.json |
| Enumerated edge library, nested, after costs | Portfolio alpha **−9.2%/yr** vs exposure-matched SPY | nested_run.json |
| Cross-sectional factor deciles (momentum, low-vol, reversal) | Only architecture with *positive* holdout alpha — **but not statistically significant** (best t = 1.45) | cross_sectional.json |

**Direct answer to the mandate ("selective active + leverage should beat passive
SPY"):** On this data, after costs, **no tested active strategy beat SPY buy-and-hold
with statistical significance, and every leverage-concentration variant lost badly.**
The defensible refinements are risk-shaping, not return-beating: the SPY 200-DMA gate
cut max drawdown roughly in half in both eras (−19.8% vs −33.7% research; −10.0% vs
−18.8% holdout) at the cost of ~2–4% CAGR, and the momentum+low-vol decile sleeve
delivered a higher holdout Sharpe (1.44 vs 1.31) with beta 0.40.

**Hardware note (verified):** the AMD RX 6950 XT is present but unusable for this
stack — ROCm has no Windows support, and scikit-learn/interpret are CPU-only. All
compute ran CPU-side. Stating this per the no-fabrication rule.

---

## Section 2 — Ranked opportunities (frozen pre-2024 system, close-confirmed board)

Signals from the frozen live system (edges discovered pre-2024, never retrained).
These are **weak statistical tilts, not convictions**: expected net excess ≈ +0.6%
per 10 sessions, hit rate ≈ 54%, calibrated conviction ≈ 32/100. That conviction
level *is* the honest confidence statement. **Leverage: NO on every row** (see §7).

| # | Symbol | Direction | Entry | Invalidation (exit early) | Target / exit | Est. R:R | Confidence | Max size | Leverage OK? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | NRG | LONG | Thu 2026-07-16 market-on-open | −6% from entry or close below entry−1.5×ATR(14) | time exit close of 10th session (~2026-07-30) | ~1.3:1 | LOW (32/100) | 2% NAV | NO |
| 2 | FCX | LONG | Thu MOO | same rule | 10-session time exit | ~1.3:1 | LOW | 2% | NO |
| 3 | TEL | LONG | Thu MOO | same | 10-session | ~1.3:1 | LOW | 2% | NO |
| 4 | NCLH | LONG | Thu MOO | same | 10-session | ~1.2:1 | LOW | 1.5% (high beta) | NO |
| 5 | MCD | LONG | Thu MOO | same | 10-session | ~1.3:1 | LOW | 2% | NO |
| 6 | HAS | LONG | Thu MOO | same | 10-session | ~1.2:1 | LOW | 1.5% | NO |
| 7 | ISRG | LONG | Thu MOO | same | 10-session | ~1.3:1 | LOW | 2% | NO |
| 8 | STZ | LONG | Thu MOO | same | 10-session | ~1.3:1 | LOW | 2% | NO |
| 9 | PEP | LONG | Thu MOO | same | 10-session | ~1.3:1 | LOW | 2% | NO |
| 10 | ORCL | LONG | Thu MOO | same | 10-session; may extend to ~1 month (off-cycle earnings Sep) | ~1.4:1 | LOW | 2% | NO |

Timestamp: signals computed on 2026-07-15 data; board regenerated after the US close
(22:00 CET). Do-not-chase rule: cancel any entry that gaps more than +1.5% above the
prior close at the open. R:R estimates come from measured MAE/MFE distributions on
the same edge families (estimate, not guarantee).

**Cross-check with today's fundamentals snapshot (503 names, fetched live today):**
NRG carries flags — trailing P/E 151 (forward 11.9), debt/equity 4.8×, negative 12-mo
price change; the tilt is purely statistical. ORCL and MCD have no red flags.

---

## Section 3 — Best setup per horizon

- **Next day (1d): NO TRADE — abstain.** Verified: no model beats the base rate at
  the 1-day horizon (OOF log-loss improvement exactly 0.000). Any 1-day pick would be
  noise dressed as analysis.
- **Next week (5–10d):** the §2 basket (top 5: NRG, FCX, TEL, MCD, ISRG), equal
  weight, Thursday market-on-open, 10-session time exit, stops as above. This is the
  only horizon where the frozen validated system produces signals at all.
- **Next month (21d): ORCL** — §2 tilt plus no earnings date inside the window
  (off-cycle reporter, next report ~September), removing the largest single-event
  risk in a 1-month hold. Same entry/stop discipline, exit ~2026-08-14.
- **Next 12 months:** not a single-stock call — the evidence says structure, not
  stories: **momentum+low-vol top-decile sleeve** (holdout Sharpe 1.44, maxDD −9.3%,
  beta 0.40). If individual names are required, the intersection of the backtested
  momentum decile with today's value-quality-momentum composite top ranks:
  **NEM, MU, EXE/EQT, BAC, AVGO, GOOGL** (composite pillar scores in vqm_rank.json).
  *Label: the composite uses today's snapshot only — no historical fundamental
  vintages exist in this stack, so this ranking is informed judgment on current data,
  NOT a backtested result.*

---

## Section 4 — Watchlist (not yet actionable, with exact triggers)

| Symbol / instrument | Why watching | Trigger to act |
|---|---|---|
| SPY | Regime anchor | Close below 200-DMA → cut gross exposure to ≤50%; two consecutive weekly closes below → conservative portfolio only |
| MU | VQM #2 (quality 0.97, growth 0.99), beta 2.1 | Pullback ≥8% from high **while** SPY holds 200-DMA → eligible for 12-mo sleeve at 1.5% NAV |
| NVDA / AVGO | VQM top-10, momentum decile members | Same pullback-in-uptrend trigger |
| NEM | VQM #1 (value 0.85 + quality 0.88 — rare combo) | Any 5%+ dip; low beta (0.48) allows 2.5% NAV |
| BAC, C, CFG | Banks cluster in VQM top-10 (value + momentum) | Add on sector ETF confirmation (XLF > its 50-DMA) |
| Realized vol (SPY 20d) | Playbook switch | >20% annualized → stop buying gap-downs entirely (measured: 20% hit rate, −8.6% mean in high-vol sideways) |

---

## Section 5 — Avoid list (verified reasons)

| Symbol | Reason (from today's snapshot + composite) |
|---|---|
| ARE | Bottom composite: negative growth, −39% 12-mo, negative trailing earnings |
| COIN | Growth pillar 0.00, −59% 12-mo, beta 3.35, HIGH short interest |
| INTC | Negative FCF, forward P/E 65, quality pillar 0.31 — priced for a turnaround it hasn't shown |
| CLX | HIGH short interest + extreme leverage flags |
| GIS, BAX | Value traps: cheap but negative growth + heavy short interest |
| EL, DOW, LYB | Negative earnings growth, weak quality |
| **Short-selling any of these** | NOT actionable: no borrow-cost/availability data in this stack; the hypothetical momentum long-short leg was untradable as specified |
| **Leveraged "perfect setups"** | The single worst tested idea: −62% (§7) |

---

## Section 6 — Model portfolios (research illustrations, not advice)

**Conservative** — objective: sleep, capped drawdown (~−10% expected worst case):
- 70% SPY with the 200-DMA gate (exit to cash on close below; verified maxDD −19.8%
  research / −10.0% holdout vs −33.7% / −18.8% unhedged)
- 30% T-bills/cash. No individual stocks, no tilts.

**Balanced** — objective: market return, better risk shape:
- 60% SPY buy-and-hold (the benchmark nothing beat significantly)
- 25% momentum+low-vol top-decile sleeve, monthly rebalance (holdout Sharpe 1.44, beta 0.40)
- 15% cash. Optional: express the §2 weekly basket inside a 5% carve-out.

**Aggressive** — objective: max defensible expected return, **still unlevered**:
- 45% momentum top-decile, monthly rebalance (holdout CAGR 31.1%; but research-era
  alpha was −2.8%/yr — this sleeve's edge is NOT statistically proven)
- 35% SPY
- 15% 12-month VQM names (NEM, MU, EXE, BAC, AVGO, GOOGL — equal weight)
- 5% weekly system basket (§2)
- Leverage 0×: every leverage test amplified the negative-alpha episodes first.
  The aggressive portfolio takes more *breadth* risk, not borrowed risk.

---

## Section 7 — Strategy validation (the full evidence table)

**Protocol:** PIT S&P membership (survivorship measured: 47% of removed names would
be silently missing otherwise), signals at close T → executed at open T+1, 15 bps
one-way costs × two-sided turnover (portfolios) / 25 bps round-trip (per-trade
system), purged walk-forward with embargo, 2024–2026 never touched until this
final read-out.

**Cross-sectional portfolios vs SPY (after costs):**

| Strategy | 2012–23 CAGR | Sharpe | maxDD | NW α/yr (t) | 2024–26 CAGR | Sharpe | maxDD | NW α/yr (t) |
|---|---|---|---|---|---|---|---|---|
| SPY buy-hold | 13.9% | 0.86 | −33.7% | — | 21.5% | 1.31 | −18.8% | — |
| Momentum 12-1 top decile | 10.8% | 0.61 | −36.7% | −2.8% (−1.05) | **31.1%** | 1.31 | −22.8% | +4.3% (0.64) |
| Low-vol bottom decile | 10.8% | 0.79 | −34.8% | +1.7% (0.57) | 12.8% | 1.09 | −7.9% | +8.1% (1.30) |
| Mom+LowVol combined | 9.2% | 0.64 | −33.5% | −1.4% (−0.53) | 17.6% | **1.44** | **−9.3%** | +8.7% (1.45) |
| 5d-reversal weekly | 2.5% | 0.23 | −61.0% | −12.2% (−3.06) | 10.0% | 0.54 | −29.4% | −10.6% (−1.28) |
| SPY 200-DMA gate | 9.8% | 0.87 | **−19.8%** | +3.4% (1.29) | 17.1% | 1.38 | −10.0% | +4.8% (0.88) |
| Mom L/S (hypothetical, no borrow) | −6.0% | −0.13 | −66.2% | −0.1% (−0.02) | 21.6% | 0.92 | −17.6% | +14.3% (1.12) |

No alpha t-statistic reaches even 2.0, let alone the Chordia–Goyal–Saretto ≥3.4
hurdle this stack enforces for discovered rules. **Conclusion: statistically, SPY
buy-and-hold remains unbeaten in this sample.** Notable risk fact: momentum lost only
−5.4% in 2022 when SPY lost −18.2%, and won 2024/2025/2026 YTD (29.6% / 18.7% /
28.6% vs SPY 24.9% / 17.7% / 11.1%) — economically interesting, statistically
unproven on 2.5 holdout years.

**Machine learning by horizon (OOF, purged, pre-2024):** base rate selected at every
horizon; log-loss improvement 0.00000 at 1d/5d/10d/21d. **Rule discovery:** 0 stable
rules from ~23,000 honest trials across three nested windows; the pipeline provably
works (synthetic canary passes both directions). **Concentration+leverage:** −62%.
**Near-miss rule ensemble:** α = −12.2%/yr (t = −3.93) — significantly *worse* than
SPY, because the composite saturates (matches 72–95% of all rows).

**Known limitations (stated, not hidden):** free daily OHLCV only (no intraday,
options, insider, or PIT fundamentals); ~20 glitch label rows quarantined-by-median
but present in raw means; dividends approximated via adjusted close for portfolios,
excluded in per-trade labels; borrow costs unknown (all short results hypothetical);
2.5 years is a short holdout; the fundamentals composite has no historical vintages.

---

## Section 8 — Final action plan

**This week (from Thursday 2026-07-16):**
1. Core: hold/establish SPY exposure per chosen portfolio (§6). Regime is UP —
   timing the entry within the week is statistically irrelevant; do not wait for dips.
2. Optional 5% satellite: §2 basket at Thursday open, 10-session time exit, −6%/ATR
   stop, cancel-on-gap rule. Expected edge ≈ +0.6% per 10 sessions — treat as a
   live paper-validation of the frozen system, not as income.
3. Do nothing at the 1-day horizon. Place no leveraged trades.

**Three signals that would change the stance:**
1. **SPY closes below its 200-DMA** → halve gross exposure; second weekly close below
   → conservative portfolio. (Verified: this single gate was the best drawdown
   reducer in 14 years of data.)
2. **20-day realized vol > 20% while trendless** → suspend all dip-buying; the
   measured playbook flips (gap-down buys: 20% hit rate, −8.6% mean).
3. **Monthly momentum-decile turnover spikes with breadth collapse** (leaders
   rolling over) → exit the momentum sleeve to SPY/cash; momentum's failure mode is
   fast crash-reversal.

**Decision tree:**

```
Is SPY above its 200-DMA?
├─ NO → conservative portfolio only (gated SPY + cash). No stock picks. Re-check weekly.
└─ YES → Is 20d realized vol < 20%?
    ├─ NO → hold core SPY only; no satellites; no new entries until vol < 16%
    └─ YES (today) → run chosen §6 portfolio
        ├─ Weekly: rebalance §2 basket Thursdays (10-session cycle, stops as specified)
        ├─ Monthly: rebalance momentum/MOMLV sleeve at month-end close +1 open
        └─ Any single position −6% from entry OR flag triggered (§4/§5) → exit, no averaging down
```

**The honest bottom line:** this project tested the "selective sniper with leverage
beats passive" hypothesis with institutional discipline — nested validation, honest
trial counting, a held-aside historical partition, four cost scenarios — and the data said no.
What survives is humbler and real: own the market, gate the catastrophic drawdowns
with the 200-DMA, tilt modestly toward momentum+quality if you accept unproven-but-
positive holdout evidence, size small, and never leverage a signal whose conviction
score is 32/100. The system's greatest measurable edge was knowing when **not** to
trade.
