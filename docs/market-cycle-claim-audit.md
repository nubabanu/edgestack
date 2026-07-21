# Market-cycle claim audit

Audited 2026-07-21 from the supplied weekly, daily, and confluence narrative. A
mechanism being real does **not** establish that the proposed trading rule is
predictive after costs. Claims of novelty ("nobody has studied this", "virgin
territory") are withdrawn unless a systematic literature review can establish
them.

Status vocabulary:

- `SUPPORTED` — the stated empirical fact or operational mechanism has direct evidence.
- `PARTLY_SUPPORTED` — a narrower mechanism is documented; the proposed edge is not.
- `OUTDATED_OR_FALSE` — a material factual premise is obsolete or contradicted.
- `UNVERIFIED_HYPOTHESIS` — plausible enough to test, but not established here.
- `UNTESTABLE_WITH_CURRENT_DATA` — EdgeStack lacks the point-in-time data needed for a fair test.

## Weekly claims

| ID | Claim | Status | Evidence audit | Data / EdgeStack treatment |
|---|---|---|---|---|
| W01 | Recurring-investment platforms overwhelmingly default weekly purchases to Monday, creating a Monday-open bid. | `OUTDATED_OR_FALSE` | Major brokers document user-selected frequency and start day; Monday is an example, not a demonstrated market-wide default ([Fidelity](https://www.fidelity.com/learning-center/trading-investing/recurring-investment), [Robinhood](https://robinhood.com/us/en/support/articles/recurring-investments/)). | Broker-level scheduled-order data would be required. No code mapping. |
| W02 | Robo-advisers rebalance on a weekly cadence and create an early-week counter-flow after large moves. | `OUTDATED_OR_FALSE` | Betterment documents drift thresholds, cash-flow-aware rebalancing, and approximately daily opportunity checks, not a general weekly clock ([method](https://www.betterment.com/help/portfolio-rebalancing), [disclosure](https://www.betterment.com/legal/auto-adjust-disclosure)). | Adviser transaction data is unavailable. `post_down_month` is only a price/calendar feature, not robo flow. |
| W03 | Short sellers cover Friday to avoid weekend borrow/headline risk and re-short Monday. | `UNVERIFIED_HYPOTHESIS` | Borrow cost and weekend risk provide a possible motive, but they do not establish a recurring cover/re-entry pattern. | Needs point-in-time securities-lending balances, fees, and intraday short-sale data. |
| W04 | Custodian recall processing causes Thursday/Friday squeezes in hard-to-borrow stocks. | `UNVERIFIED_HYPOTHESIS` | No evidence found for a stable weekly operational cycle or tradable effect. | Needs recalls, loan availability, corporate actions, and intraday prices. |
| W05 | Option-implied volatility falls Friday and rises after the weekend; regime-conditioned trading may exploit mispricing. | `PARTLY_SUPPORTED` | Calendar-day treatment can mechanically produce a Friday/Monday volatility pattern; Cboe explicitly describes a Friday effect in volatility calculations ([Cboe note](https://cdn.cboe.com/resources/indices/documents/Cboe_USO_ImpliedCorrelation_0421_v2.0.2.pdf)). Profitability and calm/stress miscalibration remain unverified. | Requires historical option surfaces and executable quotes; unavailable in the current free stack. |
| W06 | 0DTE pinning is strongest Mon/Wed/Fri and weakest Tue/Thu because only the former are major expiry days. | `OUTDATED_OR_FALSE` | SPX now has same-day expirations every trading day; Tuesday/Thursday expirations were added in 2022 ([Cboe](https://www.cboe.com/tradable-products/0dte)). | A modern test needs daily option chains and trade-classified positioning. No WIND mapping. |
| W07 | Friday charm mechanically predicts a 14:00–16:00 drift toward large open-interest strikes. | `PARTLY_SUPPORTED` | Expiration pinning has academic support, but open interest alone does not identify dealer direction and daily expirations weaken a Friday-only premise ([Avellaneda and Lipkin](https://www.tandfonline.com/doi/abs/10.1088/1469-7688/3/6/301)). | Needs intraday option inventory/flow and index futures; not current-data testable. |
| W08 | Retail attention peaks Monday and makes retail-favorite stocks rise early week and fade late week. | `UNVERIFIED_HYPOTHESIS` | General attention effects do not establish this basket-specific weekly return rule. | Needs a frozen retail-favorite universe plus point-in-time app/social attention data. |
| W09 | Weekend Bitcoin direction predicts Monday equity gaps only in high-correlation regimes. | `UNVERIFIED_HYPOTHESIS` | This is a well-formed conditional hypothesis, not an established result. | Requires continuous crypto bars, frozen regime rules, and equity opening prices; EdgeStack has no crypto feed. |
| W10 | Friday-after-close corporate news that retains weekend attention predicts larger Monday gaps. | `UNVERIFIED_HYPOTHESIS` | Release timing is observable, but the proposed weekend-attention classifier is not validated. | Needs timestamped filings/news and point-in-time social/search data. EDGAR alone is insufficient. |
| W11 | Weekly Treasury-bill settlements create Tuesday/Thursday funding waves and tradable rate-ETF effects. | `PARTLY_SUPPORTED` | Bill issue/settlement days do recur on Tuesday or Thursday depending on tenor; the downstream alpha claim is unverified ([TreasuryDirect](https://www.treasurydirect.gov/auctions/general-auction-timing/)). | Needs issue amounts, funding rates, and preregistered rate-ETF tests. |
| W12 | Less-liquid ETF premiums widen through the week and compress Friday as authorized participants square books. | `UNVERIFIED_HYPOTHESIS` | Creation/redemption arbitrage is real, but a Friday book-squaring accordion was not established. | Needs timestamped indicative NAV, quotes, spreads, creations, and redemptions. |
| W13 | A public weekly “tide table” does not exist. | `UNVERIFIED_HYPOTHESIS` | Absence and novelty cannot be established by an informal search. | Withdraw the novelty claim; a descriptive chart would not itself be evidence. |
| W14 | Tiny weekly effects are useful mainly for timing an existing purchase because round trips consume the edge. | `PARTLY_SUPPORTED` | The cost principle is sound, but the claimed basis-point magnitude and benefit of deferral must be measured for each rule. | The new prospective paired-fill shadow tests exactly one-session deferral. |

## Daily and intraday claims

| ID | Claim | Status | Evidence audit | Data / EdgeStack treatment |
|---|---|---|---|---|
| D00 | German local time is always ET plus six hours. | `OUTDATED_OR_FALSE` | The offset is usually six hours but temporarily five because US and European daylight-saving transitions occur on different dates. | Store timestamps with `America/New_York` and convert by timezone, never a fixed offset. |
| D01 | A large share of long-run US index returns accrued close-to-open rather than during regular hours. | `SUPPORTED` | Positive overnight and weak/negative intraday index returns are documented historically ([ETF/futures study](https://www.sciencedirect.com/science/article/pii/S1059056016301563)). Persistence, taxes, and implementability are separate questions. | Daily OHLC can reproduce the historical decomposition; any new rule remains previously accessed. |
| D02 | Overnight-only futures exposure becomes profitable after conditioning on weekday and event nights. | `UNVERIFIED_HYPOTHESIS` | The base overnight pattern does not validate these interactions or a futures strategy. | Needs continuous futures, rolls, margin, financing, and event-time data. |
| D03 | No-news opening gaps fade at a materially different rate from headline gaps. | `UNVERIFIED_HYPOTHESIS` | The proposed classifier is sensible but has not been validated here. | Opening-fade infrastructure exists; honest testing needs point-in-time news coverage and next-open execution. |
| D04 | Retail overnight orders exhaust near 09:45 and cause social-attention-scaled reversals by 10:15. | `UNVERIFIED_HYPOTHESIS` | Neither the exact exhaustion time nor the proposed causal attribution is established. | Needs order-origin classification, social data, and minute bars. |
| D05 | The first half-hour market return predicts the last half-hour return. | `SUPPORTED` | Documented for SPY and other active ETFs, with stronger effects on volatile/high-volume days ([Gao et al.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2440866)). Current profitability still requires replication. | Current free intraday history is too short; collect forward without backfilling claims. |
| D06 | Lunch breakouts fail and reverse more often than otherwise identical breakouts. | `UNVERIFIED_HYPOTHESIS` | Intraday volume commonly has a U shape, but that does not establish this breakout rule. | Requires minute bars, causal breakout definitions, spreads, and multiple-testing control. |
| D07 | European-market closure causes a predictable 11:00–11:45 ET move in EU-sensitive US stocks. | `UNVERIFIED_HYPOTHESIS` | A flow change at the European close is plausible; direction and tradability are unproven. | Requires timezone-aware European calendars, revenue exposures, FX, and intraday bars. |
| D08 | Distance from 14:00 VWAP predicts afternoon reversion because execution algorithms trade toward VWAP. | `UNVERIFIED_HYPOTHESIS` | VWAP benchmarking exists, but “magnetism” and the stated causal direction require order-flow evidence. | Historical consolidated volume and executable minute bars are required. |
| D09 | Public open interest provides a clean daily long-/short-dealer-gamma flag that flips momentum versus mean reversion. | `PARTLY_SUPPORTED` | Dealer hedging can dampen or amplify moves, but public open interest has no holder-side identity and is updated with a lag. The existing score never measures gamma. | Rename the existing feature `trend_vol_regime`; genuine GEX work needs option chains plus explicit positioning assumptions or classified trades. |
| D10 | NYSE publishes a closing imbalance at 15:50 that predicts closing drift and next-morning reversal. | `PARTLY_SUPPORTED` | NYSE imbalance dissemination begins at 15:50 and updates every second, but it covers the NYSE auction feed and is not a universal free signal; predictability remains unverified ([NYSE](https://www.nyse.com/trade/auctions)). | Needs archived imbalance messages, quotes, auction prints, and venue-aware symbols. |
| D11 | Daily-reset leveraged ETFs must rebalance with the day’s direction near the close, amplifying large moves. | `PARTLY_SUPPORTED` | Daily exposure reset and same-direction adjustment are documented in fund filings; the proposed ±1.5% threshold and time-varying return edge are not ([SEC-filed prospectus](https://www.sec.gov/Archives/edgar/data/1424958/000119312517018182/d306485d485apos.htm)). | Needs historical AUM/exposure estimates and intraday market-impact controls. |
| D12 | A second imbalance-driven drift reliably occurs around 15:54–15:57. | `UNVERIFIED_HYPOTHESIS` | The precise “second wind” is asserted without evidence. | Requires second-level NYSE imbalance and trade data; unavailable. |
| D13 | The same price move has different meaning across the intraday liquidity landscape. | `PARTLY_SUPPORTED` | Trading volume and activity commonly follow strong time-of-day patterns, including high activity near the open and close ([Block, French, and Maberly](https://www.sciencedirect.com/science/article/abs/pii/S0148296399000235)). Individual directional rules still need tests. | Treat time windows as features, not automatic signals. |
| D14 | Daily stock round trips below a universal 0.2% edge are uneconomic, while futures are effectively free. | `PARTLY_SUPPORTED` | Costs and slippage can erase small effects, but neither the 0.2% cutoff nor “free” futures trading is universal. | Use instrument-specific spread, commission, financing, tax, and impact sensitivity. |

## Confluence and WIND claims

| ID | Claim | Status | Evidence audit | Data / EdgeStack treatment |
|---|---|---|---|---|
| C01 | First session of a month that is Monday after a down month creates five independent long tailwinds. | `UNVERIFIED_HYPOTHESIS` | Several labels are proxies or overlapping calendar stories; independence and causal flow counts were not demonstrated. | Legacy mappings are `turn_of_month`, `post_down_month`, `weekday_reversal`, `trend_vol_regime`, and `short_term_dip`. |
| C02 | A stressed Friday afternoon creates four independent headwind families. | `UNVERIFIED_HYPOTHESIS` | Dealer position, attention decay, and risk shedding are unobserved; “thinning into the close” also conflicts with documented high closing activity. | Daily WIND cannot test an afternoon window. |
| C03 | A ±1.5% day plus short dealer gamma plus aligned 15:50 imbalance predicts continuation and next-day reversal. | `UNTESTABLE_WITH_CURRENT_DATA` | The stack requires intraday return, classified dealer positioning, and archived auction imbalance data. | Do not proxy these inputs with daily trend/volatility. |
| C04 | Panic near quarter-end plus rebalancing mandates is the strongest rare-event alignment. | `UNVERIFIED_HYPOTHESIS` | Historical anecdotes such as March 2020 do not identify a repeatable rule. | Needs preregistered event definitions and cross-asset allocation-flow data. |
| C05 | One ±1/0 vote per mechanism family prevents double counting. | `PARTLY_SUPPORTED` | Grouping related inputs is good discipline, but equal votes do not prove independence; legacy `turn_of_month` and `post_down_month` have roughly 0.5 correlation. | Report correlations and leave-one-out results; call them components, not independent mechanisms. |
| C06 | Trade small at score ≥2 and full size at score ≥4. | `OUTDATED_OR_FALSE` | The implemented score≥2 and score≥3 rules failed the repository’s old survivor bar. Higher-score buckets are sparse and unstable. | No sizing, alerts, tickets, leverage, or promotion may use WIND. |
| C07 | WIND return means increase monotonically with score. | `OUTDATED_OR_FALSE` | Exact buckets are non-monotonic in every SPY and QQQ split; the old report hid extremes by clipping them into tail buckets. | The audit report emits exact buckets and `strict_monotonic=false`. |
| C08 | Avoiding score≤−1 days validates postponing planned buys. | `OUTDATED_OR_FALSE` | Only a QQQ all-in/all-out daily switching rule passed historically; SPY did not, and switching exposure is not a delayed-purchase experiment. | Watcher output is `DESCRIPTIVE_ONLY`/`NO_ACTION`; the prospective one-session paired-fill shadow addresses the actual decision. |
| C09 | The original file constitutes a preregistration. | `OUTDATED_OR_FALSE` | The rules were documented at evaluation time but lack an independently timestamped, result-blind registration. | Legacy results are `previously_accessed=true` and `promotion_eligible=false`. Future experiments use the proposal ledger before outcomes exist. |

## Operational conclusion

WIND remains a neutral historical diagnostic. It must not produce a trade,
position size, leverage decision, alert, order ticket, or recommendation. The
only new evidence collection is the unlevered prospective paired-fill shadow;
even a positive result merely becomes review-eligible after at least 252
prospective sessions and 30 completed negative-score events.
