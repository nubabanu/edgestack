# Recommendation Engine V2 architecture and operations

> Research and paper trading only. No live broker adapter exists. The initial allocation is a policy baseline, not an alpha claim.

## Authority and data flow

`CanonicalRecommendationBundleV2` is the sole recommendation authority. It contains the fixed baseline policy, profile-independent `BaseRecommendationV2`, default `RiskProfileV2`, and default `PortfolioRecommendationV2`. A bundle validator requires identical session, as-of time, execution time, data version, artifact version, policy version, base weights, and default-profile hash across its children.

```text
point-in-time data + frozen artifacts
              |
       profile-independent base
              |
     uniform risk/leverage sizing
              |
 CanonicalRecommendationBundleV2
      /       |        |       \
    API    Android   dashboard  paper
```

Legacy board, picks, master, signals, and candidates are projections of this object. They do not load legacy signal reports. Board rows/picks are empty and research ideas appear only in `watchlist_v2`.

## Status semantics

- `ACTIVE`: at least one promoted sleeve has nonzero unlevered base weight.
- `WATCHLIST_ONLY`: the baseline is actionable and current zero-weight research ideas exist.
- `BASELINE_ONLY`: the baseline is actionable and no current idea is publishable.
- `NO_ALLOCATION`: increases are forbidden. Invalid artifacts/baseline data or a drawdown latch target cash. Stale but compatible data may preserve or reduce prior positions only.

## Research artifacts

An `ExperimentManifestV2` hashes code revision, data/universe definitions and hashes, point-in-time coverage, features, labels/horizons, complete candidate family, execution/cost/financing rules, folds, seeds, outputs, and policy. Candidates must enter the trial ledger before evaluation. Failed, rejected, cached, threshold, execution, optimizer, ensemble, interaction, voting, and other combination trials stay in the searched family.

Frozen artifacts are content-addressed and carry manifest, data, feature, policy, horizon, and payload checksums. Loading fails if any requested context differs. Stable semantic hashes exclude publication-generation time; timestamps remain in audit/publication records.

## Nested validation

Outer folds expand annually, require at least 756 training sessions, and test the following calendar year. A fold is valid only with at least 200 test sessions, compatible artifacts, executable benchmark/baseline data, and a resolved portfolio return. Five valid folds are mandatory and folds receive equal weight for the 70% positive-fold gate.

Discovery, parameters, thresholds, compounds, execution, calibration, and optimizer selection occur inside each outer training period. Calibration evaluates calibrators trained on other out-of-fold predictions, then fits the live calibrator on all training OOF predictions after metrics are reported. Outer-test mutation tests verify frozen decision hashes do not change.

Concurrent trades collapse into one portfolio contribution per session. Inference concatenates outer-test daily returns and uses session ESS, stationary 20-session blocks, deterministic 2,000-replication bootstraps, SPA over the complete searched family, and StepM for individual superiority. Risk-matched benchmarks use scaling estimated only in the corresponding outer training period.

## Portfolio construction

Sleeve expected returns shrink toward zero by ESS and allocation uses their 95% lower-confidence active return after modeled costs. Baseline alpha is zero. If no promoted sleeve has positive lower-confidence utility, baseline weights remain exactly 25% each.

Covariance blends deterministic Ledoit–Wolf estimates over 60, 126, and 252 sessions, stresses short-window volatility/correlation, and projects to positive semidefinite form. The long-only optimizer maintains unit unlevered gross exposure and charges risk, uncertainty, turnover, spread, impact, and concentration. Added assets are financed by reducing baseline weights.

Default limits are 3% per stock, 20% per equity sector, 10% of 60-day median ADV traded per session, liquidation within five sessions at that rate, and no shorts. Broad policy ETFs are exempt from stock/sector caps but not liquidity or policy limits.

## Risk, drawdown, and funding

The risk overlay computes independent leverage limits for stressed volatility, the worse parametric/historical 99.5% one-day loss, 99th-percentile 20-session path drawdown, liquidity, post-leverage name/sector concentration, freshness, DGS3MO freshness, drawdown state, user cap, and 5× system cap. The minimum is applied uniformly to every base weight. Schema invariants reject profile-dependent asset selection or tilting.

The drawdown state machine is `NORMAL → DELEVERAGING → CASH_LATCHED → RESET_ELIGIBLE`. Deleveraging begins at half the user limit and declines linearly to zero at the full limit. Reset eligibility needs 20 distinct sessions, recovery below half-limit, fresh compatible data, acceptable stress, and healthy monitoring. Preview reset is stateless; the CLI performs the persisted atomic reset.

Positive cash earns DGS3MO. Negative cash costs DGS3MO plus the user spread using actual calendar-day gaps. The recommendation and paper ledger expose funding, dividends, splits, transaction costs, stresses, actual fills, and actual-fill returns.

## Atomic operation

Each run contains recommendation, risk inputs, paper state, monitoring, compatibility projections, version set, publication record, and checksums. Readers resolve exactly one `current.json` pointer and verify the entire checksum manifest before parsing any child file. Paper execution creates a new content-addressed run and pointer; it never mutates the current run in place.

The nightly sequence is data update/quality, features, artifact/policy verification, base assembly, prior paper execution, risk sizing, target queueing, monitoring, bundle validation, run finalization, and pointer swap. Any exception aborts before the swap.

## Android behavior

Android caches the canonical bundle, optional server preview, and last server-generated instrument analysis. Its Analyze screen accepts a ticker/commodity proxy and optional timezone-qualified intended entry. It shows best/worst windows, exits, effects and their counter-effects, evidence, news context, cautions, and whether all four horizons are genuinely promoted and aligned. The device never calculates a signal, rating, or window.

Profile fields migrate the former base-leverage value once into the 0–5× ceiling. Offline mode displays cached output and performs no signal or leverage calculation. Notifications compare server-owned status, targets, freshness, and risk state only. Calendar/overlay/ensemble engines and their alerts were removed.

## User-selected instrument timing

`POST /instruments/analyze` is version-bound to the current canonical publication and cannot mutate portfolio selection, promotion, paper targets, or risk state. It resolves common commodity requests to tradable proxies and discloses the mapping. Daily adjusted-open total returns support day/week/month/year studies; separately stored timezone-aware 60-minute and 15-minute bars support intraday entry and exit studies. Daily data is never used to infer an hour or 15-minute slot.

For a user-selected time, the response rates the nearest compatible 15-minute slot, hour, weekday under each holding horizon, month bucket, and month of year. Each rating includes rank, alternatives, cost-adjusted net return, confidence bound, ESS, and a win score. Win score shrinks observed net win frequency toward 50% and pulls it further toward 50 as ESS falls. It is descriptive and is never presented as a forecast probability.

Conditional exit maps search exit slots for same-day, one-week, one-month, and one-year holding offsets. Same-day uses 15-minute data; longer horizons use hourly data. Fewer than 20 compatible observations produces no exit time. Every entry/exit pair remains in the searched family.

`POST /instruments/recheck` repeats the analysis against the current canonical version and compares server-owned ratings and alternatives. The returned schedule is daily more than 30 days away, six-hourly within 30 days, hourly within seven days, and 15-minute within one day. Android schedules one-shot WorkManager rechecks from that server cadence and notifies only when the choice deteriorates or a better alternative appears.

`POST /instruments/pattern-leaders` ranks an explicit bounded universe (up to 50 symbols), including common SPY/QQQ/GLD/USO and large-cap watchlists. The entire cross-symbol/slot scan is disclosed as watchlist research and cannot promote a sleeve.

The response separates three concepts:

- descriptive conditions such as trend, momentum, mean reversion, and volatility, which have zero actionable contribution until promoted;
- searched historical timing windows, reported after costs with session ESS, confidence bounds, and family-size p-value adjustment, but labeled observational;
- frozen timing artifacts that already passed nested selection, complete-family inference, cost/execution stress, ablation, and promotion. Only these can be actionable.

Every edge reports the adverse effect of an apparent tailwind and the protective/avoidance value of a headwind. An `ALL PROMOTED HORIZONS ALIGNED` setup requires compatible promoted day, week, month, and year timing artifacts, at least one promoted positive contribution, and no promoted negative contribution. Absence of hourly data, frozen news, or promoted evidence yields an explicit abstention—not a guessed hour, sentiment, or “perfect” buy/sell time.

News is a frozen, timestamped context input with source, age, and freshness. Its actionable contribution is structurally fixed at zero unless a future promoted artifact explicitly validates a point-in-time news feature.

## Known limitations

- Free Yahoo and FRED endpoints may be unavailable, restated, or delayed.
- Yahoo coverage is survivorship-biased and does not supply a point-in-time delisted universe.
- DGS3MO is a proxy; broker-specific margin terms can be materially worse.
- Daily bars cannot reconstruct intraday queue priority or market impact exactly.
- Free hourly and 15-minute history is limited and can be throttled; no intraday slot is reported when it is absent. Yahoo requests are capped at 729 and 59 calendar days respectively.
- News context may be absent, delayed, duplicated, or wrong and does not change the score by default.
- The baseline is diversified by instrument labels, not guaranteed economic risk parity.
- No sleeve is currently entitled to weight solely from a legacy status or report.
- V2 prospective stock evidence begins only with the first frozen V2 publication.
