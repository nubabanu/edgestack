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

Android caches the canonical bundle and optional server preview. Profile fields migrate the former base-leverage value once into the 0–5× ceiling. Offline mode displays cached output and performs no signal or leverage calculation. Notifications compare server-owned status, targets, freshness, and risk state only. Calendar/overlay/ensemble engines and their alerts were removed.

## Known limitations

- Free Yahoo and FRED endpoints may be unavailable, restated, or delayed.
- Yahoo coverage is survivorship-biased and does not supply a point-in-time delisted universe.
- DGS3MO is a proxy; broker-specific margin terms can be materially worse.
- Daily bars cannot reconstruct intraday queue priority or market impact exactly.
- The baseline is diversified by instrument labels, not guaranteed economic risk parity.
- No sleeve is currently entitled to weight solely from a legacy status or report.
- V2 prospective stock evidence begins only with the first frozen V2 publication.
