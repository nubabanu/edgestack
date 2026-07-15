# Statistical validation

## Why so much machinery

Searching a rule space over one historical path manufactures false positives.
The pipeline is built so that the *default* outcome of testing noise is
rejection, verified by the statistical acceptance suite
(`tests/statistical/`): a pure random walk yields zero validated edges; a
strong injected calendar effect is detected; an effect present only in the
training era is rejected out of sample; an effect smaller than conservative
costs is refused.

## Walk-forward splits with purging and embargo

Sessions are split into an expanding (or rolling) train window followed by
`n_folds` consecutive test windows. A training row is **purged** when its
outcome interval `[signal date, label_end]` overlaps the test window —
interval-based, so variable-length triple-barrier labels are purged exactly,
where a fixed offset would under-purge. An **embargo** additionally drops
training rows whose signal date falls within `embargo_sessions` after any
earlier test window (default 60 = the longest horizon). The config refuses an
embargo shorter than the longest signal horizon.

## Per-candidate out-of-sample evaluation

For each fold, quantile thresholds are fit on the fold's TRAIN rows only
(`QuantileBinner`), the rule is evaluated on the TEST rows, and the matched
gross returns are converted to net returns under each cost scenario. Pooled
OOS net returns feed:

- **bootstrap p-value** for `H0: mean = 0` — the sample is centered to the
  null, resampled (vectorized), `p = (#{mean* >= mean_obs} + 1)/(n_boot + 1)`;
- **circular block-bootstrap CI** (default block 20 sessions) — honest about
  serial dependence from overlapping windows;
- **shrinkage posterior**: conjugate normal update with a zero-mean prior
  worth `prior_pseudo_n = 30` observations:
  `post_mean = n·x̄ / (n + n0)`, `var(post_mean) = s² / (n + n0)`;
- **effective sample size**: `ESS = n / (1 + 2·Σ ρ_k)` with the
  autocorrelation sum truncated at the first non-positive lag;
- **subperiod sign consistency** across four contiguous chunks.

## Multiple-testing control

All candidates in a discovery batch — including the ones that never had
enough support — are persisted with the batch, and the batch size is the
trial count. Benjamini-Hochberg adjusted p-values (q-values) are computed
across the whole batch (`q_(i) = min over j>=i of p_(j)·m/j`);
Benjamini-Yekutieli (`× Σ 1/k`) is available for arbitrary dependence.
Validation refuses candidates that lack a batch id.

## Deflated Sharpe ratio

Probabilistic Sharpe ratio (per-period SR, non-annualized):

```
PSR(SR*) = Φ( (SR − SR*)·√(n−1) / √(1 − γ₃·SR + (γ₄−1)/4·SR²) )
```

with sample skew γ₃ and kurtosis γ₄. The deflation benchmark is the expected
maximum SR among N zero-skill trials:

```
SR* = √V · ( (1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)) ),  γ = 0.5772…
```

where V is the cross-trial SR variance — approximated by `1/n` when unknown
(documented approximation). `DSR = PSR(SR*)`; an edge needs `DSR >= 0.5`.

## Gates for VALIDATED status (all must pass)

1. `q <= fdr_alpha` (default 0.05);
2. `ESS >= min_effective_sample_size` (default 100);
3. net mean >= `min_expected_net_return` under CONSERVATIVE costs
   (economic, not just statistical, significance);
4. survives CONSERVATIVE costs (positive net mean);
5. positive in >= `min_subperiod_consistency` of walk-forward folds;
6. `DSR >= 0.5` given the batch's full trial count.

Failures are recorded verbatim in `lifecycle.failure_reasons`. "No edge
survived" is a legitimate, common outcome.

## The final untouched test period

Everything above runs strictly before `validation.final_test_start`. The
`TestPeriodGuard` truncates all catalog reads at that boundary; access
requires an audited, single-use unlock key, and the integration suite asserts
zero `test_set_accessed` rows for a full research run. Evaluate on the final
period once, after freezing the procedure — the audit trail shows whether
that discipline was kept.
