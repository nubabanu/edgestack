# Probability and conviction

Every candidate carries two separate numbers. Conflating them is exactly the
failure mode the spec forbids ("do not call an arbitrary score a probability").

## 1. Calibrated probability of positive net return

`P(net trade return > 0 | information at signal time)`.

- Source: the selected model for the (horizon, side) — a base-rate floor,
  regularized logistic, or shallow gradient boosting; a more complex model is
  used only when it beats the simpler one by ≥1% out-of-fold log loss.
- Calibration: isotonic regression fit on **out-of-fold** predictions from
  the purged walk-forward splits (Platt scaling available). In-sample
  calibration would only launder overconfidence.
- Diagnostics stored per model: Brier score `mean((p−y)²)`, log loss,
  expected calibration error `Σ_b (n_b/n)·|p̄_b − ȳ_b|`, and reliability-
  diagram bins.
- Fallback: with no applicable model, the evidence-weighted historical hit
  rate of the matched edges is used and the candidate is flagged
  `probability_is_calibrated = false`.

## 2. Conviction score (0–100)

Summarizes whether the signal is trustworthy and economically actionable.
Implemented as a pure function (`edgestack.scoring.conviction`), fully
property-tested; 50 is neutral "no usable edge".

```
prob_edge   = clip(2·(p − 0.5), 0, 1)
econ_edge   = clip(E[net] / 0.01, 0, 1)          # 1% per trade saturates
base_edge   = prob_edge · econ_edge

reliability   = mean(fold stability, OOS score, deflated Sharpe)     ∈ [0,1]
applicability = regime similarity                                    ∈ [0,1]
tradability   = mean(liquidity, data quality, cost-scenario survival)∈ [0,1]

uncertainty = clip(CI width / (6·|E[net]|), 0, 1)
tail        = clip(|ES₉₅| / 0.10, 0, 1)
disagreement= |model P − edge hit rate|
risk_penalty= 0.5 · clip(0.4·uncertainty + 0.3·tail + 0.3·disagreement, 0, 1)

raw    = base_edge · reliability · applicability · tradability − risk_penalty
shrunk = raw · ESS / (ESS + 30)                  # small samples → neutral
score  = 100 / (1 + exp(−6 · shrunk))            # raw 0 → exactly 50
```

Multiplicative structure means any weak link (bad regime match, illiquidity,
cost fragility) collapses the score; penalties and shrinkage make the score
drop sharply for wide intervals, small samples, model/evidence disagreement
and tail risk — each behavior is asserted by tests. All intermediates are
returned so explanations can show the arithmetic.

## Family capping (no double counting)

Evidence items are grouped into families (trend, momentum, mean-reversion,
volatility, volume, structure, calendar, event, regime, sector). Within a
family the combined contribution is `max + 0.25·mean(rest)`, hard-capped at
`family_cap = 0.4`; only across families do contributions add. Five momentum
indicators agreeing is therefore worth barely more than one — enforced by a
test that five same-family clones cannot out-score five diverse families.
