# Interpretable rule discovery (the rules campaign)

The campaign discovers human-readable trading rules and validates them with
the strictest battery in the platform. Method references: docs/references.md.
Everything below runs **inside each outer training window** — nothing is ever
chosen using outer-period data.

## Pipeline

```
EBM (effect shapes)  ->  RuleFit (candidate rules)  ->  stability selection
   -> distillation (small position policy)  ->  frozen  ->  outer-year trading
   -> SPA / Ledoit-Wolf / Newey-West / concentration -> verdict label
```

### 1. EBM — Explainable Boosting Machine (`discovery/ebm_shapes.py`)
Diagnostic only. Fits a GA2M regressor on the excess-net target and reports
per-feature importances, shapes and pairwise interactions. Features with no
stable effect are dropped from the search (shortlist). An EBM curve is never
itself a trading rule.

### 2. Constrained RuleFit (`discovery/rulefit_engine.py`)
In-house Friedman–Popescu implementation (imodels' alpha search was measured
at minutes per fit and replaced): shallow gradient-boosted trees generate a
candidate rule at **every node's path prefix** (so one-condition rules exist),
paths are simplified to tightest bounds, near-vacuous conditions (matching
>=90% of rows) are pruned, and a bounded-path Lasso keeps a sparse subset.
Constraints: <=3 conditions per rule, minimum support in rows AND unique
symbols, only registered causal features. **Every rule the ensemble generates
is counted as a trial**, filtered or not.

### 3. Stability selection (`discovery/stability.py`)
Within the training window: the first 80% of sessions is the discovery
region; RuleFit runs on 3 expanding windows x 3 symbol subsamples of it.
Rules canonicalize by structural signature — feature set + direction +
threshold *decile* — so `vol<22%`, `vol<24%`, `vol<25%` are one rule. The
final 20% of sessions (untouched by every discovery run) then scores each
structural survivor. Gates, all required:

- selection frequency >= 70% of runs, sign consistency >= 90%;
- inner-window t-stat of the DEMEANED edge (selected mean minus unconditional
  mean — matching Lasso-coefficient semantics) >= 3.4, the Chordia–Goyal–
  Saretto multiplicity-calibrated hurdle;
- inner direction agrees with discovery direction; minimum inner trade count;
- for tradeable (long) rules additionally: positive ABSOLUTE mean net of
  costs (a stable relative edge can still lose money).

### 4. Distillation (`discovery/distill.py`)
Stable-rule indicators + sign of excess-net feed the smallest adequate
policy: GOSDT -> CORELS -> imodels greedy rule list -> pruned depth-3 sklearn
tree (first importable wins; the verdict names which ran). Output: an ordered
position policy with no-trade / small / normal tiers.

### 5. Outer evaluation (`scripts/rules_campaign.py`)
The frozen policy trades the untouched outer year through the backtest
engine under conservative costs, judged against SPY, exposure-matched SPY and
the equal-weight point-in-time universe, with Hansen-SPA over the model set,
a Ledoit–Wolf-style Sharpe-difference test, Newey–West HAC alpha, and profit-
concentration falsification. Verdict labels: Rejected / Interesting
hypothesis / Promising structure, unproven edge. ("Supported edge" is
unreachable from three outer windows by design.)

## The canary (acceptance test of the apparatus)
`python scripts/rules_campaign.py --stage canary` — a synthetic market with
an injected turn-of-month drift must yield the rule back (measured: 89%
selection frequency, |t|=6.4), and a pure-noise market must yield ~zero
passing rules (measured: 0 of ~9,300 trials). This runs in CI as
`tests/statistical/test_rules_canary.py`.

## Running the campaign

```bash
python scripts/rules_campaign.py --stage prep
python scripts/rules_campaign.py --stage year --year 2021 --phase discover
python scripts/rules_campaign.py --stage year --year 2021 --phase trade
# ... repeat for 2022, 2023 ...
python scripts/rules_campaign.py --stage report   # artifacts/rules_campaign.json
```

"No stable rules found" is a valid, reportable outcome — per the replication
literature it is the expected one on free daily OHLCV data.

## Campaign result on the broad point-in-time universe (2021-2023)

Fully nested per outer year (~1M training rows, ~7,600-7,800 counted trials
per window): 224-291 structural rules per window and **zero passed the
stability + CGS gates in any window** — the campaign abstained all three
years. Verdict: **Rejected (no stable rules discovered)**
(`artifacts/rules_campaign.json`).

Strong-looking effects do appear in-window (deep-loser reversal and
volatility rules with inner |t| up to 19), but none rediscover consistently
across subsamples and windows — exactly the fragility the stability
literature predicts. The canary proves the apparatus detects real structure
when it exists, so this is the market's answer, not the machinery's.
