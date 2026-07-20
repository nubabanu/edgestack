# Strange-edge campaign

This campaign preserves all nine standalone hypotheses and all five proposed
combinations from the supplied brief. It does not silently replace unavailable
alternative data with price features. Exact input deficits are persisted as
research campaigns and evidence gaps for the API and Android Edge Lab.

## Frozen scope

The manifest is [`configs/strange_edges.yaml`](../configs/strange_edges.yaml).
It was frozen before any return result was inspected and expands to 108
government-contract trials:

- three past-only surprise features;
- 30, 45, and 60 calendar-day availability lags;
- 5, 21, and 63-session holding periods;
- top 20% and top 30% equal-weight selections;
- raw and size/beta/momentum-neutralized scores.

Signals are observed only after the conservative availability lag, execute at
the next session's adjusted open, and include idle DGS3MO yield, turnover costs,
and overlapping-cohort accounting. The locked final holdout begins on
2023-01-01 and was not opened.

## Result

Run `3b9eeb0f5884873c6f3cb0de` completed all 108 trials on 1,440 monthly
USAspending rows for ten listed defense/industrial parents and 2014–2022 common
price sessions. No trial qualified:

- family-wide SPA consistent p-value: **0.6125**;
- StepM-superior candidates: **0**;
- shadow strategies started: **0**;
- promoted sleeves: **0**.

The highest raw result was a 60-day-lag, neutralized year-over-year surprise
held for 63 sessions. It showed 14.44% annualized historical log growth versus
9.50% for SPY and 11.62% for risk-matched SPY, but its benchmark-relative 95%
lower bound was -5.55% annualized, raw bootstrap p-value was 0.2924, FDR q-value
was 1.0, and maximum drawdown was -42.41%. That is an interesting watchlist
observation, not evidence of an edge.

The immutable report is stored at
`artifacts/research/strange_edges/3b9eeb0f5884873c6f3cb0de/report.json`. A
cache-only rerun reproduced the same run id and trial-result hash.

## Data boundary

USAspending's search API is unauthenticated and exposes monthly prime-contract
obligations, but the current response is a revised snapshot rather than a
historical publication-vintage archive. Recipient text also requires a verified
point-in-time subsidiary-to-parent map. Consequently this run was exploratory
and promotion-ineligible even if a candidate had cleared the statistical gates.

The remaining hypotheses were not numerically tested because one or more
pre-registered inputs are absent:

- licensed historical job postings, aggregate résumé revisions, app downloads,
  or TRACE transactions;
- historical GitHub adoption vintages and revenue-dependency mappings;
- historical flight tracks and point-in-time aircraft ownership;
- complete speaker-attributed call transcripts;
- patent-to-parent mappings combined with historical commercialization hiring.

The combinations remain blocked until every named component exists. Partial
components are not treated as a test of the stated interaction.

Provider references:

- [USAspending API endpoints](https://api.usaspending.gov/docs/endpoints)
- [GitHub releases API](https://docs.github.com/en/rest/releases/releases)
- [FINRA TRACE historic data](https://www.finra.org/industry/trace-historic-academic-data)
- [USPTO Open Data Portal](https://data.uspto.gov/)

## Operation

```bash
edgestack research strange-edges --config configs/default.yaml
```

The default reuses immutable request caches. `--refresh` deliberately acquires
a new provider snapshot and therefore produces a different input hash if the
source has changed. A rejected frozen family must not be retuned against the
same interval; materially new features require a versioned manifest.
