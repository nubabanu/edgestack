# Upstream research-system review

Reviewed 2026-07-20. EdgeStack adopts interfaces and validation ideas, not
upstream code or example market data. An upstream framework is not evidence
that a strategy works, and a sample dataset is not canonical research data.

| System | License / data reality | What EdgeStack takes | What remains deferred |
|---|---|---|---|
| [Microsoft Qlib](https://github.com/microsoft/qlib) | MIT. Its public README says the official dataset is currently disabled and the community Yahoo dataset can be imperfect; users are advised to build high-quality data. | Immutable datasets and handlers, feature definitions, recorder/registry concepts, point-in-time data, reproducible workflows, and rolling updates. EdgeStack already had most of this in `DataCatalog`, `FeatureRegistry`, model manifests, and the continuous worker. | Qlib sample/community data is not imported into promotion evidence. A Qlib adapter may be added if it preserves EdgeStack's guard and trial ledger. |
| [Microsoft RD-Agent](https://github.com/microsoft/RD-Agent) | MIT. Agentic factor/model R&D integrates with Qlib; generated ideas still depend on the supplied data and evaluation process. | `CandidateProposalV1`, complete parameter-grid registration, immutable attempt history, data-query logging, guarded-data disclosure, and read-only proposal/audit APIs. | Agents cannot run a final holdout, promote, or write canonical recommendations. Arbitrary generated code is not executed by proposal import. |
| [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) | LGPL-3.0-or-later. Deterministic event-driven trading platform; users provide venue and historical data adapters. | Separate event, receive, and process clocks; venue-qualified instruments; deterministic event ordering and content hashes. | No live adapter. A future simulator adapter must reconcile orders against EdgeStack's broker contracts. |
| [QuantConnect LEAN](https://github.com/QuantConnect/Lean) | Apache-2.0. Modular event engine with data feeds, universes, corporate actions, brokerage and buying-power models; useful datasets are provider- or user-supplied. | Reference behavior for symbol changes, corporate actions, exchange calendars, universe selection, order state, margin, and event types. | EdgeStack does not embed LEAN or claim its data library as free canonical evidence. |
| [hftbacktest](https://github.com/nkaz001/hftbacktest) | MIT. Accurate queue/latency simulation requires full order-book or trade-tick feeds; the project notes that accuracy does not automatically mean conservative fills. | Explicit `ExecutionDataCapabilitiesV1` gates: L2 is required for estimated queue position, L3 for exact queue position, and recorded receive/process timestamps for latency. | Queue replay stays disabled for Alpaca bar data. It becomes eligible only when licensed L2/L3 provenance and quality are present. |
| [VectorBT](https://github.com/polakowo/vectorbt) | Source-available under Apache 2.0 with Commons Clause, not plain Apache-2.0. Very fast vectorized analysis; examples and integrations do not constitute a vetted dataset. | Stage-zero vectorized triage, deterministic grids, parameter-surface checks, and placebos. Every triaged variant remains charged to its family. | No code is copied. Vectorized fills cannot establish execution eligibility; surviving candidates must enter event/paper simulation. |
| [PyBroker](https://github.com/edtechre/pybroker) | Apache 2.0 with Commons Clause. Callback-oriented backtesting and model caching; data can be supplied by the user or fetched through integrations. | Windowed train/predict callbacks, cached model artifacts, and bootstrap-oriented diagnostics are useful adapter patterns. | No code is copied and no second backtest engine is embedded while EdgeStack's existing engine covers the required daily workflow. |
| [Zipline Reloaded](https://github.com/stefan-jansen/zipline-reloaded) | Apache-2.0. Event-driven engine; research data must be ingested as bundles. | Reference behavior for calendars, corporate actions, and point-in-time event iteration. | A second daily event engine would duplicate current functionality; add only as a differential-test adapter. |
| [Alphalens Reloaded](https://github.com/stefan-jansen/alphalens-reloaded) | Apache-2.0. Consumes user-supplied factor and pricing data. | Native `analyze_factor`: forward-return quantiles, rank IC, group-neutral IC, turnover, monotonicity, decay horizons, deterministic hashes, and explicit trial charge. | Tear sheets are diagnostic only and cannot satisfy promotion gates. |
| [skfolio](https://github.com/skfolio/skfolio) | BSD-3-Clause. Scikit-learn-style portfolio optimization and model selection over user-supplied returns. | Promoted-only allocation candidates, Ledoit-Wolf covariance, stressed covariance, family/per-sleeve caps, expanding cross-validation, expected shortfall, correlation, and effective breadth. | Portfolio optimization cannot rescue weak alpha. More complex HRP/HERC/NCO and stacking remain deferred until multiple independently promoted sleeves exist. |
| [ABIDES](https://github.com/jpmorganchase/abides-jpmc-public) | BSD-3-Clause. Synthetic agent-based market simulation, not historical evidence. | Scenario and causal microstructure canaries are useful future stress tests. | Synthetic P&L cannot promote a historical strategy or substitute for venue data. The upstream public repository is archived. |
| [MlFinLab](https://github.com/hudson-and-thames/mlfinlab) | The current public repository says it exists to receive bugs, feature requests, and issues; it is not treated as a market-data source or a vendorable implementation. | Method names are useful literature pointers only; EdgeStack implements independently from papers when a method is justified and testable. | No code, datasets, or proprietary snippets are copied. |
| [FinRL](https://github.com/AI4Finance-Foundation/FinRL) | MIT. Research/education framework with Yahoo, Alpaca and other connectors; the original stack is not itself a validated execution or promotion process. | The train/validate/trade separation and environment contract are useful design references. | Reinforcement learning is deferred until state features, execution fidelity, and prospective validation are reliable. RL does not create information absent from the data. |

## Adopted contracts

The review produced four additions rather than another monolithic framework:

1. `research/proposals.py` accepts JSON-only finite hypotheses, registers all
   parameter combinations before evaluation, records every attempt or
   revision, and sends missing data through the existing coverage planner.
2. `research/factor_diagnostics.py` performs fast cross-sectional diagnostics
   on explicitly supplied point-in-time factor and forward-return columns. Its
   output declares `promotion_evidence=false`.
3. `execution/events.py` supplies deterministic three-clock event replay and
   rejects execution models unsupported by the available feed.
4. `recommendation/ensemble.py` accepts only independently promoted,
   shadow-qualified sleeves with positive lower-confidence returns. It compares
   three fixed allocators out of sample and returns no ensemble when the best
   lower-confidence log growth is non-positive.

These additions strengthen the search process without changing the core rule:
more data and more hypotheses are welcome; hidden trials, reused holdouts, and
unsupported execution assumptions are not.
