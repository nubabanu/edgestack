"""Walk-forward validation of candidate edges → Edge records.

For every candidate in a discovery batch:

1. purged walk-forward folds over the (features ⨝ labels) frame;
2. per fold: quantile thresholds fit on TRAIN only, rule evaluated on TEST,
   producing out-of-sample net-of-cost trade returns;
3. pooled OOS returns → distribution stats, bootstrap p-value, shrinkage
   posterior, cost-scenario survival, regime breakdown, fold stability;
4. batch-level BH/BY FDR over all candidates' p-values, deflated Sharpe with
   the batch's full trial count;
5. gates → VALIDATED or REJECTED (with recorded failure reasons).

No candidate can be validated outside a batch: the trial count is the batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC

import numpy as np
import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.discovery.candidate_generation import (
    CONTINUOUS_RULE_FEATURES,
    DiscoveryBatch,
)
from edgestack.discovery.conditions import evaluate_condition
from edgestack.discovery.event_studies import study_returns
from edgestack.discovery.multiple_testing import fdr_adjust
from edgestack.execution.costs import CostModel
from edgestack.features.binning import QuantileBinner
from edgestack.types import (
    CandidateEdge,
    CostScenario,
    Edge,
    EdgeIdentity,
    EdgeLifecycle,
    EdgeRobustness,
    EdgeStats,
    EdgeStatus,
    Side,
    condition_features,
)
from edgestack.validation.metrics import deflated_sharpe_ratio
from edgestack.validation.splits import Fold, PurgedWalkForwardSplitter


@dataclass
class _CandidateOutcome:
    candidate: CandidateEdge
    returns_by_scenario: dict[CostScenario, np.ndarray]
    dates: np.ndarray
    fold_net_means: tuple[float, ...]
    regime_performance: dict[str, float]
    p_value: float


def _prepare_horizon(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    horizon: int,
    splitter: PurgedWalkForwardSplitter,
) -> tuple[pd.DataFrame, list[Fold]]:
    h_labels = labels.loc[labels["horizon"] == horizon]
    merged = features.merge(
        h_labels[["symbol", "date", "label_end", "gross_ret"]],
        on=["symbol", "date"],
        how="inner",
    ).reset_index(drop=True)
    folds = splitter.split_frame(merged["date"], merged["label_end"])
    return merged, folds


def validate_batch(
    batch: DiscoveryBatch,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cfg: EdgeStackConfig,
    *,
    rng: np.random.Generator | None = None,
) -> list[Edge]:
    rng = rng or np.random.default_rng(cfg.project.random_seed)
    splitter = PurgedWalkForwardSplitter.from_config(cfg)
    scenarios = list(CostScenario)
    cost_models = {s: CostModel.from_config(cfg, s) for s in scenarios}
    conservative = CostScenario.CONSERVATIVE

    available = set(features.columns)
    binnable = tuple(
        c for c in {*CONTINUOUS_RULE_FEATURES, "bench_trend_200", "bench_vol_20"}
        if c in available
    )

    # Evaluate every candidate out of sample, grouped by horizon so folds and
    # binners are computed once per horizon.
    outcomes: list[_CandidateOutcome] = []
    horizons = sorted({c.horizon for c in batch.candidates})
    for horizon in horizons:
        merged, folds = _prepare_horizon(features, labels, horizon, splitter)
        binners = [
            QuantileBinner(quantiles=cfg.discovery.quantile_bins).fit(
                merged.iloc[fold.train_idx], binnable
            )
            for fold in folds
        ]
        h_candidates = [c for c in batch.candidates if c.horizon == horizon]
        for cand in h_candidates:
            per_scenario: dict[CostScenario, list[np.ndarray]] = {s: [] for s in scenarios}
            fold_means: list[float] = []
            date_chunks: list[np.ndarray] = []
            regime_chunks: list[np.ndarray] = []
            for fold, binner in zip(folds, binners):
                test = merged.iloc[fold.test_idx]
                mask = evaluate_condition(cand.condition, test, binner)
                hits = test.loc[mask]
                gross = hits["gross_ret"].to_numpy()
                directional = gross if cand.direction is Side.LONG else -gross
                for scenario in scenarios:
                    cost = cost_models[scenario].roundtrip_cost(cand.direction, horizon)
                    per_scenario[scenario].append(directional - cost)
                net_cons = per_scenario[conservative][-1]
                if len(net_cons):
                    fold_means.append(float(net_cons.mean()))
                date_chunks.append(hits["date"].to_numpy())
                if "bench_trend_200" in hits:
                    regime_chunks.append(hits["bench_trend_200"].to_numpy())

            pooled = {s: np.concatenate(v) if v else np.array([]) for s, v in per_scenario.items()}
            net = pooled[conservative]
            dates = np.concatenate(date_chunks) if date_chunks else np.array([])
            order = np.argsort(dates) if len(dates) else np.array([], dtype=int)
            for s in scenarios:
                pooled[s] = pooled[s][order]
            dates = dates[order]

            regime_perf: dict[str, float] = {}
            if regime_chunks and len(net):
                regime = np.concatenate(regime_chunks)[order]
                up = regime > 0
                if up.any():
                    regime_perf["market_uptrend"] = float(pooled[conservative][up].mean())
                if (~up).any():
                    regime_perf["market_downtrend"] = float(pooled[conservative][~up].mean())

            # p-value placeholder; the real study happens below if n allows.
            outcomes.append(
                _CandidateOutcome(
                    candidate=cand,
                    returns_by_scenario=pooled,
                    dates=dates,
                    fold_net_means=tuple(fold_means),
                    regime_performance=regime_perf,
                    p_value=1.0,
                )
            )

    # Studies + batch-level FDR.
    studies = []
    for outcome in outcomes:
        net = outcome.returns_by_scenario[conservative]
        study = study_returns(
            net,
            rng=rng,
            n_boot=cfg.validation.bootstrap_samples,
            block_length=cfg.validation.block_length_sessions,
            sessions_per_trade=outcome.candidate.horizon,
        )
        studies.append(study)
        outcome.p_value = study.p_value if study is not None else 1.0

    pvals = np.array([o.p_value for o in outcomes])
    qvals = fdr_adjust(pvals, cfg.validation.fdr_method)

    edges: list[Edge] = []
    all_dates = pd.to_datetime(features["date"])
    discovery_start, discovery_end = all_dates.min().date(), all_dates.max().date()

    for outcome, study, q in zip(outcomes, studies, qvals):
        cand = outcome.candidate
        net = outcome.returns_by_scenario[conservative]
        reasons: list[str] = []

        if study is None:
            reasons.append(f"insufficient out-of-sample signals (n={len(net)})")
            edges.append(_build_edge(cand, None, outcome, float(q), 0.0, cfg,
                                     discovery_start, discovery_end, tuple(reasons)))
            continue

        dsr = deflated_sharpe_ratio(net, n_trials=batch.trial_count)
        survival = {
            s.value: bool(len(r) and r.mean() > 0)
            for s, r in outcome.returns_by_scenario.items()
        }

        if q > cfg.validation.fdr_alpha:
            reasons.append(f"failed FDR control (q={q:.3f} > {cfg.validation.fdr_alpha})")
        if study.effective_n < cfg.signals.min_effective_sample_size:
            reasons.append(
                f"effective sample size {study.effective_n:.0f} < "
                f"{cfg.signals.min_effective_sample_size}"
            )
        if study.mean < cfg.signals.min_expected_net_return:
            reasons.append(
                f"net mean {study.mean:.4f} below economic floor "
                f"{cfg.signals.min_expected_net_return}"
            )
        if not survival.get(CostScenario.CONSERVATIVE.value, False):
            reasons.append("does not survive conservative costs")
        n_folds_pos = sum(1 for m in outcome.fold_net_means if m > 0)
        n_folds_obs = max(1, len(outcome.fold_net_means))
        if n_folds_pos / n_folds_obs < cfg.validation.min_subperiod_consistency:
            reasons.append(
                f"positive in only {n_folds_pos}/{n_folds_obs} walk-forward folds"
            )
        if dsr < 0.5:
            reasons.append(f"deflated Sharpe probability {dsr:.2f} < 0.50 "
                           f"(searched {batch.trial_count} rules)")

        edges.append(_build_edge(cand, study, outcome, float(q), dsr, cfg,
                                 discovery_start, discovery_end, tuple(reasons)))
    return edges


def _build_edge(
    cand: CandidateEdge,
    study,
    outcome: _CandidateOutcome,
    q_value: float,
    dsr: float,
    cfg: EdgeStackConfig,
    discovery_start,
    discovery_end,
    failure_reasons: tuple[str, ...],
) -> Edge:
    from datetime import datetime

    status = EdgeStatus.VALIDATED if not failure_reasons else EdgeStatus.REJECTED
    net = outcome.returns_by_scenario[CostScenario.CONSERVATIVE]
    roundtrip = CostModel.from_config(cfg, CostScenario.CONSERVATIVE).roundtrip_cost(
        cand.direction, cand.horizon
    )
    gross_mean = float(net.mean() + roundtrip) if len(net) else 0.0

    if study is not None:
        n_pos = sum(1 for m in outcome.fold_net_means if m > 0)
        n_folds = max(1, len(outcome.fold_net_means))
        stability = n_pos / n_folds
        regime_vals = list(outcome.regime_performance.values())
        regime_stability = (
            sum(1 for v in regime_vals if v > 0) / len(regime_vals) if regime_vals else 0.0
        )
        survival = {
            s.value: bool(len(r) and r.mean() > 0)
            for s, r in outcome.returns_by_scenario.items()
        }
        cost_robustness = sum(survival.values()) / len(survival)
        half = len(net) // 2
        decay = 1.0 if half and net[half:].mean() > 0 else (0.3 if len(net) else 0.0)
        stats = EdgeStats(
            sample_size=study.n,
            effective_sample_size=study.effective_n,
            gross_mean_return=gross_mean,
            net_mean_return=study.mean,
            median_return=study.median,
            return_std=study.std,
            downside_deviation=study.downside_deviation,
            probability_of_profit=study.hit_rate,
            probability_of_positive_net_return=study.hit_rate,
            expected_shortfall=study.expected_shortfall,
            value_at_risk=study.value_at_risk,
            max_drawdown=study.max_drawdown,
            sharpe_ratio=study.sharpe,
            sortino_ratio=study.sortino,
            profit_factor=study.profit_factor,
            win_loss_ratio=study.win_loss_ratio,
            p_value=study.p_value,
            adjusted_p_value=q_value,
            q_value=q_value,
            bayesian_posterior_probability=study.posterior_prob_positive,
            bayesian_credible_interval=(study.credible_low, study.credible_high),
            bootstrap_confidence_interval=(study.ci_low, study.ci_high),
            deflated_sharpe_ratio=dsr,
        )
        robustness = EdgeRobustness(
            stability_score=stability,
            regime_stability_score=regime_stability,
            cost_robustness_score=cost_robustness,
            parameter_robustness_score=study.subperiod_consistency,
            out_of_sample_score=min(1.0, stability * min(
                1.0, study.effective_n / cfg.signals.min_effective_sample_size)),
            decay_score=decay,
            cost_scenario_survival=survival,
            regime_performance=outcome.regime_performance,
            fold_net_means=outcome.fold_net_means,
        )
    else:
        stats = EdgeStats(
            sample_size=len(net),
            effective_sample_size=float(len(net)),
            gross_mean_return=gross_mean,
            net_mean_return=float(net.mean()) if len(net) else 0.0,
            median_return=0.0, return_std=0.0, downside_deviation=0.0,
            probability_of_profit=0.0, probability_of_positive_net_return=0.0,
            expected_shortfall=0.0, value_at_risk=0.0, max_drawdown=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0,
            p_value=1.0, adjusted_p_value=q_value, q_value=q_value,
            bayesian_posterior_probability=0.0,
            bayesian_credible_interval=(0.0, 0.0),
            bootstrap_confidence_interval=(0.0, 0.0),
            deflated_sharpe_ratio=0.0,
        )
        robustness = EdgeRobustness(
            stability_score=0.0, regime_stability_score=0.0, cost_robustness_score=0.0,
            parameter_robustness_score=0.0, out_of_sample_score=0.0, decay_score=0.0,
            fold_net_means=outcome.fold_net_means,
        )

    identity = EdgeIdentity(
        edge_id=cand.candidate_id,
        name=cand.name,
        description=cand.condition.describe(),
        direction=cand.direction,
        condition=cand.condition,
        feature_dependencies=condition_features(cand.condition),
        family=cand.family,
        holding_horizon=cand.horizon,
    )
    lifecycle = EdgeLifecycle(
        status=status,
        discovery_start=discovery_start,
        discovery_end=discovery_end,
        validation_start=discovery_start,
        validation_end=discovery_end,
        last_validated_at=datetime.now(UTC),
        failure_reasons=failure_reasons,
        discovery_batch_id=cand.discovery_batch_id,
        experiment_id=cand.experiment_id,
    )
    return Edge(identity=identity, stats=stats, robustness=robustness, lifecycle=lifecycle)
