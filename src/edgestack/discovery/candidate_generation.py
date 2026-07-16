"""Candidate-edge generation.

Enumerates interpretable rule conditions — event singles over binary
features, quantile predicates over continuous features, and bounded depth-2
conjunctions with market-context predicates — across the configured discovery
horizons and both directions.

Honesty contract: every enumerated (condition, side, horizon) tuple becomes a
persisted candidate. The batch's trial count is the FULL enumeration size and
feeds FDR and the deflated Sharpe ratio; there is no way to quietly evaluate
a rule without it counting as a trial.

In-sample screening statistics are computed on fold 0's TRAIN window only
(quantiles fitted there too) — they never touch any walk-forward test window.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.discovery.conditions import evaluate_condition
from edgestack.execution.costs import CostModel
from edgestack.features.binning import QuantileBinner
from edgestack.features.registry import get_spec
from edgestack.types import AllOf, CandidateEdge, Condition, Family, Predicate, Side
from edgestack.validation.splits import PurgedWalkForwardSplitter

# Continuous features searched with outer-quantile predicates.
CONTINUOUS_RULE_FEATURES: tuple[str, ...] = (
    "rsi_14_pctile",
    "bb_pctb_20",
    "stoch_k_14",
    "mom_60",
    "mom_12_1",
    "ret_5d",
    "dist_from_high_252",
    "vol_pctile_252",
    "vol_ratio_5_60",
    "rel_volume_20",
    "volume_z_60",
    "close_vs_sma20",
    "close_vs_sma200",
    "trend_slope_60",
    "di_diff_14",
    "rel_strength_60",
    "cs_rank_mom_60",
    "overnight_gap",
    "donchian_pos_55",
)

# Binary event features searched as `feature == 1`.
BINARY_RULE_FEATURES: tuple[str, ...] = (
    "cal_is_friday",
    "cal_is_monday",
    "cal_pre_holiday",
    "cal_post_holiday",
    "cal_turn_of_month",
    "cal_opex_week",
    "cal_year_end",
    "cal_santa_window",
    "breakout_20",
    "breakdown_20",
    "false_breakout_20",
    "squeeze_on",
    "candle_hammer",
    "candle_bull_engulf",
    "candle_bear_engulf",
    "candle_inside_bar",
    "candle_doji",
    "candle_nr7",
    "above_sma200",
    "golden_state",
    "macd_state",
    "pivot_high_2",
    "pivot_low_2",
)

# Depth-2 context predicates conjoined with promising singles.
CONTEXT_PREDICATES: tuple[Predicate, ...] = (
    Predicate(feature="above_sma200", op="==", value=1.0),
    Predicate(feature="above_sma200", op="==", value=0.0),
    Predicate(feature="bench_trend_200", op=">", quantile=0.5),
    Predicate(feature="bench_vol_20", op="<", quantile=0.5),
    Predicate(feature="rel_volume_20", op=">", quantile=0.75),
)


@dataclass(frozen=True)
class DiscoveryBatch:
    batch_id: str
    experiment_id: str
    candidates: tuple[CandidateEdge, ...]

    @property
    def trial_count(self) -> int:
        return len(self.candidates)


def _condition_family(cond: Condition) -> Family:
    from edgestack.types import condition_features

    return get_spec(condition_features(cond)[0]).family


def _condition_id(cond: Condition, side: Side, horizon: int) -> str:
    if isinstance(cond, Predicate | AllOf):
        payload = cond.model_dump_json()
    else:  # pragma: no cover - AnyOf not generated today
        payload = cond.model_dump_json()
    raw = f"{payload}|{side.value}|{horizon}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _describe(cond: Condition, side: Side, horizon: int) -> str:
    return f"{side.value.lower()}_h{horizon}: {cond.describe()}"


def enumerate_conditions(cfg: EdgeStackConfig, available_columns: set[str]) -> list[Condition]:
    """All rule conditions in scope for a discovery batch (deterministic order)."""
    lo_q, hi_q = cfg.discovery.quantile_bins[0], cfg.discovery.quantile_bins[-1]
    singles: list[Condition] = []
    for name in CONTINUOUS_RULE_FEATURES:
        if name not in available_columns:
            continue
        singles.append(Predicate(feature=name, op="<", quantile=lo_q))
        singles.append(Predicate(feature=name, op=">", quantile=hi_q))
    for name in BINARY_RULE_FEATURES:
        if name not in available_columns:
            continue
        singles.append(Predicate(feature=name, op="==", value=1.0))

    conditions: list[Condition] = list(singles)
    if cfg.discovery.max_condition_depth >= 2:
        for single in singles:
            for context in CONTEXT_PREDICATES:
                if context.feature not in available_columns:
                    continue
                if isinstance(single, Predicate) and single.feature == context.feature:
                    continue
                conditions.append(AllOf(conditions=(single, context)))
    return conditions


def generate_candidates(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cfg: EdgeStackConfig,
    experiment_id: str,
    *,
    conditions: list[Condition] | None = None,
    rng: np.random.Generator | None = None,
) -> DiscoveryBatch:
    """Enumerate and in-sample-screen candidate edges.

    ``conditions`` overrides the default enumeration (used by tests and
    focused research runs). Screening stats come from fold 0's train window.
    """
    rng = rng or np.random.default_rng(cfg.project.random_seed)
    batch_id = uuid.uuid4().hex[:12]
    available = set(features.columns)
    conds = conditions if conditions is not None else enumerate_conditions(cfg, available)

    splitter = PurgedWalkForwardSplitter.from_config(cfg)
    cost_model = CostModel.from_config(cfg)
    candidates: list[CandidateEdge] = []

    for horizon in cfg.discovery.horizons:
        h_labels = labels.loc[labels["horizon"] == horizon]
        merged = features.merge(
            h_labels[["symbol", "date", "label_end", "gross_ret"]],
            on=["symbol", "date"],
            how="inner",
        )
        folds = splitter.split_frame(merged["date"], merged["label_end"])
        train = merged.iloc[folds[0].train_idx]

        continuous = tuple(
            c
            for c in (*CONTINUOUS_RULE_FEATURES, "bench_trend_200", "bench_vol_20")
            if c in available
        )
        binner = QuantileBinner(quantiles=cfg.discovery.quantile_bins).fit(train, continuous)

        for cond in conds:
            mask = evaluate_condition(cond, train, binner)
            gross = train.loc[mask, "gross_ret"].to_numpy()
            for side in (Side.LONG, Side.SHORT):
                if len(candidates) >= cfg.discovery.max_rules_per_batch:
                    break
                n = len(gross)
                if n >= cfg.discovery.min_support:
                    directional = gross if side is Side.LONG else -gross
                    net = directional - cost_model.roundtrip_cost(side, horizon)
                    mean_net = float(net.mean())
                    hit = float((net > 0).mean())
                    sd = float(net.std(ddof=1)) if n > 1 else 0.0
                    if sd > 0:
                        from scipy import stats as sps

                        t = mean_net / (sd / np.sqrt(n))
                        p = float(1.0 - sps.t.cdf(t, df=n - 1))
                    else:
                        p = 1.0
                    gross_mean = float(directional.mean())
                else:
                    mean_net, hit, p, gross_mean = 0.0, 0.0, 1.0, 0.0
                candidates.append(
                    CandidateEdge(
                        candidate_id=_condition_id(cond, side, horizon),
                        discovery_batch_id=batch_id,
                        experiment_id=experiment_id,
                        name=_describe(cond, side, horizon),
                        direction=side,
                        condition=cond,
                        family=_condition_family(cond),
                        horizon=horizon,
                        in_sample_n=int(n),
                        in_sample_gross_mean=gross_mean,
                        in_sample_net_mean=mean_net,
                        in_sample_hit_rate=hit,
                        in_sample_p_value=p,
                    )
                )

    return DiscoveryBatch(
        batch_id=batch_id,
        experiment_id=experiment_id,
        candidates=tuple(candidates),
    )
