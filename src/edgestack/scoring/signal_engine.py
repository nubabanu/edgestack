"""Signal generation: validated edges + calibrated models -> ranked candidates.

For one analysis date (a session close), per symbol and side:

1. match every VALIDATED/ACTIVE edge whose condition holds on the symbol's
   feature row (quantile thresholds fitted on all history up to the date);
2. aggregate matched edges with family capping -> expected net return,
   reliability inputs, recommended horizon;
3. calibrated model probability for that (horizon, side) when a model exists,
   with disagreement vs. the edge posterior feeding the conviction penalty;
4. conviction score, ATR stops/targets/entry zone, regime context;
5. gates -> SignalCandidate or Abstention (with every failed reason).

Ranking is by conviction (which embodies conservative expected utility),
tie-broken by expected net return.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.calendar import TradingCalendar
from edgestack.data.universe import static_universe
from edgestack.discovery.candidate_generation import CONTINUOUS_RULE_FEATURES
from edgestack.discovery.conditions import evaluate_condition
from edgestack.exceptions import DataError
from edgestack.features.binning import QuantileBinner
from edgestack.models.base import TrainedModel
from edgestack.regimes.deterministic import edge_regime_similarity, label_row, regime_context
from edgestack.reporting.explanations import explain_candidate
from edgestack.risk.sizing import build_entry_plan, build_risk_plan
from edgestack.scoring.abstention import gate_reasons
from edgestack.scoring.conviction import ConvictionInputs, conviction
from edgestack.scoring.ranking import AggregatedEvidence, aggregate_evidence
from edgestack.types import (
    Abstention,
    CandidateStatus,
    Edge,
    EdgeStatus,
    Side,
    SignalCandidate,
    SignalReport,
)

BORROW_WARNING = (
    "No short-borrow data source is configured: short candidates are research "
    "output only (SHORT_RESEARCH_CANDIDATE), not executable recommendations."
)


def generate_signal_report(
    cfg: EdgeStackConfig,
    features: pd.DataFrame,
    panel: pd.DataFrame,
    edges: list[Edge],
    models: list[TrainedModel],
    as_of: date | None = None,
) -> SignalReport:
    calendar = TradingCalendar(cfg.data.calendar)
    feature_dates = pd.to_datetime(features["date"])
    as_of_ts = pd.Timestamp(as_of) if as_of else feature_dates.max()
    if as_of_ts not in set(feature_dates):
        raise DataError(
            f"{as_of_ts.date()} is not a feature date; latest is {feature_dates.max().date()}"
        )

    history = features.loc[feature_dates <= as_of_ts]
    today = history.loc[pd.to_datetime(history["date"]) == as_of_ts].set_index("symbol")

    binnable = tuple(
        c
        for c in (*CONTINUOUS_RULE_FEATURES, "bench_trend_200", "bench_vol_20")
        if c in features.columns
    )
    binner = QuantileBinner(quantiles=cfg.discovery.quantile_bins).fit(history, binnable)

    active_edges = [
        e for e in edges if e.lifecycle.status in (EdgeStatus.VALIDATED, EdgeStatus.ACTIVE)
    ]
    model_lookup = {(m.horizon, m.side): m for m in models}

    dollar_volume = _median_dollar_volume(panel, as_of_ts)
    closes = _latest_closes(panel, as_of_ts)

    longs: list[SignalCandidate] = []
    shorts: list[SignalCandidate] = []
    abstentions: list[Abstention] = []
    universe = static_universe(cfg, as_of_ts.date())

    for symbol in sorted(today.index):
        row = today.loc[symbol]
        matched = [
            e
            for e in active_edges
            if bool(evaluate_condition(e.identity.condition, today.loc[[symbol]], binner).iloc[0])
        ]
        for side in (Side.LONG, Side.SHORT):
            side_edges = [e for e in matched if e.identity.direction is side]
            outcome = _assess(
                cfg,
                calendar,
                symbol,
                row,
                side,
                side_edges,
                model_lookup,
                closes.get(symbol, float("nan")),
                dollar_volume.get(symbol, 0.0),
                as_of_ts,
                universe.limitations,
            )
            if isinstance(outcome, SignalCandidate):
                (longs if side is Side.LONG else shorts).append(outcome)
            else:
                abstentions.append(outcome)

    longs.sort(key=lambda c: (-c.conviction_score, -c.expected_net_return))
    shorts.sort(key=lambda c: (-c.conviction_score, -c.expected_net_return))
    cap = cfg.signals.max_candidates_per_side

    return SignalReport(
        as_of_date=as_of_ts.date(),
        generated_at=datetime.now(UTC),
        config_hash=cfg.config_hash(),
        long_candidates=tuple(longs[:cap]),
        short_candidates=tuple(shorts[:cap]),
        abstentions=tuple(abstentions),
        universe_warnings=universe.limitations,
    )


def _assess(
    cfg: EdgeStackConfig,
    calendar: TradingCalendar,
    symbol: str,
    row: pd.Series,
    side: Side,
    side_edges: list[Edge],
    model_lookup: dict[tuple[int, str], TrainedModel],
    close: float,
    median_dv: float,
    as_of_ts: pd.Timestamp,
    universe_limitations: tuple[str, ...],
) -> SignalCandidate | Abstention:
    evidence: AggregatedEvidence | None = aggregate_evidence(side_edges, cfg.scoring)
    data_quality = float(row.notna().mean())

    if evidence is None or not np.isfinite(close):
        reasons = gate_reasons(
            cfg=cfg,
            price=close if np.isfinite(close) else 0.0,
            median_dollar_volume=median_dv,
            data_quality_score=data_quality,
            n_matched_edges=0,
            calibrated_probability=None,
            expected_net_return=None,
            conviction_score=None,
            reward_to_risk=None,
            effective_sample_size=None,
        )
        return Abstention(
            as_of_date=as_of_ts.date(), symbol=symbol, side=side, reasons=tuple(reasons)
        )

    horizon = evidence.recommended_horizon
    # Per-trade probability implied by the historical edge evidence (hit rate
    # of net returns) — comparable with the model's per-trade P(net > 0).
    # The posterior-of-the-mean is a different quantity and must not be
    # compared against per-trade probabilities.
    edge_probability = float(
        np.mean([e.stats.probability_of_positive_net_return for e in side_edges])
    )
    model = model_lookup.get((horizon, side.value))
    if model is not None:
        probability = float(model.predict_probability(row.to_frame().T)[0])
        is_calibrated = model.is_calibrated
        disagreement = abs(probability - edge_probability)
    else:
        probability = edge_probability
        is_calibrated = False
        disagreement = 0.0

    atr = float(row.get("natr_14", np.nan)) * close
    if not np.isfinite(atr) or atr <= 0:
        atr = 0.02 * close  # volatility unknown: assume a wide 2% ATR

    risk_plan = build_risk_plan(side, close, atr, cfg)
    entry_plan = build_entry_plan(
        side,
        close,
        atr,
        as_of_ts.date(),
        calendar,
        cfg.signals.execution_delay_sessions,
    )

    liquidity = float(np.clip((np.log10(max(median_dv, 1.0)) - 6.0) / 3.0, 0.0, 1.0))
    similarity = float(np.mean([edge_regime_similarity(e, label_row(row)) for e in side_edges]))
    per_trade_vol = float(row.get("realized_vol_20", np.nan))
    per_trade_vol = (
        per_trade_vol / np.sqrt(252) * np.sqrt(horizon)
        if np.isfinite(per_trade_vol)
        else 0.02 * np.sqrt(horizon)
    )

    result = conviction(
        ConvictionInputs(
            calibrated_probability=probability,
            expected_net_return=evidence.expected_net_return,
            expected_risk=per_trade_vol,
            effective_sample_size=evidence.effective_n,
            stability_score=evidence.stability,
            out_of_sample_score=evidence.out_of_sample,
            deflated_sharpe=evidence.deflated_sharpe,
            regime_similarity=similarity,
            liquidity_score=liquidity,
            data_quality_score=data_quality,
            cost_survival_fraction=evidence.cost_survival,
            ci_width=evidence.ci_width,
            tail_risk=evidence.tail_risk,
            model_edge_disagreement=disagreement,
        ),
        shrinkage_min_sample=cfg.scoring.shrinkage_min_sample,
        logistic_slope=cfg.scoring.logistic_slope,
    )

    reasons = gate_reasons(
        cfg=cfg,
        price=close,
        median_dollar_volume=median_dv,
        data_quality_score=data_quality,
        n_matched_edges=len(side_edges),
        calibrated_probability=probability,
        expected_net_return=evidence.expected_net_return,
        conviction_score=result.score,
        reward_to_risk=risk_plan.reward_to_risk,
        effective_sample_size=evidence.effective_n,
    )
    if reasons:
        return Abstention(
            as_of_date=as_of_ts.date(), symbol=symbol, side=side, reasons=tuple(reasons)
        )

    warnings = list(universe_limitations)
    if side is Side.SHORT:
        warnings.append(BORROW_WARNING)
    if not is_calibrated:
        warnings.append(
            "Probability is a shrinkage posterior, not an independently calibrated model output."
        )
    if disagreement > 0.15:
        warnings.append(
            f"Model and edge evidence disagree ({disagreement:.0%}); conviction "
            "was penalized accordingly."
        )

    status = (
        CandidateStatus.SHORT_RESEARCH_CANDIDATE
        if side is Side.SHORT
        else CandidateStatus.RESEARCH_CANDIDATE
    )
    cost = evidence.expected_net_return  # net already includes conservative costs
    candidate = SignalCandidate(
        as_of_date=as_of_ts.date(),
        symbol=symbol,
        side=side,
        status=status,
        conviction_score=round(result.score, 1),
        calibrated_probability_of_positive_net_return=round(probability, 4),
        probability_is_calibrated=is_calibrated,
        expected_gross_return=round(cost + _roundtrip(cfg, side, horizon), 4),
        expected_net_return=round(cost, 4),
        expected_volatility=round(per_trade_vol, 4),
        expected_shortfall_95=round(evidence.tail_risk, 4),
        recommended_holding_sessions=horizon,
        entry=entry_plan,
        risk=risk_plan,
        regime=regime_context(row, similarity),
        evidence=evidence.items,
        warnings=tuple(warnings),
        model_version=(f"{model.name}_h{horizon}" if model else "edge_posterior"),
    )
    return candidate.model_copy(update={"explanation": explain_candidate(candidate)})


def _roundtrip(cfg: EdgeStackConfig, side: Side, horizon: int) -> float:
    from edgestack.execution.costs import CostModel

    return CostModel.from_config(cfg).roundtrip_cost(side, horizon)


def _median_dollar_volume(
    panel: pd.DataFrame, as_of: pd.Timestamp, window: int = 60
) -> dict[str, float]:
    recent = panel.loc[pd.to_datetime(panel["date"]) <= as_of]
    out: dict[str, float] = {}
    for symbol, group in recent.groupby("symbol"):
        tail = group.sort_values("date").tail(window)
        out[str(symbol)] = float((tail["close"] * tail["volume"]).median())
    return out


def _latest_closes(panel: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    on_date = panel.loc[pd.to_datetime(panel["date"]) == as_of]
    return dict(zip(on_date["symbol"].astype(str), on_date["close"].astype(float)))
