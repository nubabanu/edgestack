"""Staged sniper policy, loss sizing, veto, and exclusion invariants."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from edgestack.data.calendar import TradingCalendar
from edgestack.recommendation.policy import load_baseline_policy
from edgestack.recommendation.risk import RiskInputsV2, size_recommendation
from edgestack.recommendation.schemas import (
    BaseRecommendationV2,
    CanonicalRecommendationBundleV2,
    FreshnessV2,
    RecommendationStatus,
    RiskProfileV2,
    RiskStateV2,
)
from edgestack.recommendation.sniper import build_sniper_plan
from edgestack.recommendation.sniper_schemas import (
    OverlayState,
    SniperActivation,
    SniperCandidateStatus,
    SniperStrategyId,
)


def _bundle(session: date) -> CanonicalRecommendationBundleV2:
    policy = load_baseline_policy()
    as_of = datetime.combine(session, datetime.min.time(), tzinfo=UTC) + timedelta(hours=20)
    execution = as_of + timedelta(days=1)
    freshness = FreshnessV2(
        as_of=as_of,
        expected_session=session,
        is_fresh=True,
        age_business_days=0,
        complete=True,
        compatible=True,
    )
    base = BaseRecommendationV2(
        status=RecommendationStatus.BASELINE_ONLY,
        as_of=as_of,
        execution_at=execution,
        artifact_version="artifact-v2",
        data_version="data-v2",
        policy_version=policy.policy_version,
        baseline_weights=policy.weights,
        unlevered_base_weights=policy.weights,
        expected_volatility=0.08,
        covariance_version="cov-v2",
        freshness=freshness,
    )
    profile = RiskProfileV2()
    inputs = RiskInputsV2(
        session=session,
        stressed_forecast_volatility=0.10,
        parametric_995_one_day_loss=0.02,
        historical_995_one_day_loss=0.025,
        bootstrapped_99_path_drawdown=0.08,
        liquidity_position_limits={item.symbol: 5 for item in policy.weights},
        funding_rate=0.04,
        funding_rate_as_of=session,
    )
    recommendation = size_recommendation(
        base=base,
        profile=profile,
        state=RiskStateV2.initial(profile.account_equity),
        inputs=inputs,
    )
    return CanonicalRecommendationBundleV2(
        generated_at=as_of,
        session=session,
        as_of=as_of,
        execution_at=execution,
        data_version=base.data_version,
        artifact_version=base.artifact_version,
        policy_version=base.policy_version,
        baseline_policy=policy,
        default_risk_profile=profile,
        base_recommendation=base,
        default_recommendation=recommendation,
    )


def _spy_panel(session: date, *, volatile: bool = False, below_200: bool = False) -> pd.DataFrame:
    dates = pd.bdate_range(end=session, periods=1_000)
    rng = np.random.default_rng(83)
    returns = rng.normal(0.00035, 0.004, len(dates))
    # Recurrent resolved dip samples for descriptive p5 sizing.
    for index in range(240, len(returns) - 10, 20):
        returns[index : index + 3] = [-0.003, -0.004, -0.005]
        returns[index + 3 : index + 6] = [0.005, 0.004, 0.003]
    returns[-3:] = [-0.006, -0.007, -0.008]
    if volatile:
        returns[-20:] = np.array([0.025, -0.024] * 10)
        returns[-3:] = [-0.025, -0.026, -0.027]
    close = 100 * np.cumprod(1 + returns)
    if below_200:
        close[-3:] *= 0.65
    open_ = close * (1 + rng.normal(0, 0.001, len(close)))
    frame = pd.DataFrame(
        {
            "symbol": "SPY",
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * 1.003,
            "low": np.minimum(open_, close) * 0.997,
            "close": close,
            "volume": 10_000_000.0,
            "adj_close": close,
        }
    )
    return frame


def _plan(
    session: date,
    *,
    volatile: bool = False,
    below_200: bool = False,
    vix_contango: float | None = None,
):
    return build_sniper_plan(
        bundle=_bundle(session),
        panel=_spy_panel(session, volatile=volatile, below_200=below_200),
        calendar=TradingCalendar(),
        account_equity=100_000,
        max_tolerable_loss=500,
        vix_contango=vix_contango,
    )


def test_c1_c2_primary_triggers_once_and_sizes_from_adverse_tail() -> None:
    plan = _plan(date(2026, 7, 16))
    primary = plan.stage_1_candidates[0]

    assert primary.strategy_id is SniperStrategyId.C1_C2_PRIMARY
    assert SniperStrategyId.C2_THREE_DOWN in primary.component_triggers
    assert primary.status is SniperCandidateStatus.TRIGGERED_SHADOW
    assert primary.entry_window and "Next regular-session open" in primary.entry_window
    assert primary.maximum_holding_sessions == 4
    assert primary.sizing is not None
    assert primary.sizing.capped_notional <= primary.sizing.account_equity
    assert (
        primary.sizing.capped_notional * abs(primary.sizing.adverse_move_p05)
        <= primary.sizing.max_tolerable_loss + 1e-8
    )
    assert primary.evidence.observations >= 30
    assert not primary.actionable and primary.paper_only


def test_primary_engine_honors_september_trend_vol_and_vix_vetoes() -> None:
    september = _plan(date(2026, 9, 18)).stage_1_candidates[0]
    trend = _plan(date(2026, 7, 16), below_200=True).stage_1_candidates[0]
    volatility = _plan(date(2026, 7, 16), volatile=True).stage_1_candidates[0]
    backwardation = _plan(date(2026, 7, 16), vix_contango=-0.05).stage_1_candidates[0]

    assert september.status is SniperCandidateStatus.VETOED
    assert "September stand-aside" in september.veto_reasons
    assert any("200-DMA" in reason for reason in trend.veto_reasons)
    assert any("volatility above 20%" in reason for reason in volatility.veto_reasons)
    assert "VIX term structure backwardation veto" in backwardation.veto_reasons


def test_santa_is_fixed_scheduled_shadow_trade_with_defined_exit() -> None:
    plan = _plan(date(2026, 7, 16))
    santa = plan.stage_1_candidates[1]

    assert santa.strategy_id is SniperStrategyId.A1_SANTA
    assert santa.status is SniperCandidateStatus.SCHEDULED
    assert santa.entry_window == "Planned close of 2026-12-24"
    assert "2027-01-05" in santa.exit_rule
    assert not santa.actionable


def test_stage_roles_overlays_and_hard_exclusions_cannot_bypass_policy() -> None:
    plan = _plan(date(2026, 7, 16))
    ranking = {item.strategy_id: item for item in plan.policy_ranking}

    assert ranking[SniperStrategyId.C1_C2_PRIMARY].rank == 1
    assert ranking[SniperStrategyId.A2_PRE_FOMC].activation is SniperActivation.BLOCKED_DATA
    assert ranking[SniperStrategyId.C3_VIX_CONTANGO].activation is SniperActivation.FILTER_ONLY
    assert all(not overlay.can_initiate for overlay in plan.overlays)
    assert (
        next(
            item for item in plan.overlays if item.strategy_id is SniperStrategyId.C3_VIX_CONTANGO
        ).state
        is OverlayState.UNAVAILABLE
    )
    assert all(
        candidate.status is SniperCandidateStatus.BLOCKED for candidate in plan.stage_2_candidates
    )
    assert set(plan.excluded_strategy_ids) == {
        SniperStrategyId.A3_MINOR_HOLIDAY,
        SniperStrategyId.A4_SMALL_CAP_JANUARY,
        SniperStrategyId.A5_WEEKEND_MONDAY,
        SniperStrategyId.B_NAIVE_OVERNIGHT,
        SniperStrategyId.D_SHORT_VOL_INCOME,
    }
    assert not plan.stage_1_promotion_satisfied
