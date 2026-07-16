"""Instrument timing is informative when unpromoted and actionable only when frozen."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from edgestack.exceptions import DataError
from edgestack.recommendation.instrument import analyze_instrument, resolve_instrument
from edgestack.recommendation.instrument_schemas import (
    DirectionalRating,
    FrozenTimingArtifactV2,
    InstrumentAnalysisStatus,
    InstrumentKind,
    NewsEvidenceV2,
    TimingHorizon,
)
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

SESSION = date(2026, 7, 16)


def _bundle() -> CanonicalRecommendationBundleV2:
    policy = load_baseline_policy()
    as_of = datetime(2026, 7, 16, 20, tzinfo=UTC)
    execution = datetime(2026, 7, 17, 13, 30, tzinfo=UTC)
    freshness = FreshnessV2(
        as_of=as_of,
        expected_session=SESSION,
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
        session=SESSION,
        stressed_forecast_volatility=0.10,
        parametric_995_one_day_loss=0.02,
        historical_995_one_day_loss=0.025,
        bootstrapped_99_path_drawdown=0.08,
        liquidity_position_limits={item.symbol: 5.0 for item in policy.weights},
        funding_rate=0.04,
        funding_rate_as_of=SESSION,
    )
    recommendation = size_recommendation(
        base=base,
        profile=profile,
        state=RiskStateV2.initial(profile.account_equity),
        inputs=inputs,
    )
    return CanonicalRecommendationBundleV2(
        generated_at=as_of,
        session=SESSION,
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


def _daily(symbol: str = "GLD", sessions: int = 900) -> pd.DataFrame:
    dates = pd.bdate_range(end=SESSION, periods=sessions)
    rng = np.random.default_rng(11)
    weekday = dates.weekday.to_numpy()
    returns = rng.normal(0.0002, 0.009, sessions) + np.where(weekday == 0, 0.0008, 0)
    returns[-60:] -= 0.002
    close = 100 * np.cumprod(1 + returns)
    open_ = close * (1 + rng.normal(0, 0.0015, sessions))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * 1.005,
            "low": np.minimum(open_, close) * 0.995,
            "close": close,
            "volume": np.full(sessions, 5_000_000.0),
            "adj_close": close,
        }
    )


def _intraday(symbol: str = "GLD") -> pd.DataFrame:
    sessions = pd.bdate_range(end=SESSION, periods=60)
    rows = []
    rng = np.random.default_rng(22)
    for session in sessions:
        # 13:30 UTC is the regular New York open during this fixture's summer dates.
        timestamps = pd.date_range(
            pd.Timestamp(session.date(), tz="UTC") + pd.Timedelta(hours=13, minutes=30),
            periods=7,
            freq="h",
        )
        price = 100.0
        for timestamp in timestamps:
            next_price = price * (1 + rng.normal(0.0001, 0.002))
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "open": price,
                    "high": max(price, next_price) * 1.001,
                    "low": min(price, next_price) * 0.999,
                    "close": next_price,
                    "volume": 100_000.0,
                }
            )
            price = next_price
    return pd.DataFrame(rows)


def _artifact(
    bundle: CanonicalRecommendationBundleV2, horizon: TimingHorizon
) -> FrozenTimingArtifactV2:
    return FrozenTimingArtifactV2(
        artifact_hash=f"promoted-{horizon.value.lower()}",
        sleeve_id=f"timing-{horizon.value.lower()}",
        symbol="GLD",
        horizon=horizon,
        label=f"Frozen {horizon.value.lower()} entry",
        entry_window="Frozen next-open rule",
        exit_window="Frozen validated exit rule",
        holding_sessions=1 if horizon is TimingHorizon.DAY else 5,
        expected_net_return=0.01,
        incremental_expected_net_return=0.005,
        lower_95=0.002,
        incremental_lower_95=0.001,
        upper_95=0.018,
        multiple_testing_adjusted_pvalue=0.01,
        observations=500,
        effective_sample_size=200,
        data_version=bundle.data_version,
        artifact_version=bundle.artifact_version,
        policy_version=bundle.policy_version,
        complete_trial_family_hash="complete-family-v2",
        stress_scenarios_passed=("conservative", "stress", "delayed_fill"),
        what_invalidates_it=("monitoring failure",),
    )


def test_commodity_alias_uses_tradable_proxy_and_discloses_tracking_risk() -> None:
    resolution = resolve_instrument(" gold ")

    assert resolution.resolved_symbol == "GLD"
    assert resolution.instrument_kind is InstrumentKind.COMMODITY_PROXY
    assert resolution.proxy_for == "physical gold"
    assert "tracking error" in " ".join(resolution.notes)


def test_observational_windows_never_create_rating_or_claim_an_hour() -> None:
    bundle = _bundle()
    result = analyze_instrument(
        bundle=bundle,
        resolution=resolve_instrument("gold", canonical=bundle),
        daily_bars=_daily(),
        intended_entry_at=datetime(2026, 7, 20, 13, 30, tzinfo=UTC),
    )

    assert result.status is InstrumentAnalysisStatus.INSUFFICIENT_EVIDENCE
    assert result.overall_rating is DirectionalRating.NOT_RATED
    assert result.overall_score is None
    assert not result.alignment.aligned_trade
    day = next(item for item in result.horizon_analyses if item.horizon is TimingHorizon.DAY)
    assert day.best_window is None
    assert "No best or worst hour" in (day.warning or "")
    for analysis in result.horizon_analyses[1:]:
        assert analysis.best_window is not None
        assert analysis.worst_window is not None
        assert not analysis.actionable
        assert analysis.searched_variants > 1
        assert analysis.best_window.multiple_testing_adjusted_pvalue is not None
    assert all(not effect.promoted for effect in result.tailwinds + result.headwinds)
    assert all(effect.net_contribution == 0 for effect in result.tailwinds + result.headwinds)
    assert any(effect.adverse_counter_effect for effect in result.mixed_effects)
    assert any(effect.protective_counter_effect for effect in result.headwinds)


def test_all_stars_requires_promoted_compatible_artifact_for_every_horizon() -> None:
    bundle = _bundle()
    artifacts = tuple(_artifact(bundle, horizon) for horizon in TimingHorizon)
    result = analyze_instrument(
        bundle=bundle,
        resolution=resolve_instrument("GLD", canonical=bundle),
        daily_bars=_daily(),
        intraday_bars=_intraday(),
        timing_artifacts=artifacts,
    )

    assert result.status is InstrumentAnalysisStatus.ACTIONABLE
    assert result.overall_rating is DirectionalRating.POSITIVE
    assert result.alignment.aligned_trade
    assert set(result.alignment.actionable_horizons) == set(TimingHorizon)
    assert all(item.actionable for item in result.horizon_analyses)
    assert all(
        item.best_window and item.best_window.artifact_hash for item in result.horizon_analyses
    )


def test_timing_artifact_version_mismatch_fails_closed() -> None:
    bundle = _bundle()
    artifact = _artifact(bundle, TimingHorizon.WEEK).model_copy(
        update={"data_version": "other-data"}
    )

    with pytest.raises(DataError, match="version mismatch"):
        analyze_instrument(
            bundle=bundle,
            resolution=resolve_instrument("GLD", canonical=bundle),
            daily_bars=_daily(),
            timing_artifacts=(artifact,),
        )


def test_compound_timing_artifact_requires_frozen_component_ablations() -> None:
    bundle = _bundle()
    payload = _artifact(bundle, TimingHorizon.WEEK).model_dump()
    payload.update({"compound": True, "component_artifact_hashes": ()})

    with pytest.raises(ValueError, match="requires frozen components"):
        FrozenTimingArtifactV2.model_validate(payload)


def test_frozen_news_is_context_only_and_future_news_is_excluded() -> None:
    bundle = _bundle()
    current = NewsEvidenceV2(
        news_id="current",
        symbol="GLD",
        headline="Gold context headline",
        source="fixture",
        published_at=datetime(2026, 7, 16, 18, tzinfo=UTC),
        sentiment_label="POSITIVE",
        relevance=0.9,
        is_fresh=True,
        age_hours=2,
    )
    future = current.model_copy(
        update={
            "news_id": "future",
            "published_at": datetime(2026, 7, 17, 18, tzinfo=UTC),
        }
    )

    result = analyze_instrument(
        bundle=bundle,
        resolution=resolve_instrument("GLD", canonical=bundle),
        daily_bars=_daily(),
        news=(current, future),
    )

    assert [item.news_id for item in result.news] == ["current"]
    assert result.news[0].actionable_contribution == 0
    assert result.overall_rating is DirectionalRating.NOT_RATED
