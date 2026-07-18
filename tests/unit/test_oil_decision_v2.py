"""Paper-only eToro OIL decision gates and Monday regression fixtures."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from edgestack.recommendation.instrument_schemas import NewsEvidenceV2
from edgestack.recommendation.oil import build_oil_decision, persist_oil_snapshot
from edgestack.recommendation.oil_schemas import (
    OilBrokerQuoteV2,
    OilDecisionRequestV2,
    OilDecisionStatus,
    OilEventFlag,
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


def _daily(symbol: str, *, scale: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range(end=SESSION, periods=900)
    rng = np.random.default_rng(sum(map(ord, symbol)))
    returns = rng.normal(0.00015, 0.008, len(dates))
    close = scale * np.cumprod(1 + returns)
    close *= scale / close[-1]
    open_ = close * (1 + rng.normal(0, 0.001, len(dates)))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * 1.005,
            "low": np.minimum(open_, close) * 0.995,
            "close": close,
            "volume": np.full(len(dates), 5_000_000.0),
            "adj_close": close,
        }
    )


def _monday_intraday() -> pd.DataFrame:
    sessions = pd.bdate_range(end=SESSION, periods=39)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260720)
    late_shocks = rng.normal(0.0, 0.010, len(sessions))
    late_shocks -= late_shocks.mean()
    for session, late_shock in zip(sessions, late_shocks, strict=True):
        timestamps = pd.date_range(
            pd.Timestamp(session.date(), tz="UTC") + pd.Timedelta(hours=13, minutes=30),
            periods=26,
            freq="15min",
        )
        returns = rng.normal(0.0, 0.0004, len(timestamps))
        returns[:4] = -0.00015
        returns[17:21] = (0.0028 + late_shock) / 4.0
        price = 100.0
        for timestamp, value in zip(timestamps, returns, strict=True):
            next_price = price * (1 + value)
            rows.append(
                {
                    "symbol": "USO",
                    "timestamp": timestamp,
                    "interval_minutes": 15,
                    "open": price,
                    "high": max(price, next_price) * 1.001,
                    "low": min(price, next_price) * 0.999,
                    "close": next_price,
                    "volume": 50_000.0,
                }
            )
            price = next_price
    return pd.DataFrame(rows)


def _request(hour: int, minute: int, *, stale: bool = False) -> OilDecisionRequestV2:
    entry = datetime(2026, 7, 20, hour, minute, tzinfo=UTC)
    observed = entry - pd.Timedelta(minutes=10 if stale else 1)
    return OilDecisionRequestV2(
        intended_entry_at=entry,
        quote=OilBrokerQuoteV2(
            observed_at=observed,
            bid=99.98,
            ask=100.02,
            offered_leverage=10,
        ),
        modeled_leverage=10,
    )


def _build(request: OilDecisionRequestV2, *, catalog_version: str = "data-v2"):
    daily = {symbol: _daily(symbol) for symbol in ("CL=F", "USO", "BNO", "XLE", "UUP", "^OVX")}
    fifteen = {"USO": _monday_intraday()}
    return build_oil_decision(
        bundle=_bundle(),
        request=request,
        evaluated_at=request.quote.observed_at + pd.Timedelta(minutes=1),
        catalog_data_version=catalog_version,
        daily=daily,
        hourly={},
        fifteen_minute=fifteen,
    )


def test_quote_and_leverage_validation_fail_closed() -> None:
    with pytest.raises(ValidationError, match="ask must be greater"):
        OilBrokerQuoteV2(
            observed_at=datetime(2026, 7, 20, 13, 29, tzinfo=UTC),
            bid=100,
            ask=99,
        )
    with pytest.raises(ValidationError, match="cannot exceed"):
        OilDecisionRequestV2(
            intended_entry_at=datetime(2026, 7, 20, 13, 30, tzinfo=UTC),
            quote=OilBrokerQuoteV2(
                observed_at=datetime(2026, 7, 20, 13, 29, tzinfo=UTC),
                bid=99.9,
                ask=100.1,
                offered_leverage=5,
            ),
            modeled_leverage=10,
        )


def test_monday_open_is_weak_and_1345_et_remains_insufficient() -> None:
    opening = _build(_request(13, 30))
    open_base = next(item for item in opening.friction_sensitivity if item.name == "BASE")
    assert opening.status is OilDecisionStatus.OBSERVE
    assert open_base.matched_slot == "09:30"
    assert open_base.observations == 39
    assert open_base.expected_net_return is not None and open_base.expected_net_return < 0
    assert not open_base.survives
    assert [item.round_trip_cost_bps for item in opening.friction_sensitivity] == [10, 25, 50]

    later = _build(_request(17, 45))
    later_base = next(item for item in later.friction_sensitivity if item.name == "BASE")
    assert later_base.matched_slot == "13:45"
    assert later_base.expected_net_return is not None and later_base.expected_net_return > 0
    assert later_base.lower_95 is not None and later_base.lower_95 < 0
    assert later_base.multiple_testing_adjusted_pvalue is not None
    assert later_base.multiple_testing_adjusted_pvalue > 0.05
    assert not later_base.survives
    assert later.status is OilDecisionStatus.OBSERVE


def test_version_staleness_manual_events_and_quote_age_are_hard_vetoes() -> None:
    mismatch = _build(_request(13, 30), catalog_version="changed")
    assert mismatch.status is OilDecisionStatus.BLOCKED
    assert any("Catalog data changed" in reason for reason in mismatch.hard_block_reasons)

    flagged_request = _request(13, 30).model_copy(
        update={"event_flags": (OilEventFlag.SHIPPING_DISRUPTION,)}
    )
    flagged = _build(flagged_request)
    assert flagged.status is OilDecisionStatus.BLOCKED
    assert any("shipping" in reason.lower() for reason in flagged.hard_block_reasons)

    stale_request = _request(13, 30, stale=True)
    stale = build_oil_decision(
        bundle=_bundle(),
        request=stale_request,
        evaluated_at=stale_request.intended_entry_at,
        catalog_data_version="data-v2",
        daily={symbol: _daily(symbol) for symbol in ("CL=F", "USO")},
        hourly={},
        fifteen_minute={"USO": _monday_intraday()},
    )
    assert stale.status is OilDecisionStatus.BLOCKED
    assert any("older than five" in reason for reason in stale.hard_block_reasons)


def test_eia_timezone_rollover_missing_primary_and_news_context_fail_closed() -> None:
    eia_entry = datetime.fromisoformat("2026-07-22T16:30:00+02:00")
    eia_request = OilDecisionRequestV2(
        intended_entry_at=eia_entry,
        quote=OilBrokerQuoteV2(
            observed_at=eia_entry - pd.Timedelta(minutes=1),
            bid=99.98,
            ask=100.02,
        ),
    )
    daily = {symbol: _daily(symbol) for symbol in ("CL=F", "USO")}
    eia = build_oil_decision(
        bundle=_bundle(),
        request=eia_request,
        evaluated_at=eia_entry,
        catalog_data_version="data-v2",
        daily=daily,
        hourly={},
        fifteen_minute={"USO": _monday_intraday()},
    )
    assert eia.status is OilDecisionStatus.BLOCKED
    assert next(item for item in eia.event_vetoes if item.code == "EIA_RELEASE_WINDOW").active

    rollover_request = _request(13, 30).model_copy(
        update={
            "event_flags": (
                OilEventFlag.WEEKEND_SUPPLY_ESCALATION,
                OilEventFlag.WTI_ROLLOVER_EXPIRY,
            )
        }
    )
    rollover = _build(rollover_request)
    assert rollover.status is OilDecisionStatus.BLOCKED
    assert {item.code for item in rollover.event_vetoes if item.active}.issuperset(
        {"WEEKEND_SUPPLY_ESCALATION", "WTI_ROLLOVER_EXPIRY"}
    )

    monday_request = _request(13, 30)
    missing_primary = build_oil_decision(
        bundle=_bundle(),
        request=monday_request,
        evaluated_at=monday_request.intended_entry_at,
        catalog_data_version="data-v2",
        daily={"USO": _daily("USO")},
        hourly={},
        fifteen_minute={"USO": _monday_intraday()},
    )
    assert missing_primary.status is OilDecisionStatus.BLOCKED
    assert not missing_primary.data_freshness.all_required_sources_present

    contextual_news = NewsEvidenceV2(
        news_id="oil-context",
        symbol="USO",
        headline="Supply headline fixture",
        source="fixture",
        published_at=datetime(2026, 7, 16, 19, tzinfo=UTC),
        sentiment_label="POSITIVE",
        is_fresh=True,
        age_hours=1,
    )
    with_news = build_oil_decision(
        bundle=_bundle(),
        request=monday_request,
        evaluated_at=monday_request.intended_entry_at,
        catalog_data_version="data-v2",
        daily=daily,
        hourly={},
        fifteen_minute={"USO": _monday_intraday()},
        news=(contextual_news,),
    )
    assert with_news.status is OilDecisionStatus.OBSERVE
    assert with_news.analysis.news == (contextual_news,)
    assert with_news.source_alignment.directional_contribution == 0


def test_stress_grid_and_snapshot_persistence_have_no_live_sizing(tmp_path: Path) -> None:
    result = _build(_request(13, 30))
    catastrophic = next(
        item
        for item in result.stress_table
        if item.leverage == 10 and item.adverse_move_fraction == 0.1
    )
    assert catastrophic.equity_loss_fraction == 1.0
    assert catastrophic.catastrophic
    assert catastrophic.liquidation_possible
    payload = result.model_dump(mode="json")
    assert result.actionable is False
    assert result.canonical_portfolio_weight == 0
    assert result.broker_profile.product_type == "NON_EXPIRING_CFD"
    assert result.broker_profile.max_modeled_leverage == 10
    assert result.broker_profile.daily_break == "21:00-22:00"
    assert not ({"order", "quantity", "notional", "position_size"} & set(payload))

    first = persist_oil_snapshot(tmp_path, result)
    second = persist_oil_snapshot(tmp_path, result)
    assert first == second
    assert first.name == f"{result.snapshot_id}.json"
    assert len(list((tmp_path / "oil_decisions" / "snapshots").glob("*.json"))) == 1
