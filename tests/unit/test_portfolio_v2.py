"""Profile-independent portfolio construction acceptance tests."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from edgestack.recommendation.policy import load_baseline_policy
from edgestack.recommendation.portfolio import (
    AssetMetadata,
    CovarianceEstimate,
    ExpectedReturnEstimate,
    build_base_recommendation,
    estimate_covariance,
    optimize_base_portfolio,
)
from edgestack.recommendation.schemas import (
    AssetKind,
    EvidenceGrade,
    FreshnessV2,
    RecommendationStatus,
    SleeveContributionV2,
    WatchlistEntryV2,
    WeightV2,
)


def _returns(symbols: tuple[str, ...], sessions: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    common = rng.normal(0.0002, 0.006, size=(sessions, 1))
    values = common + rng.normal(0, 0.004, size=(sessions, len(symbols)))
    return pd.DataFrame(
        values, columns=symbols, index=pd.bdate_range("2024-01-02", periods=sessions)
    )


def _metadata(symbols: tuple[str, ...]) -> dict[str, AssetMetadata]:
    sectors = {"SPY": "broad_equity", "TLT": "treasury", "SHY": "treasury", "GLD": "gold"}
    return {
        symbol: AssetMetadata(
            symbol=symbol,
            asset_kind=AssetKind.ETF,
            sector=sectors.get(symbol, "broad"),
            median_adv=1_000_000_000,
            broad_policy_asset=True,
        )
        for symbol in symbols
    }


def _freshness(*, fresh: bool = True) -> FreshnessV2:
    return FreshnessV2(
        as_of=datetime(2025, 2, 24, tzinfo=UTC),
        expected_session=date(2025, 2, 24),
        is_fresh=fresh,
        age_business_days=0 if fresh else 2,
        complete=True,
        compatible=True,
    )


def test_covariance_is_deterministic_psd_and_stress_preserves_higher_volatility() -> None:
    returns = _returns(("SPY", "TLT", "SHY", "GLD"))
    first = estimate_covariance(returns)
    second = estimate_covariance(returns.copy())

    assert first.version == second.version
    np.testing.assert_allclose(first.annualized, second.annualized)
    assert np.linalg.eigvalsh(first.annualized).min() >= -1e-10
    assert np.linalg.eigvalsh(first.stressed_annualized).min() >= -1e-10
    assert np.all(np.diag(first.stressed_annualized) >= np.diag(first.annualized))


def test_optimizer_preserves_baseline_without_positive_lower_confidence_alpha() -> None:
    symbols = ("SPY", "TLT", "SHY", "GLD")
    covariance = estimate_covariance(_returns(symbols))
    baseline = dict.fromkeys(symbols, 0.25)
    result = optimize_base_portfolio(
        baseline_weights=baseline,
        expected_returns_lower_95={"SPY": -0.01},
        covariance=covariance,
        metadata=_metadata(symbols),
    )

    assert result.weights == baseline
    assert result.expected_net_return == 0


def test_optimizer_enforces_stock_sector_and_liquidation_caps() -> None:
    symbols = ("SPY", "AAA", "BBB")
    covariance = CovarianceEstimate(
        symbols=symbols,
        annualized=np.diag([0.04, 0.04, 0.04]),
        stressed_annualized=np.diag([0.06, 0.06, 0.06]),
        version="synthetic",
    )
    metadata = {
        "SPY": AssetMetadata("SPY", AssetKind.ETF, "broad", 1_000_000_000, True),
        "AAA": AssetMetadata("AAA", AssetKind.STOCK, "technology", 1_000_000),
        "BBB": AssetMetadata("BBB", AssetKind.STOCK, "technology", 2_000),
    }
    result = optimize_base_portfolio(
        baseline_weights={"SPY": 1.0, "AAA": 0.0, "BBB": 0.0},
        expected_returns_lower_95={"AAA": 2.0, "BBB": 2.0},
        covariance=covariance,
        metadata=metadata,
        account_equity_reference=100_000,
        max_stock_weight=0.03,
        max_sector_weight=0.04,
    )

    assert result.success
    assert result.weights["AAA"] <= 0.03 + 1e-8
    assert result.weights["BBB"] <= 0.01 + 1e-8
    assert result.weights["AAA"] + result.weights["BBB"] <= 0.04 + 1e-8
    assert abs(sum(result.weights.values()) - 1) < 1e-8


def test_no_promoted_sleeve_preserves_fixed_baseline_even_with_estimates() -> None:
    symbols = ("SPY", "TLT", "SHY", "GLD")
    as_of = pd.Timestamp(datetime(2025, 2, 24, tzinfo=UTC))
    watch = WatchlistEntryV2(
        symbol="AAPL",
        asset_kind=AssetKind.STOCK,
        horizon_sessions=5,
        family="momentum",
        thesis="prospective shadow candidate",
        zero_weight_reason="requires 252 sessions and ESS >= 100",
    )
    base = build_base_recommendation(
        policy=load_baseline_policy(),
        promoted_sleeves=(),
        watchlist=(watch,),
        returns=_returns(symbols),
        metadata=_metadata(symbols),
        expected_estimates={"SPY": ExpectedReturnEstimate(1.0, 0.01, 1_000)},
        freshness=_freshness(),
        as_of=as_of,
        execution_at=as_of + timedelta(days=1),
        data_version="data-v1",
        artifact_version="artifact-v1",
    )

    assert base.status is RecommendationStatus.WATCHLIST_ONLY
    assert {item.symbol: item.weight for item in base.unlevered_base_weights} == {
        "SPY": 0.25,
        "TLT": 0.25,
        "SHY": 0.25,
        "GLD": 0.25,
    }
    assert base.expected_net_return == 0


def test_promoted_sleeve_can_receive_weight_but_stale_data_targets_cash() -> None:
    symbols = ("SPY", "TLT", "SHY", "GLD", "AAA")
    metadata = _metadata(symbols)
    metadata["AAA"] = AssetMetadata("AAA", AssetKind.STOCK, "technology", 10_000_000)
    sleeve = SleeveContributionV2(
        sleeve_id="stock-momentum-5d",
        artifact_hash="abc",
        horizon_sessions=5,
        family="momentum",
        symbol_weights=(WeightV2(symbol="AAA", weight=1.0, asset_kind=AssetKind.STOCK),),
        expected_net_return=0.40,
        expected_return_lower_95=0.30,
        effective_sample_size=150,
        evidence_grade=EvidenceGrade.PROMOTED,
    )
    common = {
        "policy": load_baseline_policy(),
        "promoted_sleeves": (sleeve,),
        "watchlist": (),
        "returns": _returns(symbols),
        "metadata": metadata,
        "expected_estimates": {"AAA": ExpectedReturnEstimate(0.40, 0.01, 150)},
        "as_of": pd.Timestamp(datetime(2025, 2, 24, tzinfo=UTC)),
        "execution_at": pd.Timestamp(datetime(2025, 2, 25, tzinfo=UTC)),
        "data_version": "data-v1",
        "artifact_version": "artifact-v1",
    }
    active = build_base_recommendation(freshness=_freshness(), **common)
    stale = build_base_recommendation(freshness=_freshness(fresh=False), **common)

    assert active.status is RecommendationStatus.ACTIVE
    aaa_weight = next(w.weight for w in active.unlevered_base_weights if w.symbol == "AAA")
    assert 0 < aaa_weight <= 0.03 + 1e-10
    assert stale.status is RecommendationStatus.NO_ALLOCATION
    assert all(weight.weight == 0 for weight in stale.unlevered_base_weights)
