"""Evidence-gated analysis for a user-selected tradable instrument."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats

from edgestack.exceptions import DataError
from edgestack.features.momentum import wilder_rsi
from edgestack.recommendation.hashing import stable_hash
from edgestack.recommendation.instrument_schemas import (
    AlignmentSummaryV2,
    DirectionalRating,
    EdgeEffectV2,
    EffectDirection,
    FrozenTimingArtifactV2,
    HorizonTimingAnalysisV2,
    InstrumentAnalysisStatus,
    InstrumentAnalysisV2,
    InstrumentKind,
    InstrumentResolutionV2,
    NewsEvidenceV2,
    TimingHorizon,
    TimingWindowV2,
)
from edgestack.recommendation.schemas import (
    AssetKind,
    CanonicalRecommendationBundleV2,
    EvidenceGrade,
    SleeveContributionV2,
)
from edgestack.recommendation.trade_calendar import build_trade_calendars
from edgestack.validation.clustered import session_effective_sample_size

_SYMBOL = re.compile(r"^[A-Z0-9.^=_-]{1,24}$")
_COMMODITY_ALIASES: dict[str, tuple[str, str]] = {
    "GOLD": ("GLD", "physical gold"),
    "XAU": ("GLD", "physical gold"),
    "SILVER": ("SLV", "physical silver"),
    "XAG": ("SLV", "physical silver"),
    "OIL": ("USO", "WTI crude oil"),
    "WTI": ("USO", "WTI crude oil"),
    "CRUDE": ("USO", "WTI crude oil"),
    "BRENT": ("BNO", "Brent crude oil"),
    "NATURALGAS": ("UNG", "natural gas"),
}
_KNOWN_ETFS = frozenset({"SPY", "TLT", "SHY", "GLD", "SLV", "USO", "BNO", "UNG"})
_WEEKDAY = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
_MONTH_BUCKET = {
    0: "sessions 1-5",
    1: "sessions 6-10",
    2: "sessions 11-15",
    3: "sessions 16-20",
    4: "session 21 through month-end",
}
_MONTH = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


@dataclass(frozen=True)
class _Candidate:
    horizon: TimingHorizon
    key: str
    label: str
    entry: str
    exit: str
    holding: int
    mean: float
    ci_low: float
    ci_high: float
    raw_pvalue: float
    observations: int
    ess: float


def resolve_instrument(
    symbol: str,
    *,
    requested_kind: InstrumentKind | None = None,
    canonical: CanonicalRecommendationBundleV2 | None = None,
) -> InstrumentResolutionV2:
    requested = re.sub(r"\s+", "", symbol).upper()
    if not requested or not _SYMBOL.fullmatch(requested):
        raise DataError("symbol must be 1-24 ticker characters (letters, numbers, .^=_-)")
    alias = _COMMODITY_ALIASES.get(requested)
    if alias:
        resolved, underlying = alias
        return InstrumentResolutionV2(
            requested_symbol=requested,
            resolved_symbol=resolved,
            instrument_kind=InstrumentKind.COMMODITY_PROXY,
            proxy_for=underlying,
            notes=(
                f"{resolved} is a tradable fund proxy, not spot {underlying}.",
                "Roll yield, fees, tracking error, and market hours can differ from the commodity.",
            ),
        )
    inferred = requested_kind
    if inferred is None and canonical is not None:
        kinds = {
            item.symbol: item.asset_kind
            for item in canonical.base_recommendation.unlevered_base_weights
        }
        kinds.update(
            {item.symbol: item.asset_kind for item in canonical.base_recommendation.watchlist}
        )
        if kinds.get(requested) is AssetKind.ETF:
            inferred = InstrumentKind.ETF
        elif kinds.get(requested) is AssetKind.STOCK:
            inferred = InstrumentKind.STOCK
    if inferred is None:
        inferred = InstrumentKind.ETF if requested in _KNOWN_ETFS else InstrumentKind.STOCK
    proxy_for = None
    notes: tuple[str, ...] = ()
    if requested.endswith("=F"):
        inferred = InstrumentKind.COMMODITY_PROXY
        proxy_for = "commodity futures contract"
        notes = ("Futures have expiry, roll, margin, and near-24-hour session risks.",)
    return InstrumentResolutionV2(
        requested_symbol=requested,
        resolved_symbol=requested,
        instrument_kind=inferred,
        proxy_for=proxy_for,
        notes=notes,
    )


def analyze_instrument(
    *,
    bundle: CanonicalRecommendationBundleV2,
    resolution: InstrumentResolutionV2,
    daily_bars: pd.DataFrame,
    intended_entry_at: datetime | None = None,
    rate_intraday_choice: bool = True,
    intraday_bars: pd.DataFrame | None = None,
    fifteen_minute_bars: pd.DataFrame | None = None,
    timing_artifacts: tuple[FrozenTimingArtifactV2, ...] = (),
    news: tuple[NewsEvidenceV2, ...] = (),
    round_trip_cost_bps: float = 10.0,
) -> InstrumentAnalysisV2:
    if not 0 <= round_trip_cost_bps <= 1_000:
        raise DataError("round-trip analysis cost must be between 0 and 1,000 bps")
    symbol = resolution.resolved_symbol
    daily = _prepare_daily(daily_bars, symbol=symbol, through=bundle.session)
    compatible_artifacts = _compatible_artifacts(timing_artifacts, bundle, symbol)
    effects = _edge_effects(bundle, symbol, daily, compatible_artifacts)
    tailwinds = tuple(effect for effect in effects if effect.direction is EffectDirection.TAILWIND)
    headwinds = tuple(effect for effect in effects if effect.direction is EffectDirection.HEADWIND)
    mixed = tuple(
        effect
        for effect in effects
        if effect.direction in {EffectDirection.MIXED, EffectDirection.NEUTRAL}
    )

    preferred_intraday = (
        fifteen_minute_bars
        if fifteen_minute_bars is not None and not fifteen_minute_bars.empty
        else intraday_bars
    )
    preferred_resolution = (
        "15-minute"
        if fifteen_minute_bars is not None and not fifteen_minute_bars.empty
        else "hourly"
    )
    analyses = (
        _intraday_analysis(
            preferred_intraday,
            symbol=symbol,
            intended=intended_entry_at,
            cost_bps=round_trip_cost_bps,
            artifact=_artifact_for(compatible_artifacts, TimingHorizon.DAY),
            resolution=preferred_resolution,
        ),
        _daily_horizon_analysis(
            daily,
            horizon=TimingHorizon.WEEK,
            intended=intended_entry_at,
            cost_bps=round_trip_cost_bps,
            artifact=_artifact_for(compatible_artifacts, TimingHorizon.WEEK),
        ),
        _daily_horizon_analysis(
            daily,
            horizon=TimingHorizon.MONTH,
            intended=intended_entry_at,
            cost_bps=round_trip_cost_bps,
            artifact=_artifact_for(compatible_artifacts, TimingHorizon.MONTH),
        ),
        _daily_horizon_analysis(
            daily,
            horizon=TimingHorizon.YEAR,
            intended=intended_entry_at,
            cost_bps=round_trip_cost_bps,
            artifact=_artifact_for(compatible_artifacts, TimingHorizon.YEAR),
        ),
    )
    promoted_tailwinds = sum(effect.promoted for effect in tailwinds)
    promoted_headwinds = sum(effect.promoted for effect in headwinds)
    actionable_horizons = tuple(item.horizon for item in analyses if item.actionable)
    missing = tuple(
        item.horizon.value.lower()
        for item in analyses
        if item.best_window is None or not item.actionable
    )
    aligned = (
        promoted_tailwinds > 0
        and promoted_headwinds == 0
        and set(actionable_horizons) == set(TimingHorizon)
    )
    if aligned:
        alignment_explanation = (
            "All four horizons have compatible promoted timing artifacts and promoted net "
            "evidence agrees. This remains uncertain and paper-only."
        )
    else:
        alignment_explanation = (
            "Not an all-stars-aligned trade: every day/week/month/year window must be backed by "
            "a compatible promoted artifact, with no promoted headwind."
        )
    alignment = AlignmentSummaryV2(
        aligned_trade=aligned,
        promoted_tailwinds=promoted_tailwinds,
        promoted_headwinds=promoted_headwinds,
        observational_tailwinds=sum(not effect.promoted for effect in tailwinds),
        observational_headwinds=sum(not effect.promoted for effect in headwinds),
        actionable_horizons=actionable_horizons,
        missing_inputs=missing,
        explanation=alignment_explanation,
    )

    promoted_effects = [effect for effect in effects if effect.promoted]
    promoted_net = sum(effect.net_contribution for effect in promoted_effects)
    if not promoted_effects:
        rating = DirectionalRating.NOT_RATED
        score = None
    else:
        score = float(100.0 * math.tanh(promoted_net / 0.10))
        if promoted_net > 1e-8:
            rating = DirectionalRating.POSITIVE
        elif promoted_net < -1e-8:
            rating = DirectionalRating.NEGATIVE
        else:
            rating = DirectionalRating.NEUTRAL

    enough_daily = len(daily) >= 252
    if not enough_daily:
        status = InstrumentAnalysisStatus.INSUFFICIENT_DATA
    elif aligned:
        status = InstrumentAnalysisStatus.ACTIONABLE
    elif promoted_effects:
        status = InstrumentAnalysisStatus.RESEARCH_ONLY
    else:
        status = InstrumentAnalysisStatus.INSUFFICIENT_EVIDENCE

    current_price = float(daily["close"].iloc[-1]) if not daily.empty else None
    weight = next(
        (
            item.weight
            for item in bundle.default_recommendation.personalized_target_weights
            if item.symbol == symbol
        ),
        0.0,
    )
    current_news = tuple(
        item for item in news if item.symbol == symbol and item.published_at <= bundle.as_of
    )
    calendars, choice_ratings, exit_plans, recheck_plan = build_trade_calendars(
        daily=daily,
        hourly=intraday_bars,
        fifteen_minute=fifteen_minute_bars,
        intended_entry_at=intended_entry_at,
        cost_bps=round_trip_cost_bps,
        timing_artifacts=compatible_artifacts,
        as_of=bundle.as_of,
        rate_intraday_choice=rate_intraday_choice,
    )
    warnings = list(resolution.notes)
    warnings.extend(
        [
            "Observational best/worst windows are searched descriptions, not promoted signals.",
            "Calendar, trend, oscillator, and news observations contribute zero to the rating "
            "until independently promoted.",
            "A precise best buy or sell time does not exist; limits, gaps, spreads, and fills can "
            "make realized timing materially worse.",
        ]
    )
    if not current_news:
        warnings.append("No compatible frozen news snapshot is available for this instrument.")
    if intraday_bars is None or intraday_bars.empty:
        warnings.append("No hourly history is available; the system abstains from naming an hour.")
    if fifteen_minute_bars is None or fifteen_minute_bars.empty:
        warnings.append(
            "No 15-minute history is available; the system abstains from a 15-minute calendar."
        )
    if not enough_daily:
        warnings.append(f"Only {len(daily)} daily sessions are available; at least 252 are needed.")

    return InstrumentAnalysisV2(
        analysis_id=stable_hash(
            {
                "symbol": resolution.model_dump(mode="json"),
                "intended_entry_at": intended_entry_at,
                "bundle_hash": bundle.bundle_hash,
                "daily_through": daily["date"].max() if not daily.empty else None,
                "cost_bps": round_trip_cost_bps,
                "timing_artifacts": [item.artifact_hash for item in compatible_artifacts],
                "hourly_through": _latest_timestamp(intraday_bars),
                "fifteen_minute_through": _latest_timestamp(fifteen_minute_bars),
            }
        ),
        resolution=resolution,
        as_of=bundle.as_of,
        intended_entry_at=intended_entry_at,
        data_version=bundle.data_version,
        artifact_version=bundle.artifact_version,
        policy_version=bundle.policy_version,
        current_price=current_price,
        status=status,
        overall_rating=rating,
        overall_score=score,
        canonical_portfolio_weight=weight,
        alignment=alignment,
        horizon_analyses=analyses,
        chosen_time_ratings=choice_ratings,
        exit_plans=exit_plans,
        tailwind_calendars=calendars,
        recheck_plan=recheck_plan,
        tailwinds=tailwinds,
        headwinds=headwinds,
        mixed_effects=mixed,
        news=current_news,
        what_to_watch=_watch_items(daily, current_news, resolution),
        current_year_notes=_year_notes(daily, bundle.as_of.year),
        warnings=tuple(warnings),
    )


def _prepare_daily(frame: pd.DataFrame, *, symbol: str, through: Any) -> pd.DataFrame:
    required = {"symbol", "date", "open", "high", "low", "close", "volume", "adj_close"}
    missing = required - set(frame.columns)
    if missing:
        raise DataError(f"instrument analysis daily bars missing {sorted(missing)}")
    output = frame.loc[frame["symbol"].astype(str).str.upper() == symbol].copy()
    output["date"] = pd.to_datetime(output["date"]).dt.normalize()
    output = output.loc[output["date"] <= pd.Timestamp(through)].sort_values("date")
    output = output.drop_duplicates("date", keep="last").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume", "adj_close"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.dropna(subset=["open", "close", "adj_close"])
    factor = output["adj_close"] / output["close"]
    if output.empty or not np.isfinite(factor).all() or (factor <= 0).any():
        raise DataError("adjusted-close history is required for total-return timing analysis")
    output["adjusted_open"] = output["open"] * factor
    output["session_in_month"] = output.groupby(output["date"].dt.to_period("M")).cumcount() + 1
    return output


def _compatible_artifacts(
    artifacts: tuple[FrozenTimingArtifactV2, ...],
    bundle: CanonicalRecommendationBundleV2,
    symbol: str,
) -> tuple[FrozenTimingArtifactV2, ...]:
    output = []
    for artifact in artifacts:
        if artifact.symbol != symbol:
            continue
        versions = (artifact.data_version, artifact.artifact_version, artifact.policy_version)
        expected = (bundle.data_version, bundle.artifact_version, bundle.policy_version)
        if versions != expected:
            raise DataError(f"timing artifact version mismatch for {symbol}/{artifact.horizon}")
        output.append(artifact)
    return tuple(output)


def _artifact_for(
    artifacts: tuple[FrozenTimingArtifactV2, ...], horizon: TimingHorizon
) -> FrozenTimingArtifactV2 | None:
    selected = [item for item in artifacts if item.horizon is horizon]
    if len(selected) > 1:
        raise DataError(f"multiple promoted timing artifacts for one {horizon} horizon")
    return selected[0] if selected else None


def _edge_effects(
    bundle: CanonicalRecommendationBundleV2,
    symbol: str,
    daily: pd.DataFrame,
    timing_artifacts: tuple[FrozenTimingArtifactV2, ...],
) -> tuple[EdgeEffectV2, ...]:
    output: list[EdgeEffectV2] = []
    sleeves = (
        *bundle.base_recommendation.promoted_sleeves,
        *bundle.base_recommendation.promoted_compound_sleeves,
    )
    for sleeve in sleeves:
        symbol_weight = next(
            (item.weight for item in sleeve.symbol_weights if item.symbol == symbol), 0
        )
        if abs(symbol_weight) <= 1e-12:
            continue
        output.append(_promoted_sleeve_effect(sleeve, symbol_weight))
    for artifact in timing_artifacts:
        if any(effect.edge_id == artifact.sleeve_id for effect in output):
            continue
        expected_contribution = artifact.incremental_expected_net_return
        contribution = artifact.incremental_lower_95
        positive_contribution = max(0.0, expected_contribution, contribution)
        uncertainty_drag = contribution - positive_contribution
        output.append(
            EdgeEffectV2(
                edge_id=artifact.sleeve_id,
                family="promoted_timing",
                horizon_sessions=artifact.holding_sessions,
                direction=(
                    EffectDirection.TAILWIND if contribution > 0 else EffectDirection.HEADWIND
                ),
                observation=artifact.label,
                positive_contribution=positive_contribution,
                negative_contribution=uncertainty_drag,
                net_contribution=contribution,
                protective_avoidance_value=max(0.0, -contribution),
                lower_95=artifact.incremental_lower_95,
                adverse_counter_effect=(
                    "Even a positive promoted timing edge can lose through gaps, crowding, "
                    "slippage, or a regime break."
                ),
                protective_counter_effect=(
                    "A negative timing estimate can be useful by avoiding exposure during its "
                    "validated window."
                ),
                evidence_grade=EvidenceGrade.PROMOTED,
                compound=artifact.compound,
                promoted=True,
                artifact_hash=artifact.artifact_hash,
                invalidation="Any artifact/version mismatch or failed monitoring check.",
            )
        )
    output.extend(_technical_observations(daily))
    for watch in bundle.base_recommendation.watchlist:
        if watch.symbol != symbol:
            continue
        output.append(
            EdgeEffectV2(
                edge_id=f"watchlist:{watch.family}:{watch.horizon_sessions}",
                family=watch.family,
                horizon_sessions=watch.horizon_sessions,
                direction=EffectDirection.MIXED,
                observation=watch.thesis,
                adverse_counter_effect=watch.zero_weight_reason,
                protective_counter_effect=(
                    "Keeping the idea at zero weight avoids treating incomplete evidence as edge."
                ),
                evidence_grade=watch.evidence_grade,
                invalidation=watch.zero_weight_reason,
            )
        )
    baseline = next(
        (item for item in bundle.baseline_policy.weights if item.symbol == symbol), None
    )
    if baseline is not None:
        output.append(
            EdgeEffectV2(
                edge_id=f"policy:{bundle.policy_version}:{symbol}",
                family="diversified_baseline_policy",
                horizon_sessions=63,
                direction=EffectDirection.NEUTRAL,
                observation=f"{symbol} has a {baseline.weight:.1%} unlevered policy weight.",
                adverse_counter_effect=(
                    "Policy inclusion is diversification, not an alpha forecast."
                ),
                protective_counter_effect=(
                    "The policy finances exposure across distinct asset classes."
                ),
                evidence_grade=EvidenceGrade.POLICY,
                invalidation="Invalid baseline data or incompatible policy version.",
            )
        )
    return tuple(output)


def _promoted_sleeve_effect(sleeve: SleeveContributionV2, symbol_weight: float) -> EdgeEffectV2:
    expected_contribution = sleeve.expected_net_return * symbol_weight
    contribution = sleeve.expected_return_lower_95 * symbol_weight
    positive_contribution = max(0.0, expected_contribution, contribution)
    uncertainty_drag = contribution - positive_contribution
    return EdgeEffectV2(
        edge_id=sleeve.sleeve_id,
        family=sleeve.family,
        horizon_sessions=sleeve.horizon_sessions,
        direction=EffectDirection.TAILWIND if contribution > 0 else EffectDirection.HEADWIND,
        observation=(
            f"Promoted sleeve contribution at internal symbol weight {symbol_weight:.2%}; "
            "horizons remain separate until portfolio exposure."
        ),
        positive_contribution=positive_contribution,
        negative_contribution=uncertainty_drag,
        net_contribution=contribution,
        protective_avoidance_value=max(0.0, -contribution),
        lower_95=sleeve.expected_return_lower_95 * symbol_weight,
        adverse_counter_effect=(
            "Costs, covariance, uncertainty, capacity, and adverse execution reduce the gross edge."
        ),
        protective_counter_effect=(
            "When the contribution turns negative, abstaining can preserve risk budget rather than "
            "forcing a trade."
        ),
        evidence_grade=EvidenceGrade.PROMOTED,
        compound=sleeve.compound,
        promoted=True,
        artifact_hash=sleeve.artifact_hash,
        invalidation="Sleeve monitoring failure, artifact mismatch, or promotion withdrawal.",
    )


def _technical_observations(daily: pd.DataFrame) -> list[EdgeEffectV2]:
    if len(daily) < 20:
        return []
    close = daily["adj_close"].astype(float)
    output: list[EdgeEffectV2] = []

    if len(close) >= 200:
        sma200 = float(close.tail(200).mean())
        distance = float(close.iloc[-1] / sma200 - 1.0)
        direction = EffectDirection.TAILWIND if distance >= 0 else EffectDirection.HEADWIND
        output.append(
            _observation(
                "observation:trend_200",
                "momentum",
                60,
                direction,
                f"Adjusted close is {distance:+.1%} versus its 200-session average.",
                "A strong positive trend can be crowded, extended, and vulnerable to gap reversal.",
                "A negative trend can improve future entry price, but is not a reversal signal.",
                "A cross below/above the frozen trend rule or stale close.",
            )
        )
    if len(close) >= 61:
        momentum = float(close.iloc[-1] / close.iloc[-61] - 1.0)
        output.append(
            _observation(
                "observation:momentum_60",
                "momentum",
                20,
                EffectDirection.TAILWIND if momentum >= 0 else EffectDirection.HEADWIND,
                f"Trailing 60-session total-return momentum is {momentum:+.1%}.",
                "Positive momentum can increase the price paid and downside if the regime "
                "reverses.",
                "Negative momentum warns against chasing and can make waiting valuable.",
                "A change in the 60-session sign; observational only.",
            )
        )
    if len(close) >= 15:
        rsi = float(wilder_rsi(close, 14).iloc[-1])
        if rsi >= 70:
            direction = EffectDirection.HEADWIND
        elif rsi <= 30:
            direction = EffectDirection.MIXED
        else:
            direction = EffectDirection.NEUTRAL
        output.append(
            _observation(
                "observation:rsi_14",
                "mean_reversion",
                5,
                direction,
                f"Wilder RSI(14) is {rsi:.1f}.",
                "An oversold reading may reflect persistent adverse information, not a bargain.",
                "An overbought reading can confirm durable demand despite mean-reversion risk.",
                "RSI is descriptive and never a standalone trigger without promotion.",
            )
        )
    returns = close.pct_change(fill_method=None).dropna()
    if len(returns) >= 126:
        short_vol = float(returns.tail(20).std(ddof=1) * math.sqrt(252))
        long_vol = float(returns.tail(126).std(ddof=1) * math.sqrt(252))
        ratio = short_vol / long_vol if long_vol > 0 else math.inf
        direction = EffectDirection.HEADWIND if ratio > 1.25 else EffectDirection.MIXED
        output.append(
            _observation(
                "observation:volatility_regime",
                "volatility_regime",
                20,
                direction,
                f"20-session volatility is {short_vol:.1%} versus {long_vol:.1%} over "
                "126 sessions.",
                "Low recent volatility can conceal jump risk and encourage excessive sizing.",
                "High volatility reduces permitted sizing and can prevent a larger loss.",
                "A stressed-volatility or correlation change; portfolio risk limits bind first.",
            )
        )
    return output


def _observation(
    edge_id: str,
    family: str,
    horizon: int,
    direction: EffectDirection,
    text: str,
    adverse: str,
    protective: str,
    invalidation: str,
) -> EdgeEffectV2:
    return EdgeEffectV2(
        edge_id=edge_id,
        family=family,
        horizon_sessions=horizon,
        direction=direction,
        observation=text,
        adverse_counter_effect=adverse,
        protective_counter_effect=protective,
        evidence_grade=EvidenceGrade.INSUFFICIENT,
        invalidation=invalidation,
    )


def _daily_horizon_analysis(
    daily: pd.DataFrame,
    *,
    horizon: TimingHorizon,
    intended: datetime | None,
    cost_bps: float,
    artifact: FrozenTimingArtifactV2 | None,
) -> HorizonTimingAnalysisV2:
    if len(daily) < 252:
        return HorizonTimingAnalysisV2(
            horizon=horizon,
            data_resolution="daily adjusted-open total return",
            warning="At least 252 daily sessions are required.",
        )
    candidates = _seasonal_candidates(daily, horizon=horizon, cost_bps=cost_bps)
    return _assemble_horizon(candidates, horizon, intended, artifact, "daily adjusted-open")


def _seasonal_candidates(
    daily: pd.DataFrame, *, horizon: TimingHorizon, cost_bps: float
) -> list[_Candidate]:
    if horizon is TimingHorizon.WEEK:
        keys = daily["date"].dt.weekday
        holdings = (1, 2, 5)
    elif horizon is TimingHorizon.MONTH:
        keys = ((daily["session_in_month"] - 1) // 5).clip(upper=4)
        holdings = (5, 10, 20)
    elif horizon is TimingHorizon.YEAR:
        keys = daily["date"].dt.month
        holdings = (20, 60, 126)
    else:
        raise DataError(f"unsupported daily timing horizon {horizon}")

    def entry(key: int) -> str:
        if horizon is TimingHorizon.WEEK:
            return f"{_WEEKDAY[key]} next-open"
        if horizon is TimingHorizon.MONTH:
            return f"Month {_MONTH_BUCKET[key]}, next-open"
        return f"{_MONTH[key]}, next-open"

    gross_cost = cost_bps / 10_000.0
    raw: list[_Candidate] = []
    for holding in holdings:
        future = daily["adjusted_open"].shift(-holding) / daily["adjusted_open"] - 1.0
        net = future - gross_cost
        for key in sorted(set(keys.dropna().astype(int))):
            values = net.loc[keys == key].dropna().astype(float)
            if len(values) < 20:
                continue
            ess = max(1.0, session_effective_sample_size(values))
            mean = float(values.mean())
            standard_error = float(values.std(ddof=1) / math.sqrt(ess))
            t_stat = mean / standard_error if standard_error > 0 else 0.0
            pvalue = float(2.0 * stats.t.sf(abs(t_stat), df=max(1, len(values) - 1)))
            label = f"{entry(key)} → {holding} sessions"
            raw.append(
                _Candidate(
                    horizon=horizon,
                    key=str(key),
                    label=label,
                    entry=entry(key),
                    exit=f"Next-open {holding} trading sessions after entry",
                    holding=holding,
                    mean=mean,
                    ci_low=mean - 1.96 * standard_error,
                    ci_high=mean + 1.96 * standard_error,
                    raw_pvalue=pvalue,
                    observations=len(values),
                    ess=ess,
                )
            )
    return raw


def _intraday_analysis(
    frame: pd.DataFrame | None,
    *,
    symbol: str,
    intended: datetime | None,
    cost_bps: float,
    artifact: FrozenTimingArtifactV2 | None,
    resolution: str = "hourly",
) -> HorizonTimingAnalysisV2:
    if frame is None or frame.empty:
        if artifact is None:
            return HorizonTimingAnalysisV2(
                horizon=TimingHorizon.DAY,
                data_resolution=resolution,
                warning=(
                    "Hourly history is unavailable. No best or worst hour is inferred from "
                    "daily bars."
                ),
            )
        return _assemble_horizon(
            [], TimingHorizon.DAY, intended, artifact, f"frozen {resolution} rule"
        )
    required = {"symbol", "timestamp", "open", "close"}
    if missing := required - set(frame.columns):
        raise DataError(f"intraday bars missing {sorted(missing)}")
    bars = frame.loc[frame["symbol"].astype(str).str.upper() == symbol].copy()
    timestamps = pd.to_datetime(bars["timestamp"], utc=True)
    bars["timestamp"] = timestamps
    bars["session"] = timestamps.dt.tz_convert("America/New_York").dt.date
    bars["slot"] = timestamps.dt.tz_convert("America/New_York").dt.strftime("%H:%M")
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    raw: list[_Candidate] = []
    for holding in (1, 2, 4):
        future = bars.groupby("session")["open"].shift(-holding) / bars["open"] - 1.0
        net = future - cost_bps / 10_000.0
        for slot in sorted(set(bars["slot"])):
            values = net.loc[bars["slot"] == slot].dropna().astype(float)
            if len(values) < 20:
                continue
            ess = max(1.0, session_effective_sample_size(values))
            mean = float(values.mean())
            se = float(values.std(ddof=1) / math.sqrt(ess))
            t_stat = mean / se if se > 0 else 0.0
            raw.append(
                _Candidate(
                    horizon=TimingHorizon.DAY,
                    key=slot,
                    label=f"{slot} New York → {holding} {resolution} bars",
                    entry=f"At or after the {slot} America/New_York bar open",
                    exit=(
                        f"{holding} {resolution} bars later; never carry merely because of "
                        "this study"
                    ),
                    holding=1,
                    mean=mean,
                    ci_low=mean - 1.96 * se,
                    ci_high=mean + 1.96 * se,
                    raw_pvalue=float(2.0 * stats.t.sf(abs(t_stat), df=max(1, len(values) - 1))),
                    observations=len(values),
                    ess=ess,
                )
            )
    return _assemble_horizon(raw, TimingHorizon.DAY, intended, artifact, resolution)


def _latest_timestamp(frame: pd.DataFrame | None) -> str | None:
    if frame is None or frame.empty or "timestamp" not in frame:
        return None
    return pd.to_datetime(frame["timestamp"], utc=True).max().isoformat()


def _assemble_horizon(
    candidates: list[_Candidate],
    horizon: TimingHorizon,
    intended: datetime | None,
    artifact: FrozenTimingArtifactV2 | None,
    resolution: str,
) -> HorizonTimingAnalysisV2:
    variants = len(candidates)
    windows = [_candidate_window(item, max(1, variants)) for item in candidates]
    windows.sort(
        key=lambda item: (
            item.expected_net_return if item.expected_net_return is not None else -math.inf
        ),
        reverse=True,
    )
    best = windows[0] if windows else None
    worst = windows[-1] if windows else None
    if artifact is not None:
        best = TimingWindowV2(
            horizon=horizon,
            label=artifact.label,
            entry_window=artifact.entry_window,
            exit_window=artifact.exit_window,
            holding_sessions=artifact.holding_sessions,
            expected_net_return=artifact.expected_net_return,
            lower_95=artifact.lower_95,
            upper_95=artifact.upper_95,
            multiple_testing_adjusted_pvalue=artifact.multiple_testing_adjusted_pvalue,
            observations=artifact.observations,
            effective_sample_size=artifact.effective_sample_size,
            evidence_grade=EvidenceGrade.PROMOTED,
            actionable=True,
            artifact_hash=artifact.artifact_hash,
            rationale="Frozen V2 timing rule that passed the full promotion family.",
            what_invalidates_it=artifact.what_invalidates_it,
        )
    alternatives = tuple(item for item in windows[:3] if item != best)
    assessment = _intended_assessment(candidates, horizon, intended, best, worst)
    warning = None
    if artifact is None:
        warning = (
            f"All {variants} displayed variants are observational and multiple-testing adjusted; "
            "they cannot authorize a trade."
        )
    return HorizonTimingAnalysisV2(
        horizon=horizon,
        data_resolution=resolution,
        best_window=best,
        worst_window=worst,
        alternatives=alternatives,
        intended_entry_assessment=assessment,
        actionable=bool(best and best.actionable),
        searched_variants=variants,
        warning=warning,
    )


def _candidate_window(candidate: _Candidate, trials: int) -> TimingWindowV2:
    adjusted = min(1.0, candidate.raw_pvalue * trials)
    return TimingWindowV2(
        horizon=candidate.horizon,
        label=candidate.label,
        entry_window=candidate.entry,
        exit_window=candidate.exit,
        holding_sessions=candidate.holding,
        expected_net_return=candidate.mean,
        lower_95=candidate.ci_low,
        upper_95=candidate.ci_high,
        multiple_testing_adjusted_pvalue=adjusted,
        observations=candidate.observations,
        effective_sample_size=candidate.ess,
        evidence_grade=EvidenceGrade.INSUFFICIENT,
        rationale=(
            "Historical adjusted-open conditional mean after the requested round-trip cost. "
            "This was selected from the displayed family and is not nested/promoted evidence."
        ),
        what_invalidates_it=(
            "Non-positive lower confidence bound",
            "Multiple-testing-adjusted p-value above 0.05",
            "Stale data, wide spreads, news gaps, or regime change",
        ),
    )


def _intended_assessment(
    candidates: list[_Candidate],
    horizon: TimingHorizon,
    intended: datetime | None,
    best: TimingWindowV2 | None,
    worst: TimingWindowV2 | None,
) -> str | None:
    if intended is None:
        return None
    timestamp = (
        intended.astimezone(ZoneInfo("America/New_York"))
        if intended.tzinfo
        else intended.replace(tzinfo=ZoneInfo("America/New_York"))
    )
    if horizon is TimingHorizon.DAY:
        requested_minutes = timestamp.hour * 60 + timestamp.minute
        available = sorted({item.key for item in candidates})
        closest = min(
            available,
            key=lambda value: abs(
                int(value.split(":", maxsplit=1)[0]) * 60
                + int(value.split(":", maxsplit=1)[1])
                - requested_minutes
            ),
            default=None,
        )
        matching = [item for item in candidates if item.key == closest]
    elif horizon is TimingHorizon.WEEK:
        matching = [item for item in candidates if item.key == str(timestamp.weekday())]
    elif horizon is TimingHorizon.MONTH:
        bucket = min(4, (timestamp.day - 1) // 5)
        matching = [item for item in candidates if item.key == str(bucket)]
    else:
        matching = [item for item in candidates if item.key == str(timestamp.month)]
    if not matching:
        return "The intended entry has no compatible historical timing sample."
    selected = max(matching, key=lambda item: item.mean)
    context = f"Best matching historical hold: {selected.label}, net mean {selected.mean:+.2%}."
    if worst and selected.label == worst.label:
        context += (
            " It matches the displayed worst window; if exposure is unavoidable, reduce size, "
            "use a limit, and require a predefined invalidation."
        )
    elif best and selected.label == best.label:
        context += " It matches the displayed best window, which is still uncertain."
    if best and best.actionable:
        context += " Compare it with the promoted window; do not substitute this observation."
    elif best and worst:
        context += " Both the displayed best and worst windows remain non-actionable research."
    return context


def _watch_items(
    daily: pd.DataFrame,
    news: tuple[NewsEvidenceV2, ...],
    resolution: InstrumentResolutionV2,
) -> tuple[str, ...]:
    items = [
        "Confirm the canonical data/artifact versions and freshness before the session.",
        "Use a limit or participation-aware order; verify spread, depth, and next-open gap.",
        "Predefine maximum position risk and an invalidation condition; do not average down "
        "by default.",
        "Check earnings, distributions, contract rolls, macro releases, and market holidays.",
    ]
    if len(daily) >= 20:
        median_dollar_volume = float((daily["close"] * daily["volume"]).tail(60).median())
        items.append(
            f"60-session median daily dollar volume is about ${median_dollar_volume:,.0f}."
        )
    if resolution.instrument_kind is InstrumentKind.COMMODITY_PROXY:
        items.append(
            "Check futures curve/roll yield and proxy tracking error before using commodity "
            "exposure."
        )
    if news:
        items.append(
            "Read the underlying source; headline sentiment is context and can be wrong or stale."
        )
    return tuple(items)


def _year_notes(daily: pd.DataFrame, year: int) -> tuple[str, ...]:
    if daily.empty:
        return (f"No compatible {year} price observations are available.",)
    selected = daily.loc[daily["date"].dt.year == year]
    if selected.empty:
        return (f"No compatible {year} price observations are available.",)
    ytd = float(selected["adj_close"].iloc[-1] / selected["adj_close"].iloc[0] - 1.0)
    high = float(selected["high"].max())
    low = float(selected["low"].min())
    return (
        f"{year} year-to-date adjusted-close return through the canonical session is {ytd:+.1%}.",
        f"Observed {year} raw trading range is {low:.2f} to {high:.2f} across "
        f"{len(selected)} sessions.",
        "Partial-year performance is descriptive, not an annual forecast or a reason to "
        "chase price.",
    )
