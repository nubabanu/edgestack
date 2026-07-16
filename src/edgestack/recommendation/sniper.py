"""Loss-aversion-first sniper shadow engine.

This module implements the user's staged specification without granting V2
portfolio weight.  All signals are determined from data available at the close;
the primary engine can only plan a next-open entry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from edgestack.data.calendar import TradingCalendar
from edgestack.exceptions import DataError
from edgestack.features.momentum import wilder_rsi
from edgestack.recommendation.schemas import CanonicalRecommendationBundleV2
from edgestack.recommendation.sniper_schemas import (
    OverlayState,
    SniperActivation,
    SniperCandidateStatus,
    SniperCandidateV2,
    SniperOutcomeEvidenceV2,
    SniperOverlayV2,
    SniperPlanV2,
    SniperPolicyItemV2,
    SniperRole,
    SniperSizingV2,
    SniperStage,
    SniperStrategyId,
)
from edgestack.validation.clustered import session_effective_sample_size

ROUND_TRIP_COST = 0.0010
MIN_RISK_OUTCOMES = 30
FALLBACK_ADVERSE_MOVE = -0.04
EXCLUDED_IDS = (
    SniperStrategyId.A3_MINOR_HOLIDAY,
    SniperStrategyId.A4_SMALL_CAP_JANUARY,
    SniperStrategyId.A5_WEEKEND_MONDAY,
    SniperStrategyId.B_NAIVE_OVERNIGHT,
    SniperStrategyId.D_SHORT_VOL_INCOME,
)


@dataclass(frozen=True)
class _SignalOutcome:
    net_return: float
    adverse_move: float


def sniper_policy_ranking() -> tuple[SniperPolicyItemV2, ...]:
    return (
        SniperPolicyItemV2(
            rank=1,
            strategy_id=SniperStrategyId.C1_C2_PRIMARY,
            stage=SniperStage.STAGE_1,
            role=SniperRole.PRIMARY_ENGINE,
            activation=SniperActivation.SHADOW_READY,
            conviction="HIGHEST",
            rule=(
                "SPY or approved low-vol index ETF: RSI(2)<5 OR three adjusted-close down "
                "sessions; vehicle and SPY above 200-DMA; both 20-session annualized "
                "volatilities <=20%; skip September; enter next-open; exit close>5-DMA or day 4."
            ),
            reason="Highest-ranked decorrelated dip engine; still requires frozen V2 promotion.",
        ),
        SniperPolicyItemV2(
            rank=2,
            strategy_id=SniperStrategyId.A1_SANTA,
            stage=SniperStage.STAGE_1,
            role=SniperRole.FIXED_CALENDAR_TRADE,
            activation=SniperActivation.SHADOW_READY,
            conviction="HIGH",
            rule=(
                "SPY enter at the close of the fifth-to-last December session; exit at the "
                "close of the second January session."
            ),
            reason=(
                "Clean fixed window, but historical access makes it shadow-only until promotion."
            ),
        ),
        SniperPolicyItemV2(
            rank=3,
            strategy_id=SniperStrategyId.A2_PRE_FOMC,
            stage=SniperStage.STAGE_2,
            role=SniperRole.DEFINED_RISK_TRADE,
            activation=SniperActivation.BLOCKED_DATA,
            conviction="MEDIUM",
            rule="Long call or 1-2-strike call spread expiring just after FOMC; premium=max loss.",
            reason=(
                "Decayed and options-only; point-in-time FOMC, chain, IV-rank, and fills absent."
            ),
        ),
        SniperPolicyItemV2(
            rank=4,
            strategy_id=SniperStrategyId.E1_TLT_MONTH_END,
            stage=SniperStage.STAGE_2,
            role=SniperRole.MODEST_CROSS_ASSET_TRADE,
            activation=SniperActivation.BLOCKED_VALIDATION,
            conviction="MODEST",
            rule="TLT enter 2-3 sessions before month-end; exit first trading session next month.",
            reason="Stage 2 remains locked until Stage 1 has a compatible promoted artifact.",
        ),
        SniperPolicyItemV2(
            rank=5,
            strategy_id=SniperStrategyId.C3_VIX_CONTANGO,
            stage=SniperStage.STAGE_3,
            role=SniperRole.VETO_FILTER,
            activation=SniperActivation.FILTER_ONLY,
            conviction="HIGH",
            rule="Backwardation vetoes; a validated contango re-cross may confirm C1/C2.",
            reason="Powerful as a regime filter, forbidden as a standalone trade.",
        ),
        SniperPolicyItemV2(
            rank=6,
            strategy_id=SniperStrategyId.C4_PUTCALL_BREADTH,
            stage=SniperStage.STAGE_3,
            role=SniperRole.SIZE_CONFIRMATION,
            activation=SniperActivation.FILTER_ONLY,
            conviction="MEDIUM",
            rule="Put/call plus breadth washout may confirm a triggered C1/C2 signal only.",
            reason="Secondary confirmation; cannot initiate or override the risk veto.",
        ),
        SniperPolicyItemV2(
            rank=7,
            strategy_id=SniperStrategyId.A6_OPEX_TILT,
            stage=SniperStage.STAGE_3,
            role=SniperRole.MICRO_TILT,
            activation=SniperActivation.BLOCKED_VALIDATION,
            conviction="LOW",
            rule="OpEx-week micro-tilt on an existing eligible position only.",
            reason="Marginal and decaying; no incremental promoted artifact exists.",
        ),
        SniperPolicyItemV2(
            rank=8,
            strategy_id=SniperStrategyId.E2_GOLD_JANUARY,
            stage=SniperStage.STAGE_2,
            role=SniperRole.LOW_CONVICTION_WATCHLIST,
            activation=SniperActivation.BLOCKED_VALIDATION,
            conviction="LOW",
            rule="January GLD observation only.",
            reason="Overlaps existing window and lacks incremental promoted evidence.",
        ),
        *tuple(
            SniperPolicyItemV2(
                rank=rank,
                strategy_id=strategy_id,
                stage=SniperStage.EXCLUDED,
                role=SniperRole.FORBIDDEN,
                activation=SniperActivation.EXCLUDED,
                conviction="REJECTED",
                rule="Hard-disabled: cannot generate candidates, overlays, or portfolio weights.",
                reason=(
                    "Weakly supported, likely data-mined, decayed, or has unacceptable tail risk."
                ),
            )
            for rank, strategy_id in enumerate(EXCLUDED_IDS, start=9)
        ),
    )


def build_sniper_plan(
    *,
    bundle: CanonicalRecommendationBundleV2,
    panel: pd.DataFrame,
    calendar: TradingCalendar,
    account_equity: float,
    max_tolerable_loss: float,
    vehicle: str = "SPY",
    vix_contango: float | None = None,
    put_call_ratio: float | None = None,
) -> SniperPlanV2:
    if account_equity <= 0 or max_tolerable_loss <= 0:
        raise DataError("sniper equity and max tolerable loss must be positive")
    if max_tolerable_loss > account_equity:
        raise DataError("sniper max tolerable loss cannot exceed account equity")
    requested = vehicle.strip().upper()
    if requested not in {"SPY", "USMV", "SPLV", "XLV", "XLP"}:
        raise DataError("sniper vehicle must be SPY or an approved diversified low-vol ETF")
    prepared = _prepare_panel(panel, through=bundle.session)
    bars = _symbol_bars(prepared, requested)
    spy = _symbol_bars(prepared, "SPY")
    stage_1_promoted = any(
        sleeve.sleeve_id == "sniper-stage1-v1"
        for sleeve in (
            *bundle.base_recommendation.promoted_sleeves,
            *bundle.base_recommendation.promoted_compound_sleeves,
        )
    )
    overlays = _overlays(
        prepared,
        session=bundle.session,
        vix_contango=vix_contango,
        put_call_ratio=put_call_ratio,
    )
    primary = _primary_candidate(
        bars,
        spy,
        bundle=bundle,
        account_equity=account_equity,
        max_tolerable_loss=max_tolerable_loss,
        overlays=overlays,
    )
    santa = _santa_candidate(
        spy,
        calendar=calendar,
        bundle=bundle,
        account_equity=account_equity,
        max_tolerable_loss=max_tolerable_loss,
    )
    stage_2 = _stage_2_candidates(stage_1_promoted)
    return SniperPlanV2(
        generated_at=datetime.now(UTC),
        session=bundle.session,
        data_version=bundle.data_version,
        artifact_version=bundle.artifact_version,
        policy_version=bundle.policy_version,
        account_equity=account_equity,
        max_tolerable_loss=max_tolerable_loss,
        requested_vehicle=requested,
        policy_ranking=sniper_policy_ranking(),
        stage_1_candidates=(primary, santa),
        stage_2_candidates=stage_2,
        overlays=overlays,
        excluded_strategy_ids=EXCLUDED_IDS,
        stage_1_promotion_satisfied=stage_1_promoted,
        warnings=(
            "The earlier broad all-stars sniper hypothesis remains rejected; this narrower policy "
            "does not inherit legacy validation claims.",
            "C1/C2 components are merged into one order intent so simultaneous triggers cannot "
            "double-size exposure.",
            "No sniper candidate changes the canonical portfolio until a frozen V2 sleeve passes "
            "all promotion gates.",
        ),
    )


def _prepare_panel(panel: pd.DataFrame, *, through: date) -> pd.DataFrame:
    required = {"symbol", "date", "open", "high", "low", "close", "adj_close", "volume"}
    if missing := required - set(panel.columns):
        raise DataError(f"sniper panel missing {sorted(missing)}")
    output = panel.copy()
    output["symbol"] = output["symbol"].astype(str).str.upper()
    output["date"] = pd.to_datetime(output["date"]).dt.normalize()
    output = output.loc[output["date"] <= pd.Timestamp(through)]
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output.dropna(subset=["open", "high", "low", "close", "adj_close"]).sort_values(
        ["symbol", "date"]
    )


def _symbol_bars(panel: pd.DataFrame, symbol: str) -> pd.DataFrame:
    output = panel.loc[panel["symbol"] == symbol].drop_duplicates("date", keep="last").copy()
    if len(output) < 252:
        raise DataError(f"sniper {symbol} requires at least 252 adjusted daily sessions")
    factor = output["adj_close"] / output["close"]
    if not np.isfinite(factor).all() or (factor <= 0).any():
        raise DataError(f"sniper {symbol} has incompatible adjustment factors")
    output["adj_open"] = output["open"] * factor
    output["adj_low"] = output["low"] * factor
    output["sma200"] = output["adj_close"].rolling(200).mean()
    output["sma5"] = output["adj_close"].rolling(5).mean()
    output["return"] = output["adj_close"].pct_change(fill_method=None)
    output["vol20"] = output["return"].rolling(20).std(ddof=1) * math.sqrt(252)
    output["rsi2"] = wilder_rsi(output["adj_close"], 2)
    output["three_down"] = (
        (output["return"] < 0) & (output["return"].shift(1) < 0) & (output["return"].shift(2) < 0)
    )
    return output.reset_index(drop=True)


def _primary_candidate(
    bars: pd.DataFrame,
    spy: pd.DataFrame,
    *,
    bundle: CanonicalRecommendationBundleV2,
    account_equity: float,
    max_tolerable_loss: float,
    overlays: tuple[SniperOverlayV2, ...],
) -> SniperCandidateV2:
    aligned = bars.merge(
        spy[["date", "adj_close", "sma200", "vol20"]].rename(
            columns={
                "adj_close": "spy_close",
                "sma200": "spy_sma200",
                "vol20": "spy_vol20",
            }
        ),
        on="date",
        how="inner",
    )
    last = aligned.iloc[-1]
    c1 = bool(last["rsi2"] < 5)
    c2 = bool(last["three_down"])
    components = tuple(
        strategy
        for strategy, fired in (
            (SniperStrategyId.C1_RSI2_DIP, c1),
            (SniperStrategyId.C2_THREE_DOWN, c2),
        )
        if fired
    )
    vetoes = []
    if not bool(last["adj_close"] > last["sma200"]):
        vetoes.append("vehicle below or equal to 200-DMA")
    if not bool(last["spy_close"] > last["spy_sma200"]):
        vetoes.append("SPY below or equal to 200-DMA")
    if not bool(last["vol20"] <= 0.20):
        vetoes.append("vehicle 20-session volatility above 20%")
    if not bool(last["spy_vol20"] <= 0.20):
        vetoes.append("SPY 20-session volatility above 20%")
    if pd.Timestamp(last["date"]).month == 9:
        vetoes.append("September stand-aside")
    c3 = next(item for item in overlays if item.strategy_id is SniperStrategyId.C3_VIX_CONTANGO)
    if c3.state is OverlayState.VETO:
        vetoes.append("VIX term structure backwardation veto")

    outcomes = _primary_outcomes(aligned)
    evidence = _outcome_evidence(outcomes)
    adverse = (
        evidence.adverse_move_p05
        if evidence.adverse_move_p05 is not None and evidence.observations >= MIN_RISK_OUTCOMES
        else FALLBACK_ADVERSE_MOVE
    )
    sizing = _size(
        account_equity=account_equity,
        max_tolerable_loss=max_tolerable_loss,
        adverse_move=adverse,
        latest_price=float(last["close"]),
        observations=evidence.observations,
        source=(
            "prior resolved in-regime signal 5th-percentile adverse move"
            if evidence.observations >= MIN_RISK_OUTCOMES
            else "conservative -4% fallback; fewer than 30 resolved signals"
        ),
    )
    triggered = bool(components)
    status = (
        SniperCandidateStatus.VETOED
        if triggered and vetoes
        else SniperCandidateStatus.TRIGGERED_SHADOW
        if triggered
        else SniperCandidateStatus.NOT_TRIGGERED
    )
    return SniperCandidateV2(
        strategy_id=SniperStrategyId.C1_C2_PRIMARY,
        component_triggers=components,
        symbol=str(last["symbol"]),
        status=status,
        signal_session=pd.Timestamp(last["date"]).date(),
        entry_window=(
            f"Next regular-session open at {bundle.execution_at.isoformat()}"
            if triggered and not vetoes
            else None
        ),
        exit_rule="Exit at first close above the then-known 5-DMA, otherwise fourth-session close.",
        maximum_holding_sessions=4,
        sizing=sizing if triggered else None,
        evidence=evidence,
        veto_reasons=tuple(vetoes),
        cautions=(
            "No stop is inferred from daily history; sizing caps modeled loss but gaps can "
            "exceed it.",
            "C4 confirmation may size down/up only after promotion and can never initiate.",
        ),
    )


def _primary_outcomes(aligned: pd.DataFrame) -> tuple[_SignalOutcome, ...]:
    output = []
    next_eligible = 0
    for index in range(200, len(aligned) - 5):
        if index < next_eligible:
            continue
        row = aligned.iloc[index]
        trigger = bool(row["rsi2"] < 5 or row["three_down"])
        eligible = bool(
            row["adj_close"] > row["sma200"]
            and row["spy_close"] > row["spy_sma200"]
            and row["vol20"] <= 0.20
            and row["spy_vol20"] <= 0.20
            and pd.Timestamp(row["date"]).month != 9
        )
        if not trigger or not eligible:
            continue
        entry_index = index + 1
        entry = float(aligned.iloc[entry_index]["adj_open"])
        exit_index = index + 4
        for candidate in range(entry_index, index + 5):
            if float(aligned.iloc[candidate]["adj_close"]) > float(aligned.iloc[candidate]["sma5"]):
                exit_index = candidate
                break
        holding = aligned.iloc[entry_index : exit_index + 1]
        exit_price = float(aligned.iloc[exit_index]["adj_close"])
        adverse = float(holding["adj_low"].min() / entry - 1)
        output.append(
            _SignalOutcome(
                net_return=exit_price / entry - 1 - ROUND_TRIP_COST, adverse_move=adverse
            )
        )
        next_eligible = exit_index + 1
    return tuple(output)


def _santa_candidate(
    spy: pd.DataFrame,
    *,
    calendar: TradingCalendar,
    bundle: CanonicalRecommendationBundleV2,
    account_equity: float,
    max_tolerable_loss: float,
) -> SniperCandidateV2:
    outcomes = _santa_outcomes(spy)
    evidence = _outcome_evidence(outcomes)
    adverse = (
        evidence.adverse_move_p05
        if evidence.adverse_move_p05 is not None and evidence.observations >= 10
        else FALLBACK_ADVERSE_MOVE
    )
    latest = float(spy.iloc[-1]["close"])
    sizing = _size(
        account_equity=account_equity,
        max_tolerable_loss=max_tolerable_loss,
        adverse_move=adverse,
        latest_price=latest,
        observations=evidence.observations,
        source=(
            "prior Santa-window 5th-percentile adverse move"
            if evidence.observations >= 10
            else "conservative -4% fallback; fewer than 10 completed years"
        ),
    )
    entry, exit_session = _next_santa_window(calendar, bundle.session)
    status = (
        SniperCandidateStatus.TRIGGERED_SHADOW
        if bundle.session == entry
        else SniperCandidateStatus.SCHEDULED
    )
    return SniperCandidateV2(
        strategy_id=SniperStrategyId.A1_SANTA,
        symbol="SPY",
        status=status,
        signal_session=bundle.session if status is SniperCandidateStatus.TRIGGERED_SHADOW else None,
        entry_window=f"Planned close of {entry.isoformat()}",
        exit_rule=f"Planned close of second January session {exit_session.isoformat()}.",
        maximum_holding_sessions=7,
        sizing=sizing,
        evidence=evidence,
        cautions=(
            "A close auction requires an execution policy distinct from the default next-open "
            "paper fill.",
            "The fixed window remains shadow-only and is not automatically queued for paper "
            "execution.",
        ),
    )


def _santa_outcomes(spy: pd.DataFrame) -> tuple[_SignalOutcome, ...]:
    output = []
    years = sorted(set(spy["date"].dt.year.astype(int)))
    for year in years:
        december = spy.loc[(spy["date"].dt.year == year) & (spy["date"].dt.month == 12)]
        january = spy.loc[(spy["date"].dt.year == year + 1) & (spy["date"].dt.month == 1)]
        if len(december) < 5 or len(january) < 2:
            continue
        entry_date = pd.Timestamp(december.iloc[-5]["date"])
        exit_date = pd.Timestamp(january.iloc[1]["date"])
        entry = float(december.iloc[-5]["adj_close"])
        exit_price = float(january.iloc[1]["adj_close"])
        holding = spy.loc[(spy["date"] >= entry_date) & (spy["date"] <= exit_date)]
        adverse = float(holding["adj_low"].min() / entry - 1)
        output.append(
            _SignalOutcome(
                net_return=exit_price / entry - 1 - ROUND_TRIP_COST, adverse_move=adverse
            )
        )
    return tuple(output)


def _next_santa_window(calendar: TradingCalendar, session: date) -> tuple[date, date]:
    current = pd.Timestamp(session)
    for year in (current.year, current.year + 1):
        december = calendar.sessions(
            pd.Timestamp(year, 12, 1).date(), pd.Timestamp(year, 12, 31).date()
        )
        january = calendar.sessions(
            pd.Timestamp(year + 1, 1, 1).date(), pd.Timestamp(year + 1, 1, 15).date()
        )
        if len(december) >= 5 and len(january) >= 2:
            entry = december[-5]
            if entry >= current:
                return entry.date(), january[1].date()
    raise DataError("unable to construct the next Santa window")


def _outcome_evidence(outcomes: tuple[_SignalOutcome, ...]) -> SniperOutcomeEvidenceV2:
    if not outcomes:
        return SniperOutcomeEvidenceV2(observations=0, effective_sample_size=0)
    returns = pd.Series([item.net_return for item in outcomes], dtype=float)
    adverse = pd.Series([item.adverse_move for item in outcomes], dtype=float)
    return SniperOutcomeEvidenceV2(
        observations=len(outcomes),
        effective_sample_size=session_effective_sample_size(returns),
        cost_adjusted_win_rate=float((returns > 0).mean()),
        mean_net_return=float(returns.mean()),
        adverse_move_p05=float(min(-1e-6, adverse.quantile(0.05))),
    )


def _size(
    *,
    account_equity: float,
    max_tolerable_loss: float,
    adverse_move: float,
    latest_price: float,
    observations: int,
    source: str,
) -> SniperSizingV2:
    adverse = min(-0.005, adverse_move)
    risk_notional = max_tolerable_loss / abs(adverse)
    capped = min(account_equity, risk_notional)
    return SniperSizingV2(
        account_equity=account_equity,
        max_tolerable_loss=max_tolerable_loss,
        adverse_move_p05=adverse,
        risk_notional=risk_notional,
        capped_notional=capped,
        portfolio_weight=capped / account_equity,
        estimated_shares=math.floor(capped / latest_price),
        cap_applied=capped < risk_notional,
        resolved_signal_outcomes=observations,
        estimate_source=source,
        warning=(
            "The 5th percentile is not a maximum loss. Gaps, slippage, and regime change can "
            "produce a larger loss; no leverage and no live order are permitted."
        ),
    )


def _overlays(
    panel: pd.DataFrame,
    *,
    session: date,
    vix_contango: float | None,
    put_call_ratio: float | None,
) -> tuple[SniperOverlayV2, ...]:
    if vix_contango is None:
        c3 = SniperOverlayV2(
            strategy_id=SniperStrategyId.C3_VIX_CONTANGO,
            state=OverlayState.UNAVAILABLE,
            threshold="front/second VIX future spread >0 confirms; <=0 vetoes",
            effect="No standalone trade; no extra size granted while unavailable.",
            warning="Point-in-time VIX futures term structure is not ingested.",
        )
    else:
        c3 = SniperOverlayV2(
            strategy_id=SniperStrategyId.C3_VIX_CONTANGO,
            state=OverlayState.PASS if vix_contango > 0 else OverlayState.VETO,
            value=vix_contango,
            threshold="positive contango; backwardation veto",
            effect="C1/C2 go/no-go only.",
        )
    breadth = _breadth_value(panel, session)
    if put_call_ratio is None or breadth is None:
        c4 = SniperOverlayV2(
            strategy_id=SniperStrategyId.C4_PUTCALL_BREADTH,
            state=OverlayState.UNAVAILABLE,
            value=breadth,
            threshold="elevated put/call AND breadth washout",
            effect="May confirm and size an existing C1/C2 candidate; cannot initiate.",
            warning=(
                "Compatible put/call data and a broad point-in-time equity universe are absent."
            ),
        )
    else:
        confirmed = put_call_ratio >= 1.0 and breadth <= 0.30
        c4 = SniperOverlayV2(
            strategy_id=SniperStrategyId.C4_PUTCALL_BREADTH,
            state=OverlayState.CONFIRM if confirmed else OverlayState.NEUTRAL,
            value=breadth,
            threshold="put/call >=1.0 and <=30% of universe above 200-DMA",
            effect="Confirmation only; never initiates or overrides a veto.",
        )
    a6 = SniperOverlayV2(
        strategy_id=SniperStrategyId.A6_OPEX_TILT,
        state=OverlayState.NEUTRAL,
        threshold="compatible promoted incremental OpEx artifact",
        effect="No tilt applied; marginal decaying observation only.",
        warning="A6 is blocked pending incremental promotion.",
    )
    return c3, c4, a6


def _breadth_value(panel: pd.DataFrame, session: date) -> float | None:
    latest = panel.loc[panel["date"] <= pd.Timestamp(session)].copy()
    symbols = []
    for symbol, frame in latest.groupby("symbol"):
        if len(frame) < 200 or symbol in {"SPY", "TLT", "SHY", "GLD"}:
            continue
        close = frame.sort_values("date")["adj_close"]
        symbols.append(float(close.iloc[-1] > close.tail(200).mean()))
    return float(np.mean(symbols)) if len(symbols) >= 20 else None


def _stage_2_candidates(stage_1_promoted: bool) -> tuple[SniperCandidateV2, ...]:
    locked = (
        "Stage 1 has no compatible promoted sleeve"
        if not stage_1_promoted
        else "required data absent"
    )
    empty = SniperOutcomeEvidenceV2(observations=0, effective_sample_size=0)
    return (
        SniperCandidateV2(
            strategy_id=SniperStrategyId.A2_PRE_FOMC,
            symbol="SPY_OPTIONS",
            status=SniperCandidateStatus.BLOCKED,
            exit_rule="Expiry immediately after the FOMC announcement; premium is maximum loss.",
            maximum_holding_sessions=2,
            evidence=empty,
            veto_reasons=(
                locked,
                "point-in-time FOMC calendar, options chain, IV rank, spreads, and fills absent",
            ),
            cautions=("Options can expire worthless; do not synthesize this trade with shares.",),
        ),
        SniperCandidateV2(
            strategy_id=SniperStrategyId.E1_TLT_MONTH_END,
            symbol="TLT",
            status=SniperCandidateStatus.BLOCKED,
            exit_rule="Exit first trading session of the next month.",
            maximum_holding_sessions=4,
            evidence=empty,
            veto_reasons=(locked,),
            cautions=("Modest sizing only after independent promotion.",),
        ),
        SniperCandidateV2(
            strategy_id=SniperStrategyId.E2_GOLD_JANUARY,
            symbol="GLD",
            status=SniperCandidateStatus.BLOCKED,
            exit_rule="No live exit rule; low-conviction overlapping observation.",
            maximum_holding_sessions=21,
            evidence=empty,
            veto_reasons=("low conviction and no incremental promoted evidence",),
        ),
    )
