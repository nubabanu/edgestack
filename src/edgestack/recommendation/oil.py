"""Broker-aware, paper-only oil decision service.

The service composes the generic instrument analyzer with eToro quote friction,
cross-market context, event vetoes, and normalized leverage stress.  It never
creates orders, quantities, notionals, or canonical portfolio weight.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog, atomic_write_bytes, safe_child_path
from edgestack.exceptions import DataError, ProviderError
from edgestack.recommendation.hashing import stable_hash
from edgestack.recommendation.instrument import analyze_instrument, resolve_instrument
from edgestack.recommendation.instrument_schemas import (
    CalendarResolution,
    InstrumentAnalysisV2,
    NewsEvidenceV2,
    TimingHorizon,
)
from edgestack.recommendation.oil_schemas import (
    OilBrokerProfileV2,
    OilDataFreshnessV2,
    OilDecisionRequestV2,
    OilDecisionSnapshotV2,
    OilDecisionStatus,
    OilEventFlag,
    OilEventVetoV2,
    OilFrictionScenarioV2,
    OilSourceAlignmentV2,
    OilSourceObservationV2,
    OilStressPointV2,
)
from edgestack.recommendation.schemas import CanonicalRecommendationBundleV2
from edgestack.recommendation.service import CanonicalBundleRepository

OIL_DAILY_SYMBOLS = ("CL=F", "USO", "BNO", "XLE", "UUP", "^OVX")
OIL_INTRADAY_SYMBOLS = ("CL=F", "USO")
REQUIRED_OIL_SYMBOLS = ("CL=F", "USO")
NEW_YORK = ZoneInfo("America/New_York")
MAX_QUOTE_AGE = timedelta(minutes=5)
INTRADAY_MIN_SESSIONS = 20


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None:
        raise DataError(f"{field} must include a timezone offset")
    return value.astimezone(UTC)


def _frame_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"").hexdigest()
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = pd.to_datetime(normalized[column], utc=True).astype(str)
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    hashed = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _latest_timestamp(frame: pd.DataFrame, column: str) -> datetime | None:
    if frame.empty or column not in frame:
        return None
    value = pd.to_datetime(frame[column], utc=True, errors="coerce").max()
    if pd.isna(value):
        return None
    return value.to_pydatetime()


def _source_observation(
    symbol: str,
    role: str,
    daily: pd.DataFrame,
    hourly: pd.DataFrame,
    fifteen: pd.DataFrame,
) -> OilSourceObservationV2:
    available = not daily.empty
    pieces = [frame for frame in (daily, hourly, fifteen) if not frame.empty]
    combined_hash = stable_hash([_frame_hash(frame) for frame in pieces]) if pieces else None
    warning = None
    if not available:
        warning = f"{symbol} daily history is unavailable; this source contributes no context."
    elif symbol in OIL_INTRADAY_SYMBOLS and fifteen.empty:
        warning = f"{symbol} has no 15-minute history; intraday conclusions fail closed."
    return OilSourceObservationV2(
        symbol=symbol,
        role=role,
        available=available,
        daily_through=_latest_timestamp(daily, "date"),
        hourly_through=_latest_timestamp(hourly, "timestamp"),
        fifteen_minute_through=_latest_timestamp(fifteen, "timestamp"),
        input_hash=combined_hash,
        warning=warning,
    )


def _rating_for_entry(analysis: InstrumentAnalysisV2):
    return next(
        (
            item
            for item in analysis.chosen_time_ratings
            if item.resolution is CalendarResolution.MINUTE_15 and item.horizon is TimingHorizon.DAY
        ),
        None,
    )


def _friction_scenario(
    name: Literal["LOW", "BASE", "STRESS"],
    cost_bps: float,
    analysis: InstrumentAnalysisV2,
) -> OilFrictionScenarioV2:
    rating = _rating_for_entry(analysis)
    score = rating.score if rating else None
    survives = bool(
        score
        and score.expected_net_return > 0
        and score.lower_95 > 0
        and score.multiple_testing_adjusted_pvalue <= 0.05
        and score.observations >= INTRADAY_MIN_SESSIONS
    )
    warning = (
        "Descriptive slot survives the friction and inference gates but remains unpromoted "
        "and paper-only."
        if survives
        else "Slot fails positive-return, confidence-bound, sample-size, or adjusted-p gate."
    )
    return OilFrictionScenarioV2(
        name=name,
        round_trip_cost_bps=cost_bps,
        matched_slot=rating.matched_slot if rating else None,
        expected_net_return=score.expected_net_return if score else None,
        lower_95=score.lower_95 if score else None,
        multiple_testing_adjusted_pvalue=(
            score.multiple_testing_adjusted_pvalue if score else None
        ),
        observations=score.observations if score else 0,
        survives=survives,
        warning=warning,
    )


def _is_etoro_oil_open(moment: datetime) -> bool:
    current = _aware_utc(moment, field="market time")
    weekday = current.weekday()
    clock = current.time().replace(tzinfo=None)
    if weekday == 5:
        return False
    if weekday == 6:
        return clock >= time(22, 0)
    if weekday == 4 and clock > time(20, 30):
        return False
    return not time(21, 0) <= clock < time(22, 0)


def _near_overnight_cutoff(moment: datetime) -> bool:
    current = _aware_utc(moment, field="market time")
    clock = current.time().replace(tzinfo=None)
    return time(20, 30) <= clock < time(21, 0)


def _in_eia_window(moment: datetime) -> bool:
    local = moment.astimezone(NEW_YORK)
    clock = local.time().replace(tzinfo=None)
    return local.weekday() == 2 and time(10, 15) <= clock <= time(10, 45)


def _daily_at_or_before(frame: pd.DataFrame, session: date) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"]).dt.normalize()
    return output.loc[output["date"] <= pd.Timestamp(session)].sort_values("date")


def _gap_veto(quote_midpoint: float, primary_daily: pd.DataFrame) -> tuple[bool, str, str]:
    if primary_daily.empty or len(primary_daily) < 20:
        return False, "WTI gap unavailable", "primary WTI history"
    bars = primary_daily.dropna(subset=["high", "low", "close"]).sort_values("date")
    if len(bars) < 20:
        return False, "WTI gap unavailable", "primary WTI history"
    last_close = float(bars["close"].iloc[-1])
    if last_close <= 0:
        return False, "WTI gap unavailable", "primary WTI history"
    ratio = quote_midpoint / last_close
    if not 0.75 <= ratio <= 1.25:
        return False, "WTI/eToro prices are not directly comparable", "derived price check"
    previous = bars["close"].shift(1)
    true_range = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - previous).abs(),
            (bars["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_fraction = float(true_range.tail(14).mean() / last_close)
    gap_fraction = quote_midpoint / last_close - 1.0
    threshold = max(0.02, 0.5 * atr_fraction)
    active = abs(gap_fraction) > threshold
    label = (
        f"eToro midpoint gap {gap_fraction:+.2%} versus WTI prior close; "
        f"adaptive threshold {threshold:.2%}"
    )
    return active, label, "manual quote versus CL=F"


def _event_vetoes(
    request: OilDecisionRequestV2,
    *,
    quote_fresh: bool,
    canonical_matches_catalog: bool,
    required_sources_present: bool,
    required_sources_fresh: bool,
    intraday_sufficient: bool,
    gap: tuple[bool, str, str],
) -> tuple[OilEventVetoV2, ...]:
    items = [
        OilEventVetoV2(
            code="DATA_VERSION_MISMATCH",
            label="Catalog data changed after canonical publication",
            active=not canonical_matches_catalog,
            hard_veto=True,
            source="canonical and catalog hashes",
        ),
        OilEventVetoV2(
            code="STALE_QUOTE",
            label="Manual eToro quote is older than five minutes",
            active=not quote_fresh,
            hard_veto=True,
            source="manual quote timestamp",
        ),
        OilEventVetoV2(
            code="MISSING_REQUIRED_SOURCE",
            label="Primary WTI or USO proxy daily data is unavailable",
            active=not required_sources_present,
            hard_veto=True,
            source="local data catalog",
        ),
        OilEventVetoV2(
            code="STALE_REQUIRED_SOURCE",
            label="Primary WTI or USO proxy daily data is over five calendar days stale",
            active=not required_sources_fresh,
            hard_veto=True,
            source="local data catalog",
        ),
        OilEventVetoV2(
            code="INSUFFICIENT_INTRADAY_HISTORY",
            label="USO 15-minute history has fewer than 20 compatible sessions",
            active=not intraday_sufficient,
            hard_veto=True,
            source="local 15-minute catalog",
        ),
        OilEventVetoV2(
            code="ETORO_MARKET_CLOSED",
            label="Intended entry is outside the configured eToro OIL session",
            active=not _is_etoro_oil_open(request.intended_entry_at),
            hard_veto=True,
            source="eToro OIL broker profile",
        ),
        OilEventVetoV2(
            code="OVERNIGHT_CUTOFF",
            label="Intended entry is within 30 minutes of the overnight cutoff",
            active=_near_overnight_cutoff(request.intended_entry_at),
            hard_veto=True,
            source="eToro OIL broker profile",
        ),
        OilEventVetoV2(
            code="EIA_RELEASE_WINDOW",
            label="Intended entry is inside the Wednesday 10:30 ET petroleum-release window",
            active=_in_eia_window(request.intended_entry_at),
            hard_veto=True,
            source="EIA standard release schedule",
        ),
        OilEventVetoV2(
            code="ABNORMAL_GAP",
            label=gap[1],
            active=gap[0],
            hard_veto=True,
            source=gap[2],
        ),
    ]
    manual_labels = {
        OilEventFlag.WEEKEND_SUPPLY_ESCALATION: "Weekend supply escalation reported",
        OilEventFlag.SHIPPING_DISRUPTION: "Oil-shipping disruption reported",
        OilEventFlag.WTI_ROLLOVER_EXPIRY: "WTI rollover or expiry risk is active",
        OilEventFlag.BROKER_MAINTENANCE: "Broker maintenance or deficient liquidity reported",
    }
    for flag in OilEventFlag:
        items.append(
            OilEventVetoV2(
                code=flag.value,
                label=manual_labels[flag],
                active=flag in request.event_flags,
                hard_veto=True,
                source="manual event confirmation",
            )
        )
    return tuple(items)


def _five_day_return(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    values = pd.to_numeric(frame.sort_values("date")["close"], errors="coerce").dropna()
    if len(values) < 6 or values.iloc[-6] <= 0:
        return None
    return float(values.iloc[-1] / values.iloc[-6] - 1.0)


def _intraday_context(symbol: str, frame: pd.DataFrame) -> tuple[str, ...]:
    """Describe the latest session without turning the description into a signal."""
    if frame.empty or "timestamp" not in frame or "close" not in frame:
        return ()
    ordered = frame.sort_values("timestamp").copy()
    timestamps = pd.to_datetime(ordered["timestamp"], utc=True, errors="coerce")
    ordered = ordered.loc[timestamps.notna()].copy()
    if ordered.empty:
        return ()
    ordered["local_session"] = timestamps.loc[timestamps.notna()].dt.tz_convert(NEW_YORK).dt.date
    session = ordered[ordered["local_session"] == ordered["local_session"].iloc[-1]]
    closes = pd.to_numeric(session["close"], errors="coerce")
    if closes.dropna().empty:
        return ()

    observations = []
    latest = float(closes.dropna().iloc[-1])
    if "volume" in session:
        volumes = pd.to_numeric(session["volume"], errors="coerce").fillna(0.0)
        valid = closes.notna() & (volumes > 0)
        if valid.any():
            vwap = float((closes[valid] * volumes[valid]).sum() / volumes[valid].sum())
            relation = "above" if latest > vwap else "below" if latest < vwap else "at"
            observations.append(
                f"{symbol}: latest close is {relation} latest-session VWAP (descriptive only)"
            )

    opening = session.head(4)
    if len(opening) == 4 and {"high", "low"}.issubset(opening.columns):
        opening_high = pd.to_numeric(opening["high"], errors="coerce").max()
        opening_low = pd.to_numeric(opening["low"], errors="coerce").min()
        if pd.notna(opening_high) and pd.notna(opening_low):
            if latest > float(opening_high):
                relation = "above"
            elif latest < float(opening_low):
                relation = "below"
            else:
                relation = "inside"
            observations.append(
                f"{symbol}: latest close is {relation} the first-hour range (descriptive only)"
            )
    return tuple(observations)


def _source_alignment(
    daily: dict[str, pd.DataFrame], fifteen_minute: dict[str, pd.DataFrame]
) -> OilSourceAlignmentV2:
    observations = []
    returns: dict[str, float] = {}
    for symbol in OIL_DAILY_SYMBOLS:
        value = _five_day_return(daily.get(symbol, pd.DataFrame()))
        if value is None:
            observations.append(f"{symbol}: unavailable")
        else:
            returns[symbol] = value
            observations.append(f"{symbol}: five-session return {value:+.2%}")
    oil_symbols = [symbol for symbol in ("CL=F", "USO", "BNO", "XLE") if symbol in returns]
    if not {"CL=F", "USO"}.issubset(returns):
        state: Literal["ALIGNED", "MIXED", "UNAVAILABLE"] = "UNAVAILABLE"
    else:
        signs = [np.sign(returns[symbol]) for symbol in oil_symbols]
        state = "ALIGNED" if signs and len(set(signs)) == 1 else "MIXED"
    for symbol in REQUIRED_OIL_SYMBOLS:
        observations.extend(_intraday_context(symbol, fifteen_minute.get(symbol, pd.DataFrame())))
    return OilSourceAlignmentV2(state=state, observations=tuple(observations))


def _stress_table() -> tuple[OilStressPointV2, ...]:
    output = []
    for leverage in (1, 5, 10):
        for adverse in (0.01, 0.05, 0.1):
            loss = leverage * adverse
            output.append(
                OilStressPointV2(
                    leverage=leverage,
                    adverse_move_fraction=adverse,
                    equity_loss_fraction=loss,
                    catastrophic=loss >= 0.50,
                    liquidation_possible=loss >= 1.0,
                    warning=(
                        "A stop may fill worse or liquidation may occur earlier because of gaps, "
                        "maintenance margin, spread widening, and broker rules."
                    ),
                )
            )
    return tuple(output)


def build_oil_decision(
    *,
    bundle: CanonicalRecommendationBundleV2,
    request: OilDecisionRequestV2,
    evaluated_at: datetime,
    catalog_data_version: str,
    daily: dict[str, pd.DataFrame],
    hourly: dict[str, pd.DataFrame],
    fifteen_minute: dict[str, pd.DataFrame],
    timing_artifacts=(),
    news: tuple[NewsEvidenceV2, ...] = (),
) -> OilDecisionSnapshotV2:
    """Build one immutable paper-only oil decision from already loaded frames."""
    generated = _aware_utc(evaluated_at, field="evaluated_at")
    quote_time = _aware_utc(request.quote.observed_at, field="quote observed_at")
    quote_age = (generated - quote_time).total_seconds()
    if quote_age < -60:
        raise DataError("manual oil quote cannot be more than one minute in the future")
    quote_age = max(0.0, quote_age)
    quote_fresh = quote_age <= MAX_QUOTE_AGE.total_seconds()

    proxy_daily = daily.get("USO", pd.DataFrame())
    if proxy_daily.empty:
        raise DataError("USO daily history is required for the embedded oil analysis")
    proxy_hourly = hourly.get("USO", pd.DataFrame())
    proxy_fifteen = fifteen_minute.get("USO", pd.DataFrame())
    scenario_costs: tuple[tuple[Literal["LOW", "BASE", "STRESS"], float], ...] = (
        ("LOW", max(request.quote.spread_bps, 10.0)),
        ("BASE", max(request.quote.spread_bps, 25.0)),
        ("STRESS", max(request.quote.spread_bps, 50.0)),
    )
    analyses = []
    scenarios = []
    for name, cost in scenario_costs:
        analysis = analyze_instrument(
            bundle=bundle,
            resolution=resolve_instrument("OIL", canonical=bundle),
            daily_bars=proxy_daily,
            intended_entry_at=request.intended_entry_at,
            intraday_bars=proxy_hourly,
            fifteen_minute_bars=proxy_fifteen,
            timing_artifacts=tuple(timing_artifacts),
            news=news if request.include_news else (),
            round_trip_cost_bps=cost,
        )
        analyses.append(analysis)
        scenarios.append(_friction_scenario(name, cost, analysis))
    base_analysis = analyses[1]

    roles = {
        "CL=F": "primary WTI futures context",
        "USO": "tradeable historical proxy",
        "BNO": "Brent confirmation context",
        "XLE": "energy-equity confirmation context",
        "UUP": "US-dollar risk context",
        "^OVX": "oil-volatility context",
    }
    sources = tuple(
        _source_observation(
            symbol,
            roles[symbol],
            daily.get(symbol, pd.DataFrame()),
            hourly.get(symbol, pd.DataFrame()),
            fifteen_minute.get(symbol, pd.DataFrame()),
        )
        for symbol in OIL_DAILY_SYMBOLS
    )
    required_present = all(
        not daily.get(symbol, pd.DataFrame()).empty for symbol in REQUIRED_OIL_SYMBOLS
    )
    minimum_required_date = pd.Timestamp(bundle.session - timedelta(days=5))
    required_fresh = required_present and all(
        pd.to_datetime(daily[symbol]["date"]).max() >= minimum_required_date
        for symbol in REQUIRED_OIL_SYMBOLS
    )
    intraday_sessions = 0
    if not proxy_fifteen.empty and "timestamp" in proxy_fifteen:
        local = pd.to_datetime(proxy_fifteen["timestamp"], utc=True).dt.tz_convert(NEW_YORK)
        intraday_sessions = int(local.dt.date.nunique())
    intraday_sufficient = intraday_sessions >= INTRADAY_MIN_SESSIONS
    canonical_matches = catalog_data_version == bundle.data_version
    freshness_warnings = [item.warning for item in sources if item.warning]
    if not canonical_matches:
        freshness_warnings.append(
            "Catalog data changed after the canonical publication; republish before relying on it."
        )
    freshness_warnings.append(
        "Yahoo 15-minute history is limited to 59 calendar days and cannot promote an edge."
    )
    freshness = OilDataFreshnessV2(
        canonical_matches_catalog=canonical_matches,
        bundle_as_of=bundle.as_of,
        quote_age_seconds=quote_age,
        quote_fresh=quote_fresh,
        all_required_sources_present=required_present,
        required_sources_fresh=required_fresh,
        sources=sources,
        warnings=tuple(freshness_warnings),
    )
    primary = _daily_at_or_before(daily.get("CL=F", pd.DataFrame()), bundle.session)
    gap = _gap_veto(request.quote.midpoint, primary)
    vetoes = _event_vetoes(
        request,
        quote_fresh=quote_fresh,
        canonical_matches_catalog=canonical_matches,
        required_sources_present=required_present,
        required_sources_fresh=required_fresh,
        intraday_sufficient=intraday_sufficient,
        gap=gap,
    )
    hard_blocks = tuple(item.label for item in vetoes if item.active and item.hard_veto)
    all_scenarios_survive = all(item.survives for item in scenarios)
    if hard_blocks:
        status = OilDecisionStatus.BLOCKED
        reasons = hard_blocks
    elif all_scenarios_survive:
        status = OilDecisionStatus.PAPER_ONLY
        reasons = (
            "Every descriptive friction scenario passed, but no promoted artifact exists; "
            "record a prospective shadow observation only.",
        )
    else:
        status = OilDecisionStatus.OBSERVE
        reasons = ("At least one cost or inference scenario failed; no paper entry is supported.",)

    next_recheck = generated + timedelta(minutes=15)
    payload = {
        "generated_at": generated,
        "broker": "ETORO",
        "broker_symbol": "OIL",
        "product_description": "eToro OIL non-expiring WTI crude-oil CFD",
        "broker_profile": OilBrokerProfileV2(),
        "canonical_bundle_hash": bundle.bundle_hash,
        "canonical_portfolio_weight": 0.0,
        "actionable": False,
        "status": status,
        "intended_entry_at": request.intended_entry_at,
        "quote": request.quote,
        "quote_input_hash": stable_hash(request.quote.model_dump(mode="json")),
        "modeled_leverage_cap": request.modeled_leverage,
        "decision_reasons": reasons,
        "hard_block_reasons": hard_blocks,
        "analysis": base_analysis,
        "data_freshness": freshness,
        "friction_sensitivity": tuple(scenarios),
        "event_vetoes": vetoes,
        "source_alignment": _source_alignment(
            {symbol: _daily_at_or_before(frame, bundle.session) for symbol, frame in daily.items()},
            fifteen_minute,
        ),
        "stress_table": _stress_table(),
        "next_recheck_at": next_recheck,
        "manual_inputs_required": (
            "Confirm weekend supply and shipping headlines.",
            "Confirm the broker's current rollover notice and displayed spread.",
        ),
        "warnings": (
            "OIL is a broker CFD; CL=F and USO can diverge because of rolls, tracking, and hours.",
            "The requested 100% loss case is catastrophic stress only and is never "
            "permitted sizing.",
            "Available leverage is modeled, not recommended, and cannot create actionability.",
        ),
    }
    snapshot_id = stable_hash(
        OilDecisionSnapshotV2.model_validate({"snapshot_id": "0" * 64, **payload}).model_dump(
            mode="json", exclude={"snapshot_id"}
        )
    )
    return OilDecisionSnapshotV2.model_validate({"snapshot_id": snapshot_id, **payload})


def _load_symbol_frames(
    catalog: DataCatalog, bundle: CanonicalRecommendationBundleV2
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    available = set(catalog.list_symbols())
    wanted = tuple(symbol for symbol in OIL_DAILY_SYMBOLS if symbol in available)
    if "USO" not in wanted:
        raise DataError("USO is missing from the catalog; run `edgestack oil refresh`")
    with catalog.guard.unlock(reason="paper-only broker-aware oil decision") as key:
        panel = catalog.load_panel(symbols=wanted, unlock_key=key)
    daily = {
        symbol: panel.loc[panel["symbol"] == symbol].reset_index(drop=True)
        for symbol in OIL_DAILY_SYMBOLS
    }
    hourly = {
        symbol: catalog.load_intraday_bars(symbol, interval_minutes=60, end=bundle.as_of)
        for symbol in OIL_DAILY_SYMBOLS
    }
    fifteen = {
        symbol: catalog.load_intraday_bars(symbol, interval_minutes=15, end=bundle.as_of)
        for symbol in OIL_DAILY_SYMBOLS
    }
    return daily, hourly, fifteen


def build_oil_decision_from_catalog(
    cfg: EdgeStackConfig,
    request: OilDecisionRequestV2,
    *,
    evaluated_at: datetime | None = None,
) -> OilDecisionSnapshotV2:
    catalog = DataCatalog(cfg)
    repository = CanonicalBundleRepository(Path(cfg.paths.artifacts_dir))
    bundle = repository.latest()
    daily, hourly, fifteen = _load_symbol_frames(catalog, bundle)
    return build_oil_decision(
        bundle=bundle,
        request=request,
        evaluated_at=evaluated_at or datetime.now(UTC),
        catalog_data_version=catalog.data_manifest_hash(),
        daily=daily,
        hourly=hourly,
        fifteen_minute=fifteen,
        timing_artifacts=repository.timing_artifacts(),
        news=repository.news_evidence("USO"),
    )


def persist_oil_snapshot(artifacts_dir: Path, snapshot: OilDecisionSnapshotV2) -> Path:
    root = Path(artifacts_dir) / "oil_decisions" / "snapshots"
    path = safe_child_path(root, f"{snapshot.snapshot_id}.json")
    payload = snapshot.model_dump_json(indent=2).encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise DataError("content-addressed oil snapshot already exists with different bytes")
        return path
    atomic_write_bytes(path, payload)
    return path


def refresh_oil_data(cfg: EdgeStackConfig, *, through: date) -> None:
    """Refresh free daily/hourly/15-minute oil inputs within Yahoo limits."""
    from edgestack.pipelines import run_data_download, run_intraday_download

    run_data_download(
        cfg,
        date(2015, 1, 2),
        through,
        provider="yahoo",
        symbols=OIL_DAILY_SYMBOLS,
    )
    windows = (("60m", through - timedelta(days=728)), ("15m", through - timedelta(days=58)))
    for interval, start in windows:
        for symbol in OIL_INTRADAY_SYMBOLS:
            try:
                run_intraday_download(
                    cfg,
                    start,
                    through,
                    symbols=(symbol,),
                    provider="yahoo",
                    interval=interval,
                )
            except ProviderError as exc:
                if symbol in REQUIRED_OIL_SYMBOLS:
                    raise
                print(f"optional oil source {symbol} unavailable: {exc}")
    print(
        "oil data refreshed; canonical data hash is now stale until "
        "`python scripts/nightly.py --config configs/live.yaml --skip-data-update` completes"
    )
