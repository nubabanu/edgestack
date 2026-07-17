"""Multi-resolution choice ratings, tailwind calendars, exits, and rechecks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from scipy import stats

from edgestack.exceptions import DataError
from edgestack.recommendation.instrument_schemas import (
    CalendarResolution,
    ChoiceRating,
    ChosenTimeRatingV2,
    ExitPlanV2,
    FrozenTimingArtifactV2,
    InstrumentAnalysisV2,
    InstrumentRecheckV2,
    RecheckPlanV2,
    TailwindCalendarCellV2,
    TailwindCalendarV2,
    TimingHorizon,
    WinScoreV2,
)
from edgestack.recommendation.schemas import EvidenceGrade
from edgestack.validation.clustered import session_effective_sample_size

MARKET_ZONE = ZoneInfo("America/New_York")
_WEEKDAY = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
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
_HOLDING = {
    TimingHorizon.DAY: 1,
    TimingHorizon.WEEK: 5,
    TimingHorizon.MONTH: 21,
    TimingHorizon.YEAR: 252,
}


@dataclass(frozen=True)
class _ScoredSample:
    key: str
    label: str
    entry: str
    exit: str
    values: pd.Series


def build_trade_calendars(
    *,
    daily: pd.DataFrame,
    hourly: pd.DataFrame | None,
    fifteen_minute: pd.DataFrame | None,
    intended_entry_at: datetime | None,
    cost_bps: float,
    timing_artifacts: tuple[FrozenTimingArtifactV2, ...],
    as_of: datetime,
    rate_intraday_choice: bool = True,
) -> tuple[
    tuple[TailwindCalendarV2, ...],
    tuple[ChosenTimeRatingV2, ...],
    tuple[ExitPlanV2, ...],
    RecheckPlanV2,
]:
    calendars: list[TailwindCalendarV2] = []
    if fifteen_minute is not None and not fifteen_minute.empty:
        calendars.append(
            _intraday_entry_calendar(
                fifteen_minute,
                resolution=CalendarResolution.MINUTE_15,
                hold_bars=4,
                cost_bps=cost_bps,
            )
        )
    else:
        calendars.append(_unavailable_calendar(CalendarResolution.MINUTE_15, TimingHorizon.DAY))
    if hourly is not None and not hourly.empty:
        calendars.append(
            _intraday_entry_calendar(
                hourly,
                resolution=CalendarResolution.HOUR,
                hold_bars=1,
                cost_bps=cost_bps,
            )
        )
    else:
        calendars.append(_unavailable_calendar(CalendarResolution.HOUR, TimingHorizon.DAY))

    for horizon in TimingHorizon:
        calendars.append(_weekday_calendar(daily, horizon=horizon, cost_bps=cost_bps))
    calendars.append(_month_bucket_calendar(daily, cost_bps=cost_bps))
    calendars.append(_month_of_year_calendar(daily, cost_bps=cost_bps))
    calendars = list(_globally_adjusted_calendars(calendars))

    intraday_ratings = (
        (
            _rate_choice(
                calendars, CalendarResolution.MINUTE_15, TimingHorizon.DAY, intended_entry_at
            ),
            _rate_choice(calendars, CalendarResolution.HOUR, TimingHorizon.DAY, intended_entry_at),
        )
        if rate_intraday_choice
        else ()
    )
    ratings = (
        *intraday_ratings,
        *(
            _rate_choice(calendars, CalendarResolution.DAY, horizon, intended_entry_at)
            for horizon in TimingHorizon
        ),
        _rate_choice(calendars, CalendarResolution.MONTH, TimingHorizon.MONTH, intended_entry_at),
        _rate_choice(calendars, CalendarResolution.YEAR, TimingHorizon.YEAR, intended_entry_at),
    )
    selected_ratings = tuple(item for item in ratings if item is not None)
    exits = _build_exit_plans(
        hourly=hourly,
        fifteen_minute=fifteen_minute,
        intended=intended_entry_at,
        cost_bps=cost_bps,
        artifacts=timing_artifacts,
    )
    return tuple(calendars), selected_ratings, exits, build_recheck_plan(as_of, intended_entry_at)


def build_recheck_plan(as_of: datetime, intended: datetime | None) -> RecheckPlanV2:
    if intended is None:
        return RecheckPlanV2(enabled=False, reason="No intended entry time was supplied.")
    current = as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)
    target = intended if intended.tzinfo else intended.replace(tzinfo=MARKET_ZONE)
    remaining = target.astimezone(UTC) - current.astimezone(UTC)
    if remaining <= timedelta(0):
        return RecheckPlanV2(
            enabled=False,
            intended_entry_at=target,
            reason="The intended entry time has passed; submit a new future entry time.",
        )
    if remaining > timedelta(days=30):
        cadence = 1_440
        resolution = CalendarResolution.DAY
    elif remaining > timedelta(days=7):
        cadence = 360
        resolution = CalendarResolution.DAY
    elif remaining > timedelta(days=1):
        cadence = 60
        resolution = CalendarResolution.HOUR
    else:
        cadence = 15
        resolution = CalendarResolution.MINUTE_15
    next_check = min(current + timedelta(minutes=cadence), target.astimezone(current.tzinfo))
    return RecheckPlanV2(
        enabled=True,
        intended_entry_at=target,
        next_check_at=next_check,
        cadence_minutes=cadence,
        required_resolution=resolution,
        reason=(
            f"Recheck every {cadence} minutes at this distance from entry; cadence tightens "
            "to 15 minutes during the final day because finer validated data is unavailable."
        ),
    )


def compare_recheck(
    previous: InstrumentAnalysisV2, current: InstrumentAnalysisV2
) -> InstrumentRecheckV2:
    if previous.resolution.resolved_symbol != current.resolution.resolved_symbol:
        raise DataError("recheck symbol differs from the previous analysis")
    changes: list[str] = []
    previous_map = {(item.resolution, item.horizon): item for item in previous.chosen_time_ratings}
    current_map = {(item.resolution, item.horizon): item for item in current.chosen_time_ratings}
    better_emerged = False
    still_holds = True
    for key, old in previous_map.items():
        new = current_map.get(key)
        if new is None:
            changes.append(f"{key[0]}/{key[1]} evidence became unavailable")
            still_holds = False
            continue
        old_score = old.score.win_score if old.score else None
        new_score = new.score.win_score if new.score else None
        if old_score is not None and new_score is not None and abs(new_score - old_score) >= 2:
            changes.append(f"{key[0]}/{key[1]} score {old_score:.1f} → {new_score:.1f}")
        old_alt = old.better_alternative.slot_key if old.better_alternative else None
        new_alt = new.better_alternative.slot_key if new.better_alternative else None
        if new_alt != old_alt:
            changes.append(f"{key[0]}/{key[1]} better alternative {old_alt} → {new_alt}")
            better_emerged = new_alt is not None
        if new.rating in {ChoiceRating.BELOW_AVERAGE, ChoiceRating.WEAK, ChoiceRating.NOT_RATED}:
            still_holds = False
    if previous.alignment.aligned_trade and not current.alignment.aligned_trade:
        changes.append("promoted all-horizons alignment no longer holds")
        still_holds = False
    return InstrumentRecheckV2(
        previous_analysis_id=previous.analysis_id,
        analysis=current,
        recommendation_still_holds=still_holds,
        better_alternative_emerged=better_emerged,
        changes=tuple(changes),
    )


def _prepare_intraday(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "close"}
    if missing := required - set(frame.columns):
        raise DataError(f"intraday calendar bars missing {sorted(missing)}")
    bars = frame.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    local = bars["timestamp"].dt.tz_convert(MARKET_ZONE)
    bars["session"] = local.dt.date
    bars["slot"] = local.dt.strftime("%H:%M")
    bars["open"] = pd.to_numeric(bars["open"], errors="coerce")
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    return bars.dropna(subset=["open", "close"]).sort_values("timestamp").reset_index(drop=True)


def _intraday_entry_calendar(
    frame: pd.DataFrame,
    *,
    resolution: CalendarResolution,
    hold_bars: int,
    cost_bps: float,
) -> TailwindCalendarV2:
    bars = _prepare_intraday(frame)
    future = bars.groupby("session")["open"].shift(-hold_bars) / bars["open"] - 1
    net = future - cost_bps / 10_000
    samples = [
        _ScoredSample(
            key=slot,
            label=f"{slot} New York",
            entry=f"{slot} America/New_York bar open",
            exit=f"{hold_bars} {resolution.value.lower()} bars later",
            values=net.loc[bars["slot"] == slot].dropna(),
        )
        for slot in sorted(set(bars["slot"]))
    ]
    cells = _rank_samples(samples, TimingHorizon.DAY)
    return TailwindCalendarV2(
        resolution=resolution,
        timezone=str(MARKET_ZONE),
        horizon=TimingHorizon.DAY,
        data_start=bars["timestamp"].min().to_pydatetime() if not bars.empty else None,
        data_end=bars["timestamp"].max().to_pydatetime() if not bars.empty else None,
        cells=cells,
        warning=(
            "Historical cost-adjusted slot ranking only. The entire displayed grid is one "
            "searched family and cannot authorize a trade without promotion."
        ),
    )


def _weekday_calendar(
    daily: pd.DataFrame, *, horizon: TimingHorizon, cost_bps: float
) -> TailwindCalendarV2:
    holding = _HOLDING[horizon]
    future = daily["adjusted_open"].shift(-holding) / daily["adjusted_open"] - 1
    net = future - cost_bps / 10_000
    weekday = daily["date"].dt.weekday
    samples = [
        _ScoredSample(
            key=str(key),
            label=_WEEKDAY[key],
            entry=f"{_WEEKDAY[key]} next-open",
            exit=f"Next-open {holding} sessions later",
            values=net.loc[weekday == key].dropna(),
        )
        for key in _WEEKDAY
    ]
    return _daily_calendar(CalendarResolution.DAY, horizon, daily, _rank_samples(samples, horizon))


def _month_bucket_calendar(daily: pd.DataFrame, *, cost_bps: float) -> TailwindCalendarV2:
    horizon = TimingHorizon.MONTH
    holding = _HOLDING[horizon]
    net = daily["adjusted_open"].shift(-holding) / daily["adjusted_open"] - 1
    net -= cost_bps / 10_000
    bucket = ((daily["session_in_month"] - 1) // 5).clip(upper=4)
    samples = [
        _ScoredSample(
            key=str(key),
            label=f"Month sessions {key * 5 + 1}-{(key + 1) * 5}" if key < 4 else "Month-end",
            entry=f"Bucket {key + 1} next-open",
            exit=f"Next-open {holding} sessions later",
            values=net.loc[bucket == key].dropna(),
        )
        for key in range(5)
    ]
    return _daily_calendar(
        CalendarResolution.MONTH, horizon, daily, _rank_samples(samples, horizon)
    )


def _month_of_year_calendar(daily: pd.DataFrame, *, cost_bps: float) -> TailwindCalendarV2:
    horizon = TimingHorizon.YEAR
    holding = _HOLDING[horizon]
    net = daily["adjusted_open"].shift(-holding) / daily["adjusted_open"] - 1
    net -= cost_bps / 10_000
    months = daily["date"].dt.month
    samples = [
        _ScoredSample(
            key=str(key),
            label=_MONTH[key],
            entry=f"{_MONTH[key]} next-open",
            exit=f"Next-open {holding} sessions later",
            values=net.loc[months == key].dropna(),
        )
        for key in _MONTH
    ]
    return _daily_calendar(CalendarResolution.YEAR, horizon, daily, _rank_samples(samples, horizon))


def _daily_calendar(
    resolution: CalendarResolution,
    horizon: TimingHorizon,
    daily: pd.DataFrame,
    cells: tuple[TailwindCalendarCellV2, ...],
) -> TailwindCalendarV2:
    start = pd.Timestamp(daily["date"].min(), tz=UTC).to_pydatetime() if not daily.empty else None
    end = pd.Timestamp(daily["date"].max(), tz=UTC).to_pydatetime() if not daily.empty else None
    return TailwindCalendarV2(
        resolution=resolution,
        timezone=str(MARKET_ZONE),
        horizon=horizon,
        data_start=start,
        data_end=end,
        cells=cells,
        warning="Descriptive adjusted-open calendar after costs; selection is not promoted edge.",
    )


def _rank_samples(
    samples: list[_ScoredSample], horizon: TimingHorizon
) -> tuple[TailwindCalendarCellV2, ...]:
    usable = [item for item in samples if len(item.values) >= 20]
    if not usable:
        return ()
    prelim = [(item, _sample_statistics(item.values, len(usable))) for item in usable]
    prelim.sort(key=lambda pair: (pair[1]["win_score"], pair[1]["mean"]), reverse=True)
    cells = []
    total = len(prelim)
    for rank, (item, values) in enumerate(prelim, start=1):
        score = WinScoreV2(
            net_win_rate=values["win_rate"],
            shrunk_win_rate=values["shrunk_win_rate"],
            win_score=values["win_score"],
            expected_net_return=values["mean"],
            lower_95=values["lower_95"],
            observations=int(values["observations"]),
            effective_sample_size=values["ess"],
            rank=rank,
            candidates_ranked=total,
            multiple_testing_adjusted_pvalue=values["adjusted_pvalue"],
            evidence_grade=EvidenceGrade.INSUFFICIENT,
        )
        cells.append(
            TailwindCalendarCellV2(
                slot_key=item.key,
                display_label=item.label,
                entry_window=item.entry,
                exit_window=item.exit,
                horizon=horizon,
                score=score,
                rank_percentile=1.0 if total == 1 else 1.0 - (rank - 1) / (total - 1),
            )
        )
    return tuple(cells)


def _sample_statistics(values: pd.Series, trials: int) -> dict[str, float]:
    clean = values.dropna().astype(float)
    n = len(clean)
    ess = max(1.0, session_effective_sample_size(clean))
    mean = float(clean.mean())
    se = float(clean.std(ddof=1) / math.sqrt(ess))
    t_stat = mean / se if se > 0 else 0
    raw_p = float(2 * stats.t.sf(abs(t_stat), df=max(1, n - 1)))
    wins = int((clean > 0).sum())
    win_rate = wins / n
    shrunk = (wins + 5) / (n + 10)
    confidence = math.sqrt(ess / (ess + 50))
    score = 100 * (0.5 + (shrunk - 0.5) * confidence)
    return {
        "win_rate": win_rate,
        "shrunk_win_rate": shrunk,
        "win_score": min(100.0, max(0.0, score)),
        "mean": mean,
        "lower_95": mean - 1.96 * se,
        "observations": float(n),
        "ess": ess,
        "adjusted_pvalue": min(1.0, raw_p * trials),
    }


def _rate_choice(
    calendars: list[TailwindCalendarV2],
    resolution: CalendarResolution,
    horizon: TimingHorizon,
    intended: datetime | None,
) -> ChosenTimeRatingV2 | None:
    if intended is None:
        return None
    calendar = next(
        (item for item in calendars if item.resolution is resolution and item.horizon is horizon),
        None,
    )
    target = (
        intended.astimezone(MARKET_ZONE)
        if intended.tzinfo
        else intended.replace(tzinfo=MARKET_ZONE)
    )
    if calendar is None or not calendar.cells:
        return ChosenTimeRatingV2(
            resolution=resolution,
            horizon=horizon,
            requested_time=target,
            rating=ChoiceRating.NOT_RATED,
            recommendation=f"No compatible {resolution.value} sample is available.",
        )
    key = _choice_key(resolution, target)
    selected = _closest_cell(
        calendar.cells,
        key,
        intraday=resolution in {CalendarResolution.MINUTE_15, CalendarResolution.HOUR},
    )
    if selected is None:
        return ChosenTimeRatingV2(
            resolution=resolution,
            horizon=horizon,
            requested_time=target,
            rating=ChoiceRating.NOT_RATED,
            recommendation="The selected time has no compatible historical slot.",
        )
    best = calendar.cells[0]
    better = (
        best
        if best.slot_key != selected.slot_key and best.score.win_score > selected.score.win_score
        else None
    )
    rating = _rating_band(selected.score.win_score, selected.rank_percentile)
    improvement = best.score.win_score - selected.score.win_score if better else None
    if better:
        recommendation = (
            f"Your choice ranks {selected.score.rank}/{selected.score.candidates_ranked}. "
            f"The higher-scoring historical alternative is {better.entry_window}; both remain "
            "research-only unless promoted."
        )
    else:
        recommendation = (
            "Your choice is the highest-ranked compatible historical slot; it remains "
            "research-only unless promoted."
        )
    return ChosenTimeRatingV2(
        resolution=resolution,
        horizon=horizon,
        requested_time=target,
        matched_slot=selected.slot_key,
        rating=rating,
        score=selected.score,
        better_alternative=better,
        score_improvement=improvement,
        recommendation=recommendation,
    )


def _choice_key(resolution: CalendarResolution, intended: datetime) -> str:
    if resolution is CalendarResolution.MINUTE_15:
        minute = (intended.minute // 15) * 15
        return f"{intended.hour:02d}:{minute:02d}"
    if resolution is CalendarResolution.HOUR:
        return f"{intended.hour:02d}:00"
    if resolution is CalendarResolution.DAY:
        return str(intended.weekday())
    if resolution is CalendarResolution.MONTH:
        return str(min(4, (intended.day - 1) // 5))
    return str(intended.month)


def _closest_cell(
    cells: tuple[TailwindCalendarCellV2, ...], key: str, *, intraday: bool
) -> TailwindCalendarCellV2 | None:
    exact = next((item for item in cells if item.slot_key == key), None)
    if exact or not intraday:
        return exact
    requested = _minutes(key)
    return min(cells, key=lambda item: abs(_minutes(item.slot_key) - requested), default=None)


def _minutes(slot: str) -> int:
    hour, minute = slot.split(":", maxsplit=1)
    return int(hour) * 60 + int(minute)


def _rating_band(score: float, percentile: float) -> ChoiceRating:
    if score >= 55 and percentile >= 0.8:
        return ChoiceRating.STRONG
    if score >= 51 and percentile >= 0.6:
        return ChoiceRating.ABOVE_AVERAGE
    if score >= 49:
        return ChoiceRating.AVERAGE
    if score >= 45:
        return ChoiceRating.BELOW_AVERAGE
    return ChoiceRating.WEAK


def _build_exit_plans(
    *,
    hourly: pd.DataFrame | None,
    fifteen_minute: pd.DataFrame | None,
    intended: datetime | None,
    cost_bps: float,
    artifacts: tuple[FrozenTimingArtifactV2, ...],
) -> tuple[ExitPlanV2, ...]:
    if intended is None:
        return ()
    plans = []
    for horizon in TimingHorizon:
        artifact = next((item for item in artifacts if item.horizon is horizon), None)
        frame = fifteen_minute if horizon is TimingHorizon.DAY else hourly
        resolution = (
            CalendarResolution.MINUTE_15
            if horizon is TimingHorizon.DAY
            else CalendarResolution.HOUR
        )
        if frame is None or frame.empty:
            if artifact is not None:
                plans.append(
                    ExitPlanV2(
                        horizon=horizon,
                        entry_slot=intended.astimezone(MARKET_ZONE).strftime("%H:%M"),
                        preferred_exit=artifact.exit_window,
                        holding_sessions=artifact.holding_sessions,
                        data_resolution=resolution,
                        actionable=True,
                        rationale="Frozen promoted exit rule.",
                    )
                )
            else:
                plans.append(
                    ExitPlanV2(
                        horizon=horizon,
                        entry_slot=intended.astimezone(MARKET_ZONE).strftime("%H:%M"),
                        holding_sessions=0 if horizon is TimingHorizon.DAY else _HOLDING[horizon],
                        data_resolution=resolution,
                        rationale="No compatible intraday history.",
                        warning="No exit hour is inferred from daily data.",
                    )
                )
            continue
        plans.append(
            _conditional_exit_plan(
                frame,
                intended=intended,
                horizon=horizon,
                offset=0 if horizon is TimingHorizon.DAY else _HOLDING[horizon],
                resolution=resolution,
                cost_bps=cost_bps,
            )
        )
    return tuple(plans)


def _conditional_exit_plan(
    frame: pd.DataFrame,
    *,
    intended: datetime,
    horizon: TimingHorizon,
    offset: int,
    resolution: CalendarResolution,
    cost_bps: float,
) -> ExitPlanV2:
    bars = _prepare_intraday(frame)
    matrix = bars.pivot_table(index="session", columns="slot", values="open", aggfunc="last")
    slots = sorted(str(item) for item in matrix.columns)
    requested = intended.astimezone(MARKET_ZONE).strftime("%H:%M")
    entry_slot = min(slots, key=lambda item: abs(_minutes(item) - _minutes(requested)))
    weekday = intended.astimezone(MARKET_ZONE).weekday()
    samples = []
    for exit_slot in slots:
        if offset == 0 and _minutes(exit_slot) <= _minutes(entry_slot):
            continue
        entry = matrix[entry_slot]
        exit_price = matrix[exit_slot].shift(-offset)
        values = exit_price / entry - 1 - cost_bps / 10_000
        session_weekday = pd.Series(
            [pd.Timestamp(item).weekday() for item in matrix.index], index=matrix.index
        )
        if offset > 0:
            values = values.loc[session_weekday == weekday]
        values = values.dropna()
        samples.append(
            _ScoredSample(
                key=exit_slot,
                label=f"Exit {exit_slot} New York",
                entry=f"Entry {entry_slot} New York",
                exit=f"{exit_slot} New York after {offset} sessions",
                values=values,
            )
        )
    cells = _rank_samples(samples, horizon)
    if not cells:
        return ExitPlanV2(
            horizon=horizon,
            entry_slot=entry_slot,
            holding_sessions=offset,
            data_resolution=resolution,
            rationale="Fewer than 20 conditional entry/exit observations are available.",
            warning="No exit hour is reported.",
        )
    best = cells[0]
    return ExitPlanV2(
        horizon=horizon,
        entry_slot=entry_slot,
        preferred_exit=best.exit_window,
        holding_sessions=offset,
        data_resolution=resolution,
        score=best.score,
        alternatives=cells[1:3],
        rationale=(
            "Highest cost-adjusted historical exit win score conditional on the matched entry "
            "slot and weekday. It is non-actionable until promoted."
        ),
        warning="Exit-hour search is part of the multiple-testing family and may not persist.",
    )


def _unavailable_calendar(
    resolution: CalendarResolution, horizon: TimingHorizon
) -> TailwindCalendarV2:
    return TailwindCalendarV2(
        resolution=resolution,
        timezone=str(MARKET_ZONE),
        horizon=horizon,
        warning=f"No compatible {resolution.value} history; no calendar or score is inferred.",
    )


def _globally_adjusted_calendars(
    calendars: list[TailwindCalendarV2],
) -> tuple[TailwindCalendarV2, ...]:
    total_cells = sum(len(calendar.cells) for calendar in calendars)
    if total_cells == 0:
        return tuple(calendars)
    output = []
    for calendar in calendars:
        cells = []
        for cell in calendar.cells:
            within_trials = cell.score.candidates_ranked
            global_pvalue = min(
                1.0,
                cell.score.multiple_testing_adjusted_pvalue * total_cells / within_trials,
            )
            cells.append(
                cell.model_copy(
                    update={
                        "score": cell.score.model_copy(
                            update={
                                "multiple_testing_adjusted_pvalue": global_pvalue,
                                "explanation": (
                                    cell.score.explanation
                                    + f" P-value adjusted over {total_cells} displayed cells."
                                ),
                            }
                        )
                    }
                )
            )
        output.append(calendar.model_copy(update={"cells": tuple(cells)}))
    return tuple(output)
