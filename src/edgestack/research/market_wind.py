"""Neutral, exchange-aware implementation of the legacy market WIND score.

The score is a historical research diagnostic.  Its inputs are observable price
and calendar features; their names deliberately avoid unobserved causal claims
about payroll flows, institutional mandates, or dealer inventories.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from edgestack.data.calendar import TradingCalendar

METHOD_VERSION: Literal["legacy-wind-neutral-v2"] = "legacy-wind-neutral-v2"
LEGACY_CUTOFF = date(2026, 7, 15)
COMPONENTS = (
    "turn_of_month",
    "post_down_month",
    "weekday_reversal",
    "trend_vol_regime",
    "short_term_dip",
)


class WindSnapshotV2(BaseModel):
    """Backward-readable, explicitly non-actionable watcher payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = 2
    method_version: Literal["legacy-wind-neutral-v2"] = METHOD_VERSION
    status: Literal["READY", "UNAVAILABLE"]
    as_of: date | None
    for_session: date | None
    score: int | None
    votes: dict[str, int]
    label: Literal["DESCRIPTIVE_ONLY"] = "DESCRIPTIVE_ONLY"
    evidence_status: Literal["HISTORICAL_DIAGNOSTIC_ONLY"] = "HISTORICAL_DIAGNOSTIC_ONLY"
    action: Literal["NO_ACTION"] = "NO_ACTION"
    reasons: tuple[str, ...] = ()
    note: str = "Historical diagnostic only; strict monotonicity and selective-entry rules failed."


def _adj_column(frame: pd.DataFrame) -> str:
    if "adj" in frame:
        return "adj"
    if "adj_close" in frame:
        return "adj_close"
    raise ValueError("bars require an adj or adj_close column")


def normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sorted, unique, tz-naive bars with canonical ``close``/``adj`` columns."""
    if "close" not in frame:
        raise ValueError("bars require a close column")
    adj = _adj_column(frame)
    out = frame[["close", adj]].rename(columns={adj: "adj"}).copy()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    if out.index.has_duplicates:
        raise ValueError("bars contain duplicate sessions")
    return out.sort_index()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def component_votes(
    frame: pd.DataFrame,
    *,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Compute legacy votes indexed by the session being evaluated.

    Calendar inputs describe the evaluated session and are known in advance.
    Every price-derived input is shifted one row, so a session's vote uses data
    available no later than the preceding bar.
    """
    bars = normalize_bars(frame)
    if bars.empty:
        return pd.DataFrame(index=bars.index, columns=COMPONENTS, dtype=float)
    cal = calendar or TradingCalendar("XNYS")
    facts = cal.session_facts(bars.index[0].date(), bars.index[-1].date()).reindex(bars.index)
    if facts[["sessions_since_month_start", "sessions_to_month_end"]].isna().any().any():
        raise ValueError("bars include dates that are not XNYS sessions")

    ret = bars["adj"].pct_change()
    close = bars["close"]
    index = pd.DatetimeIndex(bars.index)
    month = index.to_period("M")

    turn_of_month = (facts["sessions_since_month_start"] <= 2) | (
        facts["sessions_to_month_end"] == 0
    )
    monthly_ret = bars["adj"].groupby(month).last().pct_change()
    post_down_month = pd.Series(
        month.map(lambda period: bool(monthly_ret.get(period - 1, np.nan) < 0)),
        index=bars.index,
    )
    post_down_month &= facts["sessions_since_month_start"] <= 2

    weekday = pd.Series(index.dayofweek, index=index)
    previous_return = ret.shift(1)
    monday_reversal = (weekday == 0) & (previous_return < 0)

    volatility_20 = (ret.rolling(20).std() * np.sqrt(252)).shift(1)
    above_200 = (close > close.rolling(200).mean()).shift(1).fillna(False).astype(bool)
    calm = above_200 & (volatility_20 < 0.30)
    stressed = (~above_200) & (volatility_20 > 0.30)
    stressed_friday = (weekday == 4) & stressed

    rsi_2 = _rsi(close, 2).shift(1)
    down_3 = ((ret < 0) & (ret.shift(1) < 0) & (ret.shift(2) < 0)).shift(1)
    short_term_dip = (rsi_2 < 10) | down_3.fillna(False)

    votes = pd.DataFrame(index=bars.index)
    votes["turn_of_month"] = turn_of_month.astype(float)
    votes["post_down_month"] = post_down_month.astype(float)
    votes["weekday_reversal"] = monday_reversal.astype(float) - stressed_friday.astype(float)
    votes["trend_vol_regime"] = calm.astype(float) - stressed.astype(float)
    votes["short_term_dip"] = short_term_dip.astype(float)
    return pd.DataFrame(votes.loc[:, list(COMPONENTS)])


def snapshot_for_next_session(
    frame: pd.DataFrame,
    *,
    calendar: TradingCalendar | None = None,
    evaluated_on: date | None = None,
    max_session_lag: int = 1,
) -> WindSnapshotV2:
    """Build the next-session watcher snapshot, failing closed on unusable data."""
    cal = calendar or TradingCalendar("XNYS")
    try:
        bars = normalize_bars(frame)
    except (TypeError, ValueError) as exc:
        return WindSnapshotV2(
            status="UNAVAILABLE",
            as_of=None,
            for_session=None,
            score=None,
            votes={},
            reasons=(f"INVALID_INPUT:{exc}",),
        )
    if bars.empty:
        return WindSnapshotV2(
            status="UNAVAILABLE",
            as_of=None,
            for_session=None,
            score=None,
            votes={},
            reasons=("NO_BARS",),
        )

    as_of = bars.index[-1].date()
    target = cal.next_session(as_of).date()
    reasons: list[str] = []
    if len(bars.dropna()) < 201:
        reasons.append("INSUFFICIENT_HISTORY")
    check_date = evaluated_on or date.today()
    if check_date > as_of:
        lag = cal.sessions_between(as_of, check_date)
        if lag > max_session_lag:
            reasons.append(f"STALE_INPUT:{lag}_SESSIONS")
    if reasons:
        return WindSnapshotV2(
            status="UNAVAILABLE",
            as_of=as_of,
            for_session=target,
            score=None,
            votes={},
            reasons=tuple(reasons),
        )

    extended = pd.concat(
        [bars, pd.DataFrame(index=pd.DatetimeIndex([pd.Timestamp(target)]), columns=bars.columns)]
    )
    try:
        row = component_votes(extended, calendar=cal).iloc[-1]
    except (TypeError, ValueError) as exc:
        return WindSnapshotV2(
            status="UNAVAILABLE",
            as_of=as_of,
            for_session=target,
            score=None,
            votes={},
            reasons=(f"CALCULATION_FAILED:{exc}",),
        )
    votes = {name: int(float(cast(Any, row[name]))) for name in COMPONENTS}
    return WindSnapshotV2(
        status="READY",
        as_of=as_of,
        for_session=target,
        score=sum(votes.values()),
        votes=votes,
    )


def bars_fingerprint(frame: pd.DataFrame, *, cutoff: date = LEGACY_CUTOFF) -> str:
    """Content hash of the exact canonical input visible through ``cutoff``."""
    bars = normalize_bars(frame).loc[: cutoff.isoformat()]
    digest = hashlib.sha256()
    digest.update(b"close,adj\n")
    digest.update(pd.util.hash_pandas_object(bars, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def score_bucket_report(score: pd.Series, returns: pd.Series) -> dict:
    """Describe every exact score bucket and report strict mean monotonicity."""
    aligned = pd.concat([score.rename("score"), returns.rename("return")], axis=1).dropna()
    grouped = aligned.groupby(aligned["score"].astype(int))["return"].agg(["mean", "count"])
    means = grouped["mean"]
    buckets = {}
    for value, row in grouped.iterrows():
        bucket = int(cast(Any, value))
        buckets[str(bucket)] = {
            "mean_bps": round(float(row["mean"]) * 10_000, 1),
            "n": int(row["count"]),
        }
    return {
        "strict_monotonic": bool(len(means) > 1 and (means.diff().dropna() > 0).all()),
        "buckets": buckets,
    }
