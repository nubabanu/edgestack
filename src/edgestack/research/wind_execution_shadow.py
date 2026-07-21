"""Prospective, unlevered one-session execution-delay shadow for WIND.

This is a paired fill-price experiment, not a portfolio and not an order
generator.  A negative score for session *t* creates two hypothetical fixed-
notional buys: the baseline at *t*'s open and the delayed arm at the next XNYS
session's open.  The delayed arm always fills after one session, regardless of
the next score.
"""

from __future__ import annotations

import copy
from datetime import date
from typing import Any, cast

import numpy as np
import pandas as pd

from edgestack.data.calendar import TradingCalendar
from edgestack.recommendation.hashing import stable_hash
from edgestack.research.market_wind import METHOD_VERSION, WindSnapshotV2

SHADOW_VERSION = "wind-one-session-delay-v1"
FIXED_NOTIONAL = 1_000.0
ONE_WAY_COST = 0.0002
MIN_PROSPECTIVE_SESSIONS = 252
MIN_COMPLETED_EVENTS = 30
SHADOW_MANIFEST = {
    "experiment": SHADOW_VERSION,
    "wind_method_version": METHOD_VERSION,
    "score_condition": "score < 0",
    "baseline_fill": "evaluated XNYS session open",
    "counterfactual_fill": "exactly one following XNYS session open",
    "fixed_notional": FIXED_NOTIONAL,
    "one_way_cost": ONE_WAY_COST,
    "minimum_prospective_sessions": MIN_PROSPECTIVE_SESSIONS,
    "minimum_completed_events": MIN_COMPLETED_EVENTS,
    "promotion_eligible": False,
    "action": "NO_ACTION",
}
SHADOW_MANIFEST_HASH = stable_hash(SHADOW_MANIFEST)


def new_shadow_state(*, registered_on: date) -> dict:
    return {
        "schema_version": 1,
        "experiment": SHADOW_VERSION,
        "wind_method_version": METHOD_VERSION,
        "manifest": copy.deepcopy(SHADOW_MANIFEST),
        "manifest_hash": SHADOW_MANIFEST_HASH,
        "registered_on": registered_on.isoformat(),
        "status": "COLLECTING",
        "promotion_eligible": False,
        "action": "NO_ACTION",
        "fixed_notional": FIXED_NOTIONAL,
        "one_way_cost_bps": ONE_WAY_COST * 10_000,
        "policy": (
            "If the prior-close WIND score for session t is negative, compare a buy at "
            "t open with a buy at the next XNYS session open; fill after exactly one session."
        ),
        "review_gate": {
            "minimum_prospective_sessions": MIN_PROSPECTIVE_SESSIONS,
            "minimum_completed_events": MIN_COMPLETED_EVENTS,
        },
        "observed_sessions": [],
        "events": [],
        "metrics": _metrics([], 0),
    }


def _normalize_execution_bars(frame: pd.DataFrame) -> pd.DataFrame:
    bars = frame.copy()
    if "date" in bars:
        bars = bars.assign(date=pd.to_datetime(bars["date"])).set_index("date")
    if "open" not in bars:
        raise ValueError("execution shadow requires open prices")
    bars.index = pd.DatetimeIndex(bars.index).tz_localize(None).normalize()
    if bars.index.has_duplicates:
        raise ValueError("execution bars contain duplicate sessions")
    bars = bars.sort_index()
    if (bars["open"].dropna() <= 0).any():
        raise ValueError("execution bars contain non-positive opens")
    return bars


def _event_id(as_of: str, immediate_session: str) -> str:
    return stable_hash(
        {
            "experiment": SHADOW_VERSION,
            "wind_method_version": METHOD_VERSION,
            "as_of": as_of,
            "immediate_session": immediate_session,
        }
    )[:24]


def _metrics(events: list[dict], observed_sessions: int) -> dict:
    improvements = np.asarray(
        [event["fill_improvement_bps"] for event in events if event["status"] == "COMPLETED"],
        dtype=float,
    )
    review_eligible = bool(
        observed_sessions >= MIN_PROSPECTIVE_SESSIONS and len(improvements) >= MIN_COMPLETED_EVENTS
    )
    if len(improvements) >= 2:
        rng = np.random.default_rng(20_260_721)
        samples = rng.choice(improvements, size=(4_000, len(improvements)), replace=True).mean(
            axis=1
        )
        ci: list[float | None] = [
            round(float(np.quantile(samples, 0.025)), 2),
            round(float(np.quantile(samples, 0.975)), 2),
        ]
    else:
        ci = [None, None]
    positive = bool(
        review_eligible
        and float(np.mean(improvements)) > 0
        and float(np.median(improvements)) > 0
        and ci[0] is not None
        and ci[0] > 0
    )
    result = "NOT_REVIEWED"
    if review_eligible:
        result = "POSITIVE_DIAGNOSTIC" if positive else "NON_POSITIVE"
    return {
        "prospective_sessions": observed_sessions,
        "completed_events": len(improvements),
        "pending_events": sum(event["status"] == "PENDING" for event in events),
        "mean_fill_improvement_bps": (
            round(float(np.mean(improvements)), 2) if len(improvements) else None
        ),
        "median_fill_improvement_bps": (
            round(float(np.median(improvements)), 2) if len(improvements) else None
        ),
        "mean_ci95_bps": ci,
        "review_eligible": review_eligible,
        "result": result,
        "action": "NO_ACTION",
    }


def advance_shadow(
    state: dict,
    bars: pd.DataFrame,
    snapshot: WindSnapshotV2,
    *,
    calendar: TradingCalendar | None = None,
) -> dict:
    """Settle completed pairs and register at most one new prospective event."""
    if (
        state.get("manifest_hash") != SHADOW_MANIFEST_HASH
        or state.get("manifest") != SHADOW_MANIFEST
    ):
        raise ValueError("execution-shadow manifest mismatch; rules are immutable")
    updated = copy.deepcopy(state)
    prices = _normalize_execution_bars(bars)
    cal = calendar or TradingCalendar("XNYS")

    for event in updated["events"]:
        if event["status"] != "PENDING":
            continue
        immediate = pd.Timestamp(event["immediate_session"])
        delayed = pd.Timestamp(event["delayed_session"])
        if immediate not in prices.index or delayed not in prices.index:
            continue
        immediate_open = float(cast(Any, prices.at[immediate, "open"]))
        delayed_open = float(cast(Any, prices.at[delayed, "open"]))
        immediate_effective = immediate_open * (1 + ONE_WAY_COST)
        delayed_effective = delayed_open * (1 + ONE_WAY_COST)
        event.update(
            {
                "status": "COMPLETED",
                "immediate_open": immediate_open,
                "delayed_open": delayed_open,
                "immediate_shares": round(FIXED_NOTIONAL / immediate_effective, 8),
                "delayed_shares": round(FIXED_NOTIONAL / delayed_effective, 8),
                "fill_improvement_bps": round(
                    (immediate_effective / delayed_effective - 1) * 10_000, 2
                ),
                "settled_through": str(prices.index[-1].date()),
            }
        )

    if snapshot.status == "READY" and snapshot.for_session is not None:
        target = snapshot.for_session.isoformat()
        observed = updated.setdefault("observed_sessions", [])
        if target not in observed:
            observed.append(target)
            observed.sort()
        if snapshot.score is not None and snapshot.score < 0 and snapshot.as_of is not None:
            as_of = snapshot.as_of.isoformat()
            identifier = _event_id(as_of, target)
            existing = {event["event_id"] for event in updated["events"]}
            if identifier not in existing:
                delayed_session = cal.next_session(snapshot.for_session).date().isoformat()
                updated["events"].append(
                    {
                        "event_id": identifier,
                        "status": "PENDING",
                        "as_of": as_of,
                        "score": snapshot.score,
                        "votes": snapshot.votes,
                        "immediate_session": target,
                        "delayed_session": delayed_session,
                    }
                )

    updated["events"].sort(key=lambda event: (event["immediate_session"], event["event_id"]))
    updated["metrics"] = _metrics(updated["events"], len(updated["observed_sessions"]))
    updated["status"] = "REVIEW_ELIGIBLE" if updated["metrics"]["review_eligible"] else "COLLECTING"
    updated["action"] = "NO_ACTION"
    return updated
