"""Nightly adapter for the unlevered WIND execution-delay shadow.

The output is a prospective paired-price research ledger only.  It never
creates an order, position, alert, or promotion-eligible artifact.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from edgestack.data.catalog import atomic_write_bytes
from edgestack.research.market_wind import snapshot_for_next_session
from edgestack.research.wind_execution_shadow import (
    SHADOW_MANIFEST,
    SHADOW_MANIFEST_HASH,
    advance_shadow,
    new_shadow_state,
)

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "data" / "curated" / "prices" / "SPY.parquet"
STATE_PATH = ROOT / "artifacts" / "wind_execution_shadow.json"


def _bars() -> pd.DataFrame:
    frame = pd.read_parquet(PRICES)
    return frame.assign(date=pd.to_datetime(frame["date"])).set_index("date").sort_index()


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return new_shadow_state(registered_on=date.today())
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    # One-time migration for the outcome-free ledger created during the audit.
    if "manifest_hash" not in state:
        if state.get("events"):
            raise ValueError("cannot attach a manifest after shadow events exist")
        state["manifest"] = dict(SHADOW_MANIFEST)
        state["manifest_hash"] = SHADOW_MANIFEST_HASH
    return state


def main() -> int:
    bars = _bars()
    score_bars = pd.DataFrame({"close": bars["close"], "adj_close": bars["adj_close"]})
    snapshot = snapshot_for_next_session(score_bars)
    state = _load_state()
    state = advance_shadow(state, bars, snapshot)
    atomic_write_bytes(STATE_PATH, json.dumps(state, indent=2).encode("utf-8"))
    metrics = state["metrics"]
    print(
        f"wind execution shadow: {state['status']} | "
        f"sessions={metrics['prospective_sessions']} "
        f"completed={metrics['completed_events']} pending={metrics['pending_events']} "
        f"action={state['action']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
