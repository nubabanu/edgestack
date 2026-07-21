"""Retirement handler for the two former leveraged WIND paper experiments.

No new exposure is ever created.  A position that was already pending when
the experiments were retired is settled once against the first available real
bar.  The final state is then marked RETIRED and copied to a content-addressed
archive.  Keeping this idempotent handler in the nightly chain preserves the
historical record without silently abandoning the last committed paper fill.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from edgestack.data.catalog import atomic_write_bytes

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "data" / "curated" / "prices"
STATE_PATH = ROOT / "artifacts" / "wind_paper_book.json"
ARCHIVE_DIR = ROOT / "artifacts" / "retired_wind_paper"
RETIREMENT_REASON = (
    "Selective WIND entry failed validation; leveraged paper experiments retired by audit."
)

SAILOR = {
    "lev": 3.0,
    "stop_raw": -0.02,
    "spread_rt": 0.0002,
    "financing_annual": 0.04,
    "start_equity": 10_000.0,
}
BARRIER = {
    "lev": 20.0,
    "closeout_raw": -0.025,
    "spread_rt": 0.0004,
    "overnight_annual": 0.069,
    "stake": 5_000.0,
}


def _load_state() -> dict:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    else:
        state = {
            "registered": "2026-07-21",
            "note": "PAPER ONLY. Retired before any leveraged experiment completed.",
            "sailor_3x": {"equity": SAILOR["start_equity"], "pending": None, "trades": []},
            "barrier_20x": {
                "equity": BARRIER["stake"],
                "pending": None,
                "trades": [],
            },
        }
    state["schema_version"] = 2
    state["retirement_reason"] = RETIREMENT_REASON
    for name in ("sailor_3x", "barrier_20x"):
        state[name].setdefault("trades", [])
        state[name].setdefault("pending", None)
        if state[name]["pending"] is not None:
            state[name]["status"] = "RETIRING_PENDING_SETTLEMENT"
        elif state[name].get("status") not in {"RETIRED"}:
            state[name]["status"] = "RETIRED"
    return state


def _bars() -> pd.DataFrame:
    df = pd.read_parquet(PRICES / "SPY.parquet")
    return df.assign(date=pd.to_datetime(df["date"])).set_index("date").sort_index()


def _settle(bars: pd.DataFrame, decided: str) -> dict | None:
    """Raw session outcomes for the first session AFTER the decision close."""
    later = bars.index[bars.index > pd.Timestamp(decided)]
    if len(later) == 0:
        return None  # session not in catalog yet; keep pending
    t = later[0]
    prev = bars.index[bars.index < t][-1]
    prev_close, prev_adj = float(bars.loc[prev, "close"]), float(bars.loc[prev, "adj_close"])
    row = bars.loc[t]
    return {
        "session": str(t.date()),
        "cc": float(row["adj_close"]) / prev_adj - 1.0,
        "overnight": float(row["open"]) / prev_close - 1.0,
        "low": float(row["low"]) / prev_close - 1.0,
    }


def _sailor_net(s: dict) -> float:
    lev, stop = SAILOR["lev"], SAILOR["stop_raw"]
    if s["overnight"] <= stop:
        raw = s["overnight"]
    elif s["low"] <= stop:
        raw = stop
    else:
        raw = s["cc"]
    return lev * raw - lev * SAILOR["spread_rt"] - (lev - 1) * SAILOR["financing_annual"] / 252


def _barrier_net(s: dict) -> float:
    lev, thr = BARRIER["lev"], BARRIER["closeout_raw"]
    cost = lev * BARRIER["spread_rt"] + (lev - 1) * BARRIER["overnight_annual"] / 365
    if s["overnight"] <= thr:
        return max(lev * s["overnight"] - cost, -1.0)
    if s["low"] <= thr:
        return -0.5
    return lev * s["cc"] - cost


def _archive_retired_state(state: dict) -> str:
    """Write one immutable content-addressed copy and return its hash."""
    payload = {key: value for key, value in state.items() if key != "archive_hash"}
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    content_hash = hashlib.sha256(encoded).hexdigest()
    path = ARCHIVE_DIR / f"{content_hash}.json"
    if not path.exists():
        atomic_write_bytes(path, encoded)
    return content_hash


def main() -> int:
    bars = _bars()
    state = _load_state()
    lines: list[str] = []

    # Settle only positions committed before retirement. Never open another.
    for name, net_fn in (("sailor_3x", _sailor_net), ("barrier_20x", _barrier_net)):
        exp = state[name]
        pending = exp.get("pending")
        if pending:
            settled = _settle(bars, pending["decided_at"])
            if settled is not None and settled["session"] > pending["decided_at"]:
                net = net_fn(settled)
                exp["equity"] = round(exp["equity"] * (1.0 + net), 2)
                exp["trades"].append(
                    {**settled, "net": round(net, 5), "equity_after": exp["equity"]}
                )
                exp["pending"] = None
                lines.append(
                    f"{name}: settled {settled['session']} net {net:+.2%} "
                    f"-> equity {exp['equity']:,.2f}"
                )

        if exp["pending"] is None:
            exp["status"] = "RETIRED"
            exp.setdefault("retired_at", date.today().isoformat())
        else:
            exp["status"] = "RETIRING_PENDING_SETTLEMENT"

    if all(state[name]["status"] == "RETIRED" for name in ("sailor_3x", "barrier_20x")):
        state["status"] = "RETIRED"
        state.setdefault("retired_at", date.today().isoformat())
        state["archive_hash"] = _archive_retired_state(state)
    else:
        state["status"] = "RETIRING_PENDING_SETTLEMENT"

    atomic_write_bytes(STATE_PATH, json.dumps(state, indent=2).encode("utf-8"))
    summary = (
        f"wind paper retirement: {state['status']} | "
        f"sailor_3x {state['sailor_3x']['status']} "
        f"({len(state['sailor_3x']['trades'])} trades) | "
        f"barrier_20x {state['barrier_20x']['status']} "
        f"({len(state['barrier_20x']['trades'])} trades)"
    )
    for line in lines:
        print(line)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
