"""Per-stage run-status ledger for the nightly chain.

``artifacts/nightly_status.json`` records the outcome of every stage the
nightly chain runs — the orchestrator's internal stages (data_update,
quality_gates, features, news, publish) and the bat-chained follow-up scripts
(tranche_watch, retired WIND settlement, WIND execution shadow,
universe_snapshot, intraday_collector, edgar_earnings) — so a
failure anywhere is visible and reflected in the chain's exit code instead of
being silently discarded.

Aggregate exit codes: 0 = all ok, 2 = a core stage failed (publication is
compromised), 1 = only auxiliary stages failed (publication fine, degraded).

CLI: ``python scripts/nightly_status.py start`` resets the ledger at the top
of a run.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from edgestack.data.catalog import atomic_write_bytes

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "artifacts" / "nightly_status.json"

# Stages whose failure compromises the canonical publication itself.
CORE_STAGES = frozenset({"nightly", "data_update", "quality_gates", "features", "publish"})


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_bytes(path, json.dumps(payload, indent=2).encode("utf-8"))


def load_status(path: Path = STATUS_PATH) -> dict[str, Any]:
    """Current ledger contents ({} when missing or corrupt)."""
    return _load(path)


def start_run(run_date: str | None = None, path: Path = STATUS_PATH) -> None:
    """Reset the ledger at the top of a nightly run."""
    _write(
        path,
        {
            "run_date": run_date or date.today().isoformat(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "overall": "running",
            "exit_code": None,
            "stages": {},
        },
    )


def record_stage(
    name: str,
    status: str,
    *,
    exit_code: int | None = None,
    duration_s: float | None = None,
    detail: dict[str, Any] | None = None,
    path: Path = STATUS_PATH,
) -> None:
    """Record one stage outcome; re-recording a stage overwrites it (retry wins)."""
    data = _load(path)
    data.setdefault("stages", {})
    entry: dict[str, Any] = {"status": status}
    if exit_code is not None:
        entry["exit_code"] = exit_code
    if duration_s is not None:
        entry["duration_s"] = round(duration_s, 1)
    if detail is not None:
        entry["detail"] = detail
    data["stages"][name] = entry
    _write(path, data)


def aggregate(path: Path = STATUS_PATH) -> tuple[str, int]:
    """Fold stage outcomes into (overall, exit_code) and persist them."""
    data = _load(path)
    stages: dict[str, Any] = data.get("stages", {})
    failed = [name for name, entry in stages.items() if entry.get("status") != "ok"]
    if not stages:
        overall, code = "failed", 2  # nothing recorded: the chain never ran
    elif not failed:
        overall, code = "ok", 0
    elif any(name in CORE_STAGES for name in failed):
        overall, code = "failed", 2
    else:
        overall, code = "degraded", 1
    data["overall"] = overall
    data["exit_code"] = code
    _write(path, data)
    return overall, code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["start"])
    parser.add_argument("--date", default=None, help="ISO run date (default: today)")
    args = parser.parse_args()
    start_run(args.date)
    print(f"nightly status ledger reset: {STATUS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
