"""Export the seed data bundle for the Android companion app.

Writes android/app/src/main/assets/seed/{board,edges,calendar,meta}.json.
These small JSON files ARE committed so the app builds and runs offline from
a fresh clone; refresh them by rerunning this script after a new board.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import exchange_calendars as xcals

from edgestack.config import load_config
from edgestack.data.catalog import DataCatalog, atomic_write_bytes
from edgestack.discovery.edge_store import current_statuses, load_edges
from edgestack.types import EdgeStatus

SEED = Path("android/app/src/main/assets/seed")


def main() -> int:
    SEED.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat(timespec="seconds")

    board_path = Path("artifacts") / "live_board.json"
    if not board_path.exists():
        raise SystemExit("artifacts/live_board.json missing — run "
                         "scripts/live_signals.py first")
    atomic_write_bytes(SEED / "board.json", board_path.read_bytes())
    picks_path = Path("artifacts") / "picks.json"
    if picks_path.exists():
        atomic_write_bytes(SEED / "picks.json", picks_path.read_bytes())

    cfg = load_config("configs/live.yaml")
    catalog = DataCatalog(cfg)
    statuses = current_statuses(catalog).set_index("edge_id")["status"].to_dict()
    edges = [json.loads(e.model_dump_json())
             | {"current_status": statuses.get(e.identity.edge_id)}
             for e in load_edges(catalog, statuses=(EdgeStatus.VALIDATED,))]
    atomic_write_bytes(SEED / "edges.json", json.dumps(
        {"schema_version": 1, "generated_at": now, "edges": edges}).encode())

    cal = xcals.get_calendar("XNYS", start="2025-01-01", end="2030-12-31")
    sessions = [str(s.date()) for s in
                cal.sessions_in_range(cal.first_session, cal.last_session)]
    atomic_write_bytes(SEED / "calendar.json", json.dumps(
        {"schema_version": 1, "exchange": "XNYS", "sessions": sessions}).encode())

    atomic_write_bytes(SEED / "meta.json", json.dumps({
        "schema_version": 1, "generated_at": now,
        "config_hash": cfg.config_hash(),
        "disclaimer": "Research output only. Not investment advice.",
    }).encode())
    print(f"seed bundle written to {SEED} "
          f"({len(edges)} edges, {len(sessions)} sessions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
