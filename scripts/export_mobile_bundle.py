"""Export the seed data bundle for the Android companion app.

Writes the verified canonical recommendation plus non-actionable evidence/calendar metadata.
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
from edgestack.recommendation.service import CanonicalBundleRepository

SEED = Path("android/app/src/main/assets/seed")


def main() -> int:
    SEED.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat(timespec="seconds")

    cfg = load_config("configs/live.yaml")
    catalog = DataCatalog(cfg)
    bundle = CanonicalBundleRepository(catalog.artifacts_dir).latest()
    atomic_write_bytes(SEED / "recommendation.json", bundle.model_dump_json(indent=2).encode())
    cal = xcals.get_calendar("XNYS", start="2025-01-01", end="2030-12-31")
    sessions = [str(s.date()) for s in cal.sessions_in_range(cal.first_session, cal.last_session)]
    atomic_write_bytes(
        SEED / "calendar.json",
        json.dumps({"schema_version": 1, "exchange": "XNYS", "sessions": sessions}).encode(),
    )

    atomic_write_bytes(
        SEED / "meta.json",
        json.dumps(
            {
                "schema_version": 2,
                "generated_at": now,
                "config_hash": cfg.config_hash(),
                "disclaimer": bundle.disclaimer,
            }
        ).encode(),
    )
    print(f"canonical seed written to {SEED} ({bundle.session}, {len(sessions)} sessions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
