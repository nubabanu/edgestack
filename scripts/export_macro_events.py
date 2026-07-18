"""Export the point-in-time macro-event seed for the Android companion app.

Writes FOMC decision days (federalreserve.gov calendar, 2026-2027), CPI
release dates (BLS schedule, 2026), and weekly EIA petroleum-report
Wednesdays to android/app/src/main/assets/seed/macro_events.json. The file
is committed so the app shows event context offline; rerun this script when
the agencies publish new schedules.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgestack.data.catalog import atomic_write_bytes

SEED = Path("android/app/src/main/assets/seed")

# Decision (second) day of each scheduled meeting; press conference 14:30 ET.
FOMC_DECISION_DAYS = [
    "2026-01-28",
    "2026-03-18",
    "2026-04-29",
    "2026-06-17",
    "2026-07-29",
    "2026-09-16",
    "2026-10-28",
    "2026-12-09",
    "2027-01-27",
    "2027-03-17",
    "2027-04-28",
    "2027-06-09",
    "2027-07-28",
    "2027-09-15",
    "2027-10-27",
    "2027-12-08",
]

# BLS schedule; all releases 08:30 ET.
CPI_RELEASE_DAYS = [
    "2026-01-13",
    "2026-02-13",
    "2026-03-11",
    "2026-04-10",
    "2026-05-12",
    "2026-06-10",
    "2026-07-14",
    "2026-08-12",
    "2026-09-11",
    "2026-10-14",
    "2026-11-10",
    "2026-12-10",
]

EIA_MONTHS_AHEAD = 18


def eia_wednesdays(start: date, months: int) -> list[str]:
    end = start + timedelta(days=months * 31)
    day = start + timedelta(days=(2 - start.weekday()) % 7)  # next Wednesday
    out = []
    while day <= end:
        out.append(day.isoformat())
        day += timedelta(days=7)
    return out


def main() -> int:
    today = date.today()
    events = (
        [
            {"date": d, "type": "FOMC", "label": "FOMC rate decision", "time_et": "14:00"}
            for d in FOMC_DECISION_DAYS
        ]
        + [
            {"date": d, "type": "CPI", "label": "CPI release", "time_et": "08:30"}
            for d in CPI_RELEASE_DAYS
        ]
        + [
            {
                "date": d,
                "type": "EIA",
                "label": "EIA petroleum report (holiday weeks may shift)",
                "time_et": "10:30",
            }
            for d in eia_wednesdays(today, EIA_MONTHS_AHEAD)
        ]
    )
    events.sort(key=lambda item: (item["date"], item["type"]))
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "events": events,
    }
    SEED.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(SEED / "macro_events.json", json.dumps(payload, indent=2).encode())
    print(f"macro events written: {len(events)} ({SEED / 'macro_events.json'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
