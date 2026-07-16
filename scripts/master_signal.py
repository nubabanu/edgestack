"""Compute TODAY's master signal: the validated ensemble4 exposure per
instrument plus the diversified-core reference weights.

Output: printed summary + artifacts/master_signal.json (served by the API
at GET /master for the companion app).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
from seasonality_scan import fetch

from edgestack.data.catalog import atomic_write_bytes
from edgestack.strategies import (
    ensemble_exposure,
    family_positions,
    seasonal_multiplier,
)

INSTRUMENTS = ("SPY", "QQQ", "XLK")
# best all-era monthly package (allocation zoo; alpha t=2.20, beta 0.25)
CORE_WEIGHTS = {"SPY": 0.25, "TLT": 0.25, "SHY": 0.25, "GLD": 0.25}


def main() -> int:
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    now = int(time.time())

    out: dict = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "disclaimer": "Research output only. Not investment advice.",
        "core_reference": {
            "name": "permanent_portfolio_quarterly",
            "weights": CORE_WEIGHTS,
            "note": "best all-era monthly family (NW alpha t=2.20, beta 0.25); "
            "rebalance quarterly or on 5% bands",
        },
        "instruments": {},
    }
    print("=== MASTER SIGNAL (validated ensemble4) ===")
    for sym in INSTRUMENTS:
        df = fetch(session, sym, interval="1d", period1=now - 3 * 365 * 86400, period2=now)
        df = (
            df.assign(date=df["dt"].dt.tz_localize(None).dt.normalize())
            .set_index("date")
            .drop(columns="dt")
        )
        fams = family_positions(df)
        ens = ensemble_exposure(df)
        seas = seasonal_multiplier(df.index)
        as_of = df.index[-1]
        today = {
            "as_of": str(as_of.date()),
            "close": round(float(df["close"].iloc[-1]), 2),
            "families": {k: round(float(fams[k].iloc[-1]), 3) for k in fams.columns},
            "ensemble_exposure_next_session": round(float(ens.iloc[-1]), 3),
            "seasonal_multiplier_unvalidated": round(float(seas.iloc[-1]), 2),
        }
        out["instruments"][sym] = today
        fam_str = "  ".join(f"{k}={v:.2f}" for k, v in today["families"].items())
        print(
            f"{sym}: exposure {today['ensemble_exposure_next_session']:.2f} "
            f"(as of {today['as_of']}, close {today['close']})"
        )
        print(f"     {fam_str}")
        time.sleep(0.3)

    print(f"\nCORE (reference): {CORE_WEIGHTS} — quarterly/5%-band rebalance")
    atomic_write_bytes(Path("artifacts") / "master_signal.json", json.dumps(out, indent=1).encode())
    print("saved -> artifacts/master_signal.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
