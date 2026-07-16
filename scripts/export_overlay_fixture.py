"""Export test fixtures pinning the Kotlin OverlayCalculator port to the
Python reference implementation (scripts/calendar_overlay.py:build_exposure).

Writes android/app/src/test/resources/fixtures/:
  overlay_spy_base10.json / overlay_spy_base13.json — SPY bars 2015->present and
    the APPLIED (shifted) exposure series for base 1.0 and 1.3
  yahoo_chart_sample.json — truncated raw chart response incl. a null-close bar
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
from calendar_overlay import build_exposure
from seasonality_scan import fetch

from edgestack.data.catalog import atomic_write_bytes

FIXTURES = Path("android/app/src/test/resources/fixtures")


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

    df = fetch(session, "SPY", interval="1d", period1=0, period2=int(time.time()))
    df = df.loc[df["dt"] >= "2015-01-01"].reset_index(drop=True)
    bars = [{
        "date": str(row.dt.date()), "open": float(row.open),
        "high": float(row.high), "low": float(row.low),
        "close": float(row.close), "adj": float(row.adj),
    } for row in df.itertuples()]

    for base, name in ((1.0, "overlay_spy_base10.json"),
                       (1.3, "overlay_spy_base13.json")):
        applied, _ = build_exposure(df, base=base)
        payload = {
            "base": base,
            "bars": bars,
            "applied_exposure": [{"date": str(d.date()), "L": float(v)}
                                 for d, v in applied.items()],
        }
        atomic_write_bytes(FIXTURES / name, json.dumps(payload).encode())
        print(f"{name}: {len(bars)} bars, {len(applied)} exposures, "
              f"mean L {applied.mean():.3f}")

    # ensemble4 parity fixture: same bars, exposure from the package reference
    from edgestack.strategies import ensemble_exposure, family_positions

    bars_df = df.rename(columns={"dt": "dt"}).copy()
    bars_df = bars_df.assign(
        date=bars_df["dt"].dt.tz_localize(None).dt.normalize()
    ).set_index("date")
    ens = ensemble_exposure(bars_df)
    fams = family_positions(bars_df)
    atomic_write_bytes(FIXTURES / "ensemble_spy.json", json.dumps({
        "bars": bars,
        "exposure": [{"date": str(d.date()), "e": float(v)}
                     for d, v in ens.items()],
        "last_families": {k: float(fams[k].iloc[-1]) for k in fams.columns},
    }).encode())
    print(f"ensemble_spy.json: {len(ens)} exposures")

    # raw chart sample for DTO tests: 30 bars, one with a null close
    r = session.get(
        "https://query1.finance.yahoo.com/v8/finance/chart/SPY",
        params={"period1": int(time.time()) - 60 * 86400,
                "period2": int(time.time()), "interval": "1d",
                "events": "div,splits", "includeAdjustedClose": "true"},
        timeout=30)
    raw = r.json()
    res = raw["chart"]["result"][0]
    n = min(30, len(res["timestamp"]))
    res["timestamp"] = res["timestamp"][:n]
    quote = res["indicators"]["quote"][0]
    for key in quote:
        quote[key] = quote[key][:n]
    res["indicators"]["adjclose"][0]["adjclose"] = \
        res["indicators"]["adjclose"][0]["adjclose"][:n]
    quote["close"][5] = None  # pin drop-null-bars behavior
    quote["open"][5] = None
    atomic_write_bytes(FIXTURES / "yahoo_chart_sample.json",
                       json.dumps(raw).encode())
    print(f"yahoo_chart_sample.json: {n} bars (bar 5 nulled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
