"""Ingest point-in-time earnings events from SEC EDGAR (free, keyless).

Two sources, both official and timestamped:
  1. Submissions API (data.sec.gov/submissions/CIK##########.json):
     8-K filings whose items include 2.02 "Results of Operations" — the
     earnings press release — with exact acceptanceDateTime. This is the
     point-in-time announcement stamp the causality gates need.
  2. XBRL companyfacts API: us-gaap EarningsPerShareDiluted (fallback
     Basic) quarterly values with their fiscal period ends.

Output: one parquet per symbol under data/curated/events/ with columns
  symbol, accepted_at (UTC), form, report_period, eps, eps_source
where accepted_at rows come from 8-K 2.02 events and eps rows are joined
by nearest following acceptance (an 8-K within 90 days after period end).

Honest limitations (do not hide when using this data):
- analyst consensus does not exist here; only seasonal-random-walk SUE
  (EPS vs same quarter prior year) is computable — which is the original
  Bernard & Thomas construction;
- EDGAR item tagging of old 8-Ks (pre-2004) is unreliable; coverage is
  measured and reported per symbol, never assumed;
- symbols are mapped to CIKs via SEC's company_tickers.json — ticker
  re-use collisions are possible for long-dead names.

Usage:
  python scripts/edgar_earnings.py --symbols ACN,CTSH,EPAM
  python scripts/edgar_earnings.py --symbols-file syms.txt [--force]

Fair use: ~10 req/s allowed; this script stays far below it (0.15 s
between requests) with a declared User-Agent, and skips symbols already
ingested unless --force.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "curated" / "events"
CIK_CACHE = ROOT / "data" / "cache" / "edgar_company_tickers.json"
POLITE_DELAY_S = 0.15
UA = {"User-Agent": "edgestack research (contact: yucelnumankaradavut@gmail.com)"}
EPS_TAGS = ("EarningsPerShareDiluted", "EarningsPerShareBasic")


def _get(session: requests.Session, url: str) -> dict:
    time.sleep(POLITE_DELAY_S)
    resp = session.get(url, headers=UA, timeout=30)
    resp.raise_for_status()
    return resp.json()


def cik_map(session: requests.Session) -> dict[str, int]:
    if CIK_CACHE.exists() and (
        datetime.now(UTC).timestamp() - CIK_CACHE.stat().st_mtime < 7 * 86400
    ):
        raw = json.loads(CIK_CACHE.read_text())
    else:
        raw = _get(session, "https://www.sec.gov/files/company_tickers.json")
        CIK_CACHE.parent.mkdir(parents=True, exist_ok=True)
        CIK_CACHE.write_text(json.dumps(raw))
    return {row["ticker"].upper(): int(row["cik_str"]) for row in raw.values()}


def earnings_8k_events(session: requests.Session, cik: int) -> pd.DataFrame:
    """All 8-K item 2.02 filings with acceptance datetimes, incl. older pages."""
    sub = _get(session, f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    frames = [pd.DataFrame(sub["filings"]["recent"])]
    for extra in sub["filings"].get("files", []):
        frames.append(
            pd.DataFrame(_get(session, f"https://data.sec.gov/submissions/{extra['name']}"))
        )
    f = pd.concat(frames, ignore_index=True)
    f = f[(f["form"].isin(["8-K", "8-K/A"])) & f["items"].str.contains("2.02", na=False)]
    out = pd.DataFrame(
        {
            "accepted_at": pd.to_datetime(f["acceptanceDateTime"], utc=True),
            "form": f["form"],
            "report_period": pd.to_datetime(f["reportDate"], errors="coerce"),
        }
    ).sort_values("accepted_at")
    return out.reset_index(drop=True)


def quarterly_eps(session: requests.Session, cik: int) -> pd.DataFrame:
    """Quarterly diluted (fallback basic) EPS from companyfacts XBRL."""
    facts = _get(session, f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in EPS_TAGS:
        units = gaap.get(tag, {}).get("units", {})
        rows = units.get("USD/shares", [])
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["start"] = pd.to_datetime(df["start"])
        df["end"] = pd.to_datetime(df["end"])
        # keep true quarterly windows (roughly 3 months), first-reported value
        q = df[(df["end"] - df["start"]).dt.days.between(80, 100)].copy()
        q["filed"] = pd.to_datetime(q["filed"])
        q = q.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
        return pd.DataFrame(
            {"period_end": q["end"], "eps": q["val"], "eps_source": tag}
        ).sort_values("period_end")
    return pd.DataFrame(columns=["period_end", "eps", "eps_source"])


def ingest_symbol(session: requests.Session, symbol: str, cik: int) -> pd.DataFrame:
    events = earnings_8k_events(session, cik)
    eps = quarterly_eps(session, cik)
    events["symbol"] = symbol
    if eps.empty:
        events[["period_end", "eps", "eps_source"]] = [pd.NaT, float("nan"), None]
        return events
    # join each quarter to the first 8-K accepted within 90 days after period end
    eps = eps.copy()
    acc = events["accepted_at"].dt.tz_localize(None)
    matches = []
    for _, row in eps.iterrows():
        after = events[(acc > row["period_end"]) & (acc <= row["period_end"] + pd.Timedelta("90D"))]
        matches.append(after["accepted_at"].iloc[0] if len(after) else pd.NaT)
    eps["accepted_at"] = matches
    merged = events.merge(
        eps.rename(columns={"period_end": "period_end"}), on="accepted_at", how="left"
    )
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbols-file", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--refresh-days",
        type=int,
        default=0,
        help="re-ingest symbols whose parquet is older than this many days (0 = never)",
    )
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.symbols_file:
        symbols += [
            s.strip().upper() for s in Path(args.symbols_file).read_text().splitlines() if s.strip()
        ]
    if not symbols:
        parser.error("no symbols given")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    ciks = cik_map(session)
    ok = skipped = failed = 0
    for sym in symbols:
        out = OUT_DIR / f"{sym}.parquet"
        if out.exists() and not args.force:
            age_days = (datetime.now(UTC).timestamp() - out.stat().st_mtime) / 86400
            if args.refresh_days <= 0 or age_days < args.refresh_days:
                skipped += 1
                continue
        cik = ciks.get(sym)
        if cik is None:
            print(f"WARN {sym}: no CIK mapping — skipped")
            failed += 1
            continue
        try:
            df = ingest_symbol(session, sym, cik)
        except requests.RequestException as exc:
            print(f"WARN {sym}: EDGAR fetch failed ({exc})")
            failed += 1
            continue
        df.to_parquet(out, index=False)
        with_eps = int(df["eps"].notna().sum())
        if len(df) and df["accepted_at"].notna().any():
            span = f"({df['accepted_at'].min():%Y-%m-%d} -> {df['accepted_at'].max():%Y-%m-%d})"
        else:
            span = "(no item-2.02 8-Ks found)"
        print(f"{sym}: {len(df)} earnings 8-Ks {span}, {with_eps} matched to XBRL EPS")
        ok += 1
    print(f"\ndone: {ok} ingested, {skipped} cached, {failed} failed -> {OUT_DIR}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
