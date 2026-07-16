"""Point-in-time S&P 500 membership reconstructed from the public change log.

Method: start from today's constituent list and walk the published
addition/removal log backwards, so membership at any historical date includes
companies that were LATER removed (acquired, delisted, bankrupt) — genuine
point-in-time membership, free of selection-at-the-end bias.

Honest limitations (quantified by the ingestion script, never hidden):
- membership is point-in-time, but PRICE HISTORY for many removed tickers is
  unavailable on free feeds (acquired/delisted symbols disappear) — coverage
  is measured and reported with every universe version;
- the public change log is reliable from roughly the mid-2000s onward;
- tickers are normalized (dots to dashes) and can collide across re-uses.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

from edgestack.exceptions import ProviderError
from edgestack.logging import get_logger, log_event
from edgestack.types import UniverseSnapshot

log = get_logger("universe.pit")

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_FAR_FUTURE = date(2100, 1, 1)


@dataclass(frozen=True)
class MembershipInterval:
    symbol: str
    start: date  # first date the symbol is known to be a member
    end: date  # exclusive: removed on this date (or far future)


def _norm(symbol: str) -> str:
    return symbol.strip().upper().replace(".", "-")


def fetch_change_log(cache_dir: Path, *, max_age_days: int = 7) -> dict:
    """Download and cache the constituents + changes tables as JSON."""
    cache_file = cache_dir / "sp500_changes.json"
    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(payload["fetched_at"])
        if (datetime.now() - fetched).days < max_age_days:
            return payload

    resp = requests.get(WIKI_URL, timeout=30, headers={"User-Agent": "edgestack-research/0.1"})
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    current = tables[0]
    changes = tables[1]
    changes.columns = [
        "_".join(str(c) for c in col).lower() if isinstance(col, tuple) else str(col).lower()
        for col in changes.columns
    ]

    def col(frame: pd.DataFrame, *needles: str) -> str:
        for c in frame.columns:
            name = str(c).lower()
            if all(n in name for n in needles):
                return c
        raise ProviderError(f"missing column {needles} in {list(frame.columns)}")

    change_rows = []
    date_col = col(changes, "date")
    add_col = col(changes, "added", "ticker")
    rem_col = col(changes, "removed", "ticker")
    for row in changes.itertuples(index=False):
        raw_date = str(getattr(row, date_col, "")) if hasattr(row, date_col) else str(row[0])
        # positional access is safer with mangled tuple names
        vals = dict(zip(changes.columns, row))
        raw_date = str(vals[date_col])
        m = re.search(r"(\w+ \d{1,2}, \d{4})", raw_date)
        if not m:
            continue
        when = datetime.strptime(m.group(1), "%B %d, %Y").date()
        added = str(vals[add_col]) if pd.notna(vals[add_col]) else ""
        removed = str(vals[rem_col]) if pd.notna(vals[rem_col]) else ""
        change_rows.append(
            {
                "date": when.isoformat(),
                "added": _norm(added) if added and added != "nan" else "",
                "removed": _norm(removed) if removed and removed != "nan" else "",
            }
        )

    sym_col = col(current, "symbol")
    payload = {
        "fetched_at": datetime.now().isoformat(),
        "current": sorted(_norm(s) for s in current[sym_col].astype(str)),
        "changes": change_rows,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log_event(
        log,
        20,
        "sp500 change log fetched",
        current=len(payload["current"]),
        changes=len(change_rows),
    )
    return payload


def build_membership(payload: dict, *, earliest: date) -> list[MembershipInterval]:
    """Walk the change log backwards from today's membership."""
    changes = sorted(
        (c for c in payload["changes"] if date.fromisoformat(c["date"]) >= earliest),
        key=lambda c: c["date"],
        reverse=True,
    )
    intervals: list[MembershipInterval] = []
    open_since: dict[str, date] = dict.fromkeys(payload["current"], _FAR_FUTURE)
    # open_since[s] holds the (exclusive) END of the currently-open interval
    # while walking backwards; _FAR_FUTURE means "still a member today".
    for change in changes:
        when = date.fromisoformat(change["date"])
        added, removed = change["added"], change["removed"]
        if added and added in open_since:
            intervals.append(MembershipInterval(added, when, open_since.pop(added)))
        if removed:
            open_since.setdefault(removed, when)
    for symbol, end in open_since.items():
        intervals.append(MembershipInterval(symbol, earliest, end))
    return sorted(intervals, key=lambda i: (i.symbol, i.start))


class PitSP500Universe:
    """Point-in-time S&P 500 membership queries."""

    def __init__(self, cache_dir: Path, earliest: date) -> None:
        self.payload = fetch_change_log(cache_dir)
        self.intervals = build_membership(self.payload, earliest=earliest)
        self.earliest = earliest

    def members(self, as_of: date) -> tuple[str, ...]:
        return tuple(sorted({i.symbol for i in self.intervals if i.start <= as_of < i.end}))

    def ever_members(self, start: date, end: date) -> tuple[str, ...]:
        return tuple(sorted({i.symbol for i in self.intervals if i.start < end and i.end > start}))

    def membership_mask(self, symbols: pd.Index, dates: pd.Series) -> pd.Series:
        """Vectorized point-in-time membership for (symbol, date) rows."""
        by_symbol: dict[str, list[tuple[date, date]]] = {}
        for i in self.intervals:
            by_symbol.setdefault(i.symbol, []).append((i.start, i.end))
        out = pd.Series(False, index=dates.index)
        d = pd.to_datetime(dates).dt.date
        for symbol, spans in by_symbol.items():
            rows = symbols == symbol
            if not rows.any():
                continue
            m = pd.Series(False, index=dates.index[rows])
            dd = d[rows]
            for start, end in spans:
                m |= (dd >= start) & (dd < end)
            out.loc[rows] = m
        return out

    def snapshot(self, as_of: date, *, coverage: float | None = None) -> UniverseSnapshot:
        limitations = [
            "membership is point-in-time (public change log), but price history "
            "for many removed tickers is unavailable on free feeds",
        ]
        if coverage is not None:
            limitations.append(
                f"only {coverage:.0%} of point-in-time members have usable price "
                "data; results remain partially survivor-biased and the gap is "
                "concentrated in acquired/delisted names"
            )
        return UniverseSnapshot(
            as_of_date=as_of,
            symbols=self.members(as_of),
            source="wikipedia-sp500-changelog",
            methodology="reverse-walked addition/removal log",
            is_point_in_time=True,
            limitations=tuple(limitations),
        )
