"""Content-level data-quality checks.

Structural validity is enforced by :mod:`edgestack.data.schemas`; this module
looks for economically suspicious content: calendar gaps, price cliffs that
smell like unadjusted splits, stale quotes and zero-volume runs. Problems are
*reported and quarantined*, never silently fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from edgestack.data.calendar import TradingCalendar

# |log return| beyond this is flagged as a potential unadjusted corporate action.
JUMP_THRESHOLD = 0.35
STALE_RUN_SESSIONS = 10
ZERO_VOLUME_RUN = 5


@dataclass(frozen=True)
class SymbolQuality:
    symbol: str
    rows: int
    first_date: date
    last_date: date
    missing_sessions: int
    suspicious_jumps: tuple[str, ...]
    stale_price_runs: int
    zero_volume_runs: int
    issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class QualityReport:
    symbols: tuple[SymbolQuality, ...]

    @property
    def passed(self) -> tuple[SymbolQuality, ...]:
        return tuple(s for s in self.symbols if s.passed)

    @property
    def quarantined(self) -> tuple[SymbolQuality, ...]:
        return tuple(s for s in self.symbols if not s.passed)

    def summary(self) -> str:
        lines = [f"data quality: {len(self.passed)} passed, {len(self.quarantined)} quarantined"]
        for s in self.quarantined:
            lines.append(f"  QUARANTINE {s.symbol}: " + "; ".join(s.issues))
        for s in self.passed:
            lines.append(
                f"  OK {s.symbol}: {s.rows} rows {s.first_date}..{s.last_date}"
                + (f", {s.missing_sessions} missing sessions" if s.missing_sessions else "")
            )
        return "\n".join(lines)


def assess_panel(
    panel: pd.DataFrame, calendar: TradingCalendar, *, max_missing_fraction: float = 0.02
) -> QualityReport:
    """Assess a validated bar panel symbol by symbol."""
    results = []
    for symbol, group in panel.groupby("symbol", sort=True):
        results.append(_assess_symbol(str(symbol), group, calendar, max_missing_fraction))
    return QualityReport(symbols=tuple(results))


def _assess_symbol(
    symbol: str, bars: pd.DataFrame, calendar: TradingCalendar, max_missing_fraction: float
) -> SymbolQuality:
    bars = bars.sort_values("date")
    dates = pd.DatetimeIndex(bars["date"])
    first, last = dates[0].date(), dates[-1].date()
    issues: list[str] = []

    expected = calendar.sessions(first, last)
    missing = len(expected.difference(dates))
    extraneous = len(dates.difference(expected))
    if extraneous:
        issues.append(f"{extraneous} rows on non-session dates")
    if missing > max_missing_fraction * len(expected):
        issues.append(f"{missing}/{len(expected)} sessions missing")

    log_ret = np.log(bars["close"].to_numpy()[1:] / bars["close"].to_numpy()[:-1])
    jump_idx = np.where(np.abs(log_ret) > JUMP_THRESHOLD)[0]
    jumps = tuple(str(dates[i + 1].date()) for i in jump_idx[:10])
    if len(jump_idx):
        issues.append(
            f"{len(jump_idx)} price moves beyond +/-{JUMP_THRESHOLD:.0%} "
            f"(possible unadjusted corporate action) at {', '.join(jumps[:3])}"
        )

    stale_runs = _run_count(bars["close"].diff().to_numpy() == 0.0, STALE_RUN_SESSIONS)
    if stale_runs:
        issues.append(f"{stale_runs} stale-price runs (>={STALE_RUN_SESSIONS} flat closes)")

    zero_runs = _run_count(bars["volume"].to_numpy() == 0.0, ZERO_VOLUME_RUN)
    if zero_runs:
        issues.append(f"{zero_runs} zero-volume runs (>={ZERO_VOLUME_RUN} sessions)")

    return SymbolQuality(
        symbol=symbol,
        rows=len(bars),
        first_date=first,
        last_date=last,
        missing_sessions=missing,
        suspicious_jumps=jumps,
        stale_price_runs=stale_runs,
        zero_volume_runs=zero_runs,
        issues=tuple(issues),
    )


def _run_count(mask: np.ndarray, min_len: int) -> int:
    """Number of maximal True-runs with length >= min_len."""
    count = 0
    run = 0
    for value in mask:
        if bool(value):
            run += 1
        else:
            if run >= min_len:
                count += 1
            run = 0
    if run >= min_len:
        count += 1
    return count
