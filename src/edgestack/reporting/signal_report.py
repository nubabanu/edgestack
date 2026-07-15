"""Persist and render signal reports (JSON + terminal tables)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from edgestack.data.catalog import DataCatalog, atomic_write_bytes
from edgestack.exceptions import DataError
from edgestack.types import SignalCandidate, SignalReport


def save_report(catalog: DataCatalog, report: SignalReport) -> Path:
    payload = report.model_dump_json(indent=2)
    out_dir = catalog.data_dir / "reports"
    path = out_dir / f"signals_{report.as_of_date.isoformat()}.json"
    atomic_write_bytes(path, payload.encode("utf-8"))
    with catalog.connect() as con:
        con.execute(
            "INSERT INTO signal_reports VALUES (?, ?, ?, ?)",
            [report.as_of_date, datetime.now(UTC), report.config_hash, payload],
        )
    return path


def load_report(catalog: DataCatalog, as_of: date | None = None) -> SignalReport:
    with catalog.connect() as con:
        if as_of is None:
            row = con.execute(
                "SELECT payload FROM signal_reports ORDER BY as_of_date DESC, "
                "created_at DESC LIMIT 1"
            ).fetchone()
        else:
            row = con.execute(
                "SELECT payload FROM signal_reports WHERE as_of_date = ? "
                "ORDER BY created_at DESC LIMIT 1",
                [as_of],
            ).fetchone()
    if row is None:
        raise DataError("no signal reports stored; run `edgestack signals generate`")
    return SignalReport.model_validate_json(row[0])


def render_tables(report: SignalReport, top: int = 20) -> str:
    lines = [
        f"signal report for {report.as_of_date} "
        f"(config {report.config_hash[:12]}) — RESEARCH OUTPUT, NOT ADVICE",
    ]
    for warning in report.universe_warnings:
        lines.append(f"! {warning}")
    for title, candidates in (
        ("LONG candidates", report.long_candidates),
        ("SHORT candidates (research only)", report.short_candidates),
    ):
        lines.append("")
        lines.append(f"== {title} ==")
        if not candidates:
            lines.append("  (none — the system abstained)")
            continue
        lines.append(
            f"  {'symbol':<8}{'conv':>6}{'P(net>0)':>10}{'E[net]':>9}{'hold':>6}"
            f"{'entry zone':>18}{'stop':>9}{'target':>9}{'R:R':>6}"
        )
        for c in candidates[:top]:
            lines.append(_candidate_line(c))
    n_abstain = len(report.abstentions)
    lines.append("")
    lines.append(f"abstentions: {n_abstain} (symbol/side states with recorded reasons)")
    lines.append(f"disclaimer: {report.disclaimer}")
    return "\n".join(lines)


def _candidate_line(c: SignalCandidate) -> str:
    zone = f"{c.entry.ideal_low:.2f}-{c.entry.ideal_high:.2f}"
    target = f"{c.risk.target_1:.2f}"
    return (
        f"  {c.symbol:<8}{c.conviction_score:>6.1f}"
        f"{c.calibrated_probability_of_positive_net_return:>10.2f}"
        f"{c.expected_net_return:>9.4f}{c.recommended_holding_sessions:>6}"
        f"{zone:>18}{c.risk.stop_price:>9.2f}{target:>9}{c.risk.reward_to_risk:>6.2f}"
    )
