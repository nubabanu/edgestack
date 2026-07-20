"""Agent toolkit: one JSON-speaking CLI over everything this repo can do.

Purpose: let an AI agent (or any script) use the codebase as a tool without
reading its internals. Every subcommand prints a single JSON object to
stdout (errors too: {"error": ...}, exit 1), so output is directly
machine-parseable. Human-facing caveats ride along in the payloads —
repeat them when relaying results; nothing here is investment advice.

    python scripts/agent_toolkit.py manual              # start here
    python scripts/agent_toolkit.py status              # freshness overview
    python scripts/agent_toolkit.py snapshot ACN        # indicator state now
    python scripts/agent_toolkit.py bars ACN --tail 5   # raw OHLCV records
    python scripts/agent_toolkit.py analyze CTSH --entry-date 2026-11-02
    python scripts/agent_toolkit.py edge-check CTSH     # zoo rules on one symbol
    python scripts/agent_toolkit.py earnings ACN        # EDGAR-stamped events
    python scripts/agent_toolkit.py universe --as-of 2020-03-31
    python scripts/agent_toolkit.py research-summary    # campaign verdicts
    python scripts/agent_toolkit.py watch               # tranche trigger status

Conventions honored: research results labeled previously-accessed, zoo
timing (signal close t -> earns t+1, 2 bps), corrupted-series screens.
Requires the repo venv (.venv/Scripts/python.exe) and curated data.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402 — path setup must precede repo imports
import pandas as pd  # noqa: E402

PRICES = ROOT / "data" / "curated" / "prices"
EVENTS = ROOT / "data" / "curated" / "events"
ARTIFACTS = ROOT / "artifacts"

DISCLAIMER = "Historical research on previously-accessed data; not investment advice."

STALE_THRESHOLDS = {
    "required_daily_bar_max_session_lag": 1,
    "publication_max_session_lag": 1,
    "watcher_max_calendar_age_days": 1,
    "universe_snapshot_max_calendar_age_days": 1,
    "intraday_max_session_lag": 1,
    "earnings_refresh_max_calendar_age_days": 8,
}
WATCHER_TRIGGER_TYPES = (
    "T1",
    "T2",
    "T3",
    "REL",
    "SECTOR",
    "CAL",
    "WINDOW",
    "REVIEW",
    "GO",
)
INTRADAY_COLLECTOR_SYMBOLS = (
    "ACN",
    "CTSH",
    "EPAM",
    "DXC",
    "IBM",
    "IT",
    "SPY",
    "QQQ",
    "TLT",
    "SHY",
    "GLD",
)
EARNINGS_COLLECTOR_SYMBOLS = ("ACN", "CTSH", "EPAM", "DXC", "IBM", "IT")


def _load(sym: str) -> pd.DataFrame:
    f = PRICES / f"{sym.upper()}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"no curated bars for {sym!r} (data/curated/prices)")
    df = pd.read_parquet(f).assign(date=lambda d: pd.to_datetime(d["date"]))
    return df.set_index("date").sort_index()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


# ---------------------------------------------------------------- commands
def cmd_manual(_args: argparse.Namespace) -> dict:
    return {
        "what": "JSON CLI over the edgestack research platform, made for AI agents",
        "interpreter": ".venv/Scripts/python.exe (repo venv; plain python lacks deps)",
        "commands": {
            "status": "data/publication/watcher freshness — call first in a session",
            "snapshot SYM": "current indicator state (trend, dips, vol, drawdown, earnings)",
            "bars SYM [--tail N]": "raw daily OHLCV + adj_close records",
            "analyze SYM [--entry-date YYYY-MM-DD]": (
                "canonical V2 instrument analysis: best/worst timing windows per horizon, "
                "tail/headwinds, intended-entry assessment. Descriptive unless a promoted "
                "artifact exists."
            ),
            "edge-check SYM": (
                "run the full strategy-zoo rule library on one symbol with split Sharpes "
                "and pooled Newey-West alpha; survivor bar identical to docs/strategy-zoo.md"
            ),
            "earnings SYM": "EDGAR 8-K 2.02 acceptance timestamps + XBRL EPS (ingests if absent)",
            "universe [--as-of DATE]": "point-in-time S&P 500 membership",
            "research-summary": "headline verdicts of every stored research campaign",
            "watch": "tranche watcher status (triggers for ACN/CTSH/EPAM)",
        },
        "caveats": [
            DISCLAIMER,
            "Yahoo coverage is survivorship-biased; screen corrupted series "
            "(single-day |return|>200%) before universe scans.",
            "2024+ results are previously-accessed and can never promote a sleeve.",
        ],
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _parse_datetime(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _session_lag(observed: date, today: date) -> int:
    """NYSE sessions after ``observed`` through ``today`` (weekends do not age data)."""
    if observed >= today:
        return 0
    from edgestack.data.calendar import TradingCalendar

    return TradingCalendar("XNYS").sessions_between(observed, today)


def _required_etfs(root: Path) -> tuple[str, ...]:
    policy_path = root / "configs" / "policies" / "baseline-diversified-v1.yaml"
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    return tuple(
        str(item["symbol"]).upper()
        for item in policy["weights"]
        if str(item.get("asset_kind", "")).upper() == "ETF"
    )


def _latest_daily_bar(path: Path) -> date | None:
    if not path.exists():
        return None
    dates = pd.read_parquet(path, columns=["date"])["date"]
    return None if dates.empty else pd.Timestamp(dates.max()).date()


def _issue(issues: list[dict[str, str]], *, code: str, severity: str, message: str) -> None:
    issues.append({"code": code, "severity": severity, "message": message})


def _market_data_health(
    root: Path,
    *,
    today: date,
    required_etfs: tuple[str, ...],
    watch_symbols: tuple[str, ...],
    issues: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, str | None]]:
    prices = root / "data" / "curated" / "prices"
    ordered_symbols = tuple(dict.fromkeys(required_etfs + watch_symbols))
    symbols: dict[str, dict[str, Any]] = {}
    last_bar: dict[str, str | None] = {}
    stale_required: list[str] = []
    for symbol in ordered_symbols:
        try:
            observed = _latest_daily_bar(prices / f"{symbol}.parquet")
        except (OSError, ValueError, KeyError) as exc:
            observed = None
            _issue(
                issues,
                code="daily_bar_unreadable",
                severity="critical" if symbol in required_etfs else "warning",
                message=f"{symbol} daily bars are unreadable: {exc}",
            )
        lag = _session_lag(observed, today) if observed else None
        stale = observed is None or lag > STALE_THRESHOLDS["required_daily_bar_max_session_lag"]
        symbols[symbol] = {
            "last_bar": observed.isoformat() if observed else None,
            "session_lag": lag,
            "required": symbol in required_etfs,
            "stale": stale,
        }
        last_bar[symbol] = observed.isoformat() if observed else None
        if symbol in required_etfs and stale:
            stale_required.append(symbol)
    if stale_required:
        _issue(
            issues,
            code="required_daily_bars_stale",
            severity="critical",
            message=f"required ETF daily bars are missing or stale: {', '.join(stale_required)}",
        )
    return (
        {
            "status": "healthy" if not stale_required else "unhealthy",
            "required_etfs": list(required_etfs),
            "stale_required_etfs": stale_required,
            "symbols": symbols,
        },
        last_bar,
    )


def _publication_health(
    root: Path, *, today: date, now: datetime, issues: list[dict[str, str]]
) -> dict[str, Any]:
    recommendations = root / "artifacts" / "recommendations"
    pointer_path = recommendations / "current.json"
    if not pointer_path.exists():
        _issue(
            issues,
            code="publication_missing",
            severity="critical",
            message="canonical recommendation pointer is missing",
        )
        return {
            "run_id": None,
            "session": None,
            "published_at": None,
            "age_hours": None,
            "session_lag": None,
            "stale": True,
        }
    try:
        pointer = _read_json(pointer_path)
        run_id = str(pointer["run_id"])
        record = _read_json(recommendations / "runs" / run_id / "publication.json")
        session = date.fromisoformat(str(pointer["session"]))
        published_at = _parse_datetime(record["published_at"])
        session_lag = _session_lag(session, today)
        age_hours = max((now - published_at).total_seconds() / 3600.0, 0.0)
        stale = session_lag > STALE_THRESHOLDS["publication_max_session_lag"]
        if stale:
            _issue(
                issues,
                code="publication_stale",
                severity="critical",
                message=(
                    f"canonical publication session {session} is {session_lag} NYSE sessions behind"
                ),
            )
        return {
            "run_id": run_id,
            "session": session.isoformat(),
            "published_at": published_at.isoformat().replace("+00:00", "Z"),
            "age_hours": round(age_hours, 1),
            "session_lag": session_lag,
            "stale": stale,
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            code="publication_invalid",
            severity="critical",
            message=f"canonical publication metadata is invalid: {exc}",
        )
        return {
            "run_id": None,
            "session": None,
            "published_at": None,
            "age_hours": None,
            "session_lag": None,
            "stale": True,
        }


def _active_watcher_triggers(symbol_status: dict[str, Any]) -> list[str]:
    active = [
        trigger
        for trigger in ("T1", "T2", "T3", "REL")
        if isinstance(symbol_status.get(trigger), dict)
        and bool(symbol_status[trigger].get("fired"))
    ]
    if symbol_status.get("CAL"):
        active.append("CAL")
    if symbol_status.get("window_open"):
        active.append("WINDOW")
    return active


def _watcher_health(
    root: Path, *, today: date, issues: list[dict[str, str]]
) -> tuple[dict[str, Any], tuple[str, ...]]:
    path = root / "artifacts" / "tranche_watch.json"
    if not path.exists():
        _issue(
            issues,
            code="watcher_status_missing",
            severity="warning",
            message="tranche watcher status is missing",
        )
        return (
            {
                "status": "missing",
                "run_status": None,
                "run_date": None,
                "age_days": None,
                "stale": True,
                "trigger_types": list(WATCHER_TRIGGER_TYPES),
                "fired": {},
                "global_active": [],
                "events": [],
                "go_scores": {},
            },
            (),
        )
    try:
        payload = _read_json(path)
        run_status = str(payload.get("run_status", "ok"))
        run_date = date.fromisoformat(str(payload["run_date"]))
        age_days = max((today - run_date).days, 0)
        stale = age_days > STALE_THRESHOLDS["watcher_max_calendar_age_days"]
        symbol_rows = [row for row in payload.get("symbols", []) if isinstance(row, dict)]
        watch_symbols = tuple(
            str(row["symbol"]).upper() for row in symbol_rows if row.get("symbol")
        )
        active = {
            str(row["symbol"]).upper(): _active_watcher_triggers(row)
            for row in symbol_rows
            if row.get("symbol")
        }
        breadth = payload.get("breadth") if isinstance(payload.get("breadth"), dict) else {}
        sector_active = int(breadth.get("count", 0)) >= 4
        events = payload.get("events", [])
        if not isinstance(events, list):
            events = []
        go_scores = {
            str(row["symbol"]).upper(): row.get("GO")
            for row in symbol_rows
            if row.get("symbol") and row.get("GO") is not None
        }
        if stale:
            _issue(
                issues,
                code="watcher_stale",
                severity="warning",
                message=f"tranche watcher last ran {age_days} calendar days ago",
            )
        if run_status != "ok":
            _issue(
                issues,
                code="watcher_run_failed",
                severity="warning",
                message=f"tranche watcher reported run_status={run_status}",
            )
        return (
            {
                "status": "healthy" if not stale and run_status == "ok" else run_status,
                "run_status": run_status,
                "run_date": run_date.isoformat(),
                "age_days": age_days,
                "stale": stale,
                "trigger_types": list(WATCHER_TRIGGER_TYPES),
                # Backward-compatible key, now including CAL and WINDOW conditions.
                "fired": active,
                "global_active": ["SECTOR"] if sector_active else [],
                "events": events,
                "go_scores": go_scores,
                "go_alerts_enabled": bool(payload.get("go_alerts_enabled", False)),
                "breadth": breadth,
            },
            watch_symbols,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            code="watcher_status_invalid",
            severity="warning",
            message=f"tranche watcher status is invalid: {exc}",
        )
        return (
            {
                "status": "invalid",
                "run_status": None,
                "run_date": None,
                "age_days": None,
                "stale": True,
                "trigger_types": list(WATCHER_TRIGGER_TYPES),
                "fired": {},
                "global_active": [],
                "events": [],
                "go_scores": {},
            },
            (),
        )


def _universe_collector_health(
    root: Path, *, today: date, issues: list[dict[str, str]]
) -> dict[str, Any]:
    directory = root / "data" / "curated" / "universe_snapshots"
    snapshots = sorted(directory.glob("*.json")) if directory.exists() else []
    if not snapshots:
        _issue(
            issues,
            code="universe_snapshot_missing",
            severity="warning",
            message="no forward universe snapshot is available",
        )
        return {
            "status": "missing",
            "count": 0,
            "latest": None,
            "age_days": None,
            "stale": True,
        }
    latest = snapshots[-1]
    try:
        payload = _read_json(latest)
        observed = date.fromisoformat(str(payload.get("date", latest.stem)))
        age_days = max((today - observed).days, 0)
        source = str(payload.get("membership_source", "unknown"))
        stale = age_days > STALE_THRESHOLDS[
            "universe_snapshot_max_calendar_age_days"
        ] or source.startswith("unavailable")
        if stale:
            _issue(
                issues,
                code="universe_snapshot_stale",
                severity="warning",
                message=f"latest universe snapshot is {age_days} days old; source={source}",
            )
        return {
            "status": "healthy" if not stale else "stale",
            "count": len(snapshots),
            "latest": observed.isoformat(),
            "age_days": age_days,
            "stale": stale,
            "membership_source": source,
            "sp500_members": len(payload.get("sp500_members", [])),
            "catalog_active": len(payload.get("catalog_active", [])),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            code="universe_snapshot_invalid",
            severity="warning",
            message=f"latest universe snapshot is invalid: {exc}",
        )
        return {
            "status": "invalid",
            "count": len(snapshots),
            "latest": latest.stem,
            "age_days": None,
            "stale": True,
        }


def _intraday_collector_health(
    root: Path, *, today: date, issues: list[dict[str, str]]
) -> dict[str, Any]:
    base = root / "data" / "curated" / "intraday"
    intervals: dict[str, Any] = {}
    collector_stale = False
    for interval in ("15m", "60m"):
        last_bars: dict[str, str | None] = {}
        stale_symbols: list[str] = []
        for symbol in INTRADAY_COLLECTOR_SYMBOLS:
            path = base / interval / f"{symbol}.parquet"
            observed: date | None = None
            try:
                if path.exists():
                    timestamps = pd.read_parquet(path, columns=["timestamp"])["timestamp"]
                    if not timestamps.empty:
                        latest = pd.to_datetime(timestamps, utc=True).max()
                        observed = latest.tz_convert("America/New_York").date()
            except (OSError, ValueError, KeyError):
                observed = None
            lag = _session_lag(observed, today) if observed else None
            stale = observed is None or lag > STALE_THRESHOLDS["intraday_max_session_lag"]
            last_bars[symbol] = observed.isoformat() if observed else None
            if stale:
                stale_symbols.append(symbol)
        interval_stale = bool(stale_symbols)
        collector_stale |= interval_stale
        intervals[interval] = {
            "status": "healthy" if not interval_stale else "stale",
            "last_bar": last_bars,
            "stale_symbols": stale_symbols,
        }
    if collector_stale:
        summary = "; ".join(
            f"{interval}: {', '.join(value['stale_symbols'])}"
            for interval, value in intervals.items()
            if value["stale_symbols"]
        )
        _issue(
            issues,
            code="intraday_collector_stale",
            severity="warning",
            message=f"intraday collector outputs are missing or stale ({summary})",
        )
    return {
        "status": "healthy" if not collector_stale else "stale",
        "expected_symbols": list(INTRADAY_COLLECTOR_SYMBOLS),
        "intervals": intervals,
    }


def _earnings_collector_health(
    root: Path, *, now: datetime, issues: list[dict[str, str]]
) -> dict[str, Any]:
    directory = root / "data" / "curated" / "events"
    symbols: dict[str, Any] = {}
    stale_symbols: list[str] = []
    for symbol in EARNINGS_COLLECTOR_SYMBOLS:
        path = directory / f"{symbol}.parquet"
        refreshed_at: datetime | None = None
        if path.exists():
            refreshed_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        age_days = (now - refreshed_at).total_seconds() / 86400.0 if refreshed_at else None
        stale = (
            refreshed_at is None
            or age_days > STALE_THRESHOLDS["earnings_refresh_max_calendar_age_days"]
        )
        symbols[symbol] = {
            "refreshed_at": (
                refreshed_at.isoformat().replace("+00:00", "Z") if refreshed_at else None
            ),
            "age_days": round(max(age_days, 0.0), 1) if age_days is not None else None,
            "stale": stale,
        }
        if stale:
            stale_symbols.append(symbol)
    if stale_symbols:
        _issue(
            issues,
            code="earnings_collector_stale",
            severity="warning",
            message=f"EDGAR collector outputs are missing or stale: {', '.join(stale_symbols)}",
        )
    all_event_files = list(directory.glob("*.parquet")) if directory.exists() else []
    return {
        "status": "healthy" if not stale_symbols else "stale",
        "expected_symbols": list(EARNINGS_COLLECTOR_SYMBOLS),
        "available_symbols": len(all_event_files),
        "stale_symbols": stale_symbols,
        "symbols": symbols,
    }


def build_status_report(
    root: Path = ROOT,
    *,
    today: date | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a machine-readable operational health report without mutating state."""
    checked_at = now or datetime.now(UTC)
    if checked_at.tzinfo is None:
        checked_at = checked_at.replace(tzinfo=UTC)
    checked_at = checked_at.astimezone(UTC)
    report_date = today or date.today()
    issues: list[dict[str, str]] = []

    try:
        required_etfs = _required_etfs(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        required_etfs = ()
        _issue(
            issues,
            code="baseline_policy_invalid",
            severity="critical",
            message=f"cannot determine required ETFs from baseline policy: {exc}",
        )

    watcher, watch_symbols = _watcher_health(root, today=report_date, issues=issues)
    market_data, last_bar = _market_data_health(
        root,
        today=report_date,
        required_etfs=required_etfs,
        watch_symbols=watch_symbols,
        issues=issues,
    )
    publication = _publication_health(root, today=report_date, now=checked_at, issues=issues)
    universe = _universe_collector_health(root, today=report_date, issues=issues)
    intraday = _intraday_collector_health(root, today=report_date, issues=issues)
    earnings = _earnings_collector_health(root, now=checked_at, issues=issues)

    severities = {item["severity"] for item in issues}
    health = "unhealthy" if "critical" in severities else "degraded" if issues else "healthy"
    return {
        "health": health,
        "healthy": health == "healthy",
        "checked_at": checked_at.isoformat().replace("+00:00", "Z"),
        "today": report_date.isoformat(),
        "thresholds": STALE_THRESHOLDS,
        "market_data": market_data,
        # Preserved for callers of the old compact status contract.
        "last_bar": last_bar,
        "publication": publication,
        "tranche_watch": watcher,
        "collectors": {
            "universe_snapshot": universe,
            "intraday": intraday,
            "earnings": earnings,
        },
        # Preserved aliases; detailed collector state lives under collectors.
        "universe_snapshots": {
            "count": universe["count"],
            "latest": universe["latest"],
        },
        "events_symbols": earnings["available_symbols"],
        "issues": issues,
        "disclaimer": DISCLAIMER,
    }


def cmd_status(_args: argparse.Namespace) -> dict:
    return build_status_report()


def cmd_snapshot(args: argparse.Namespace) -> dict:
    df = _load(args.symbol)
    c, h, low = df["close"], df["high"], df["low"]
    ret = df["adj_close"].pct_change()
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    hi252 = c.rolling(252).max()
    ibs = ((c - low) / (h - low).replace(0, np.nan)).fillna(0.5)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    earnings = None
    cache = ARTIFACTS / "earnings_cache.json"
    if cache.exists():
        earnings = json.loads(cache.read_text()).get(args.symbol.upper(), {}).get("date")
    return {
        "symbol": args.symbol.upper(),
        "as_of": str(df.index.max().date()),
        "close": round(float(c.iloc[-1]), 2),
        "drawdown_from_52w_high": round(float(c.iloc[-1] / hi252.iloc[-1] - 1), 3),
        "vs_sma50": round(float(c.iloc[-1] / s50.iloc[-1] - 1), 3),
        "vs_sma200": round(float(c.iloc[-1] / s200.iloc[-1] - 1), 3),
        "rsi2": round(float(_rsi(c, 2).iloc[-1]), 1),
        "rsi14": round(float(_rsi(c, 14).iloc[-1]), 1),
        "ibs": round(float(ibs.iloc[-1]), 2),
        "vol20_annualized": round(float(vol20.iloc[-1]), 3),
        "down_days_last3": int((ret < 0).iloc[-3:].sum()),
        "ret_5d": round(float(df["adj_close"].pct_change(5).iloc[-1]), 4),
        "next_earnings_cached": earnings,
        "disclaimer": DISCLAIMER,
    }


def cmd_bars(args: argparse.Namespace) -> dict:
    df = _load(args.symbol).tail(args.tail).reset_index()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return {"symbol": args.symbol.upper(), "bars": df.to_dict(orient="records")}


def cmd_analyze(args: argparse.Namespace) -> dict:
    from edgestack.config import load_config
    from edgestack.data.catalog import DataCatalog
    from edgestack.recommendation.instrument import analyze_instrument, resolve_instrument
    from edgestack.recommendation.service import CanonicalBundleRepository

    cfg = load_config(str(ROOT / "configs" / "live.yaml"))
    catalog = DataCatalog(cfg)
    bundle = CanonicalBundleRepository(catalog.artifacts_dir).latest()
    res = resolve_instrument(args.symbol.upper(), requested_kind=None, canonical=bundle)
    with catalog.guard.unlock(
        reason="agent_toolkit descriptive instrument analysis; previously accessed"
    ) as key:
        daily = catalog.load_panel(
            symbols=(res.resolved_symbol,), end=bundle.session, unlock_key=key
        )
    intended = None
    if args.entry_date:
        from datetime import time as dtime
        from zoneinfo import ZoneInfo

        intended = datetime.combine(
            date.fromisoformat(args.entry_date), dtime(9, 30), ZoneInfo("America/New_York")
        )
    a = analyze_instrument(
        bundle=bundle, resolution=res, daily_bars=daily, intended_entry_at=intended
    )

    def win(w) -> dict | None:
        if w is None:
            return None
        return {
            "label": w.label,
            "expected_net_return": w.expected_net_return,
            "observations": w.observations,
            "evidence_grade": str(w.evidence_grade),
            "actionable": w.actionable,
        }

    return {
        "symbol": res.resolved_symbol,
        "overall_rating": str(a.overall_rating),
        "canonical_portfolio_weight": a.canonical_portfolio_weight,
        "horizons": {
            str(h.horizon): {
                "best": win(h.best_window),
                "worst": win(h.worst_window),
                "intended_entry_assessment": h.intended_entry_assessment,
                "warning": h.warning,
            }
            for h in a.horizon_analyses
        },
        "tailwinds": [e.observation for e in a.tailwinds],
        "headwinds": [e.observation for e in a.headwinds],
        "warnings": list(a.warnings),
        "disclaimer": DISCLAIMER,
    }


def cmd_edge_check(args: argparse.Namespace) -> dict:
    from strategy_zoo import build_rules, run_rule

    from edgestack.validation.advanced_tests import newey_west_alpha
    from edgestack.validation.metrics import sharpe_ratio

    df = _load(args.symbol).rename(columns={"adj_close": "adj"})
    if float(df["adj"].pct_change().abs().max()) > 2.0:
        return {"error": f"{args.symbol}: corrupted price series (>200% daily move) — refusing"}
    start_year = df.index.min().year
    mid = max(start_year + 6, 2016)
    splits = {
        f"dev_{start_year}_{mid - 1}": (f"{start_year}-01-01", f"{mid - 1}-12-31"),
        f"val_{mid}_2023": (f"{mid}-01-01", "2023-12-31"),
        "holdout_2024": ("2024-01-01", "2026-12-31"),
    }
    ret = df["adj"].pct_change()
    rows = []
    for name, pos in build_rules(df).items():
        strat = run_rule(pos, ret)
        row: dict[str, Any] = {"rule": name}
        ok = True
        for split, (lo, hi) in splits.items():
            s = strat.loc[lo:hi].dropna()
            if len(s) < 200:
                ok = False
                continue
            b = ret.loc[s.index].fillna(0.0)
            row[split] = round(sharpe_ratio(s.to_numpy()), 2)
            row[split + "_bh"] = round(sharpe_ratio(b.to_numpy()), 2)
            if row[split] < row[split + "_bh"]:
                ok = False
        pooled = strat.dropna()
        nw = newey_west_alpha(pooled, ret.loc[pooled.index].fillna(0.0))
        row["alpha_ann"] = round(nw["alpha_ann"], 4)
        row["alpha_t"] = round(nw["alpha_t"], 2)
        row["survivor"] = bool(ok and nw["alpha_t"] >= 2.0)
        rows.append(row)
    rows.sort(key=lambda r: -r["alpha_t"])
    return {
        "symbol": args.symbol.upper(),
        "splits": splits,
        "trials": len(rows),
        "expected_false_positives_at_t2": round(len(rows) * 0.05, 1),
        "top_rules": rows[: args.top],
        "survivors": [r["rule"] for r in rows if r["survivor"]],
        "disclaimer": DISCLAIMER + " Survivor bar: Sharpe>=B&H all splits AND pooled NW t>=2.",
    }


def cmd_earnings(args: argparse.Namespace) -> dict:
    sym = args.symbol.upper()
    f = EVENTS / f"{sym}.parquet"
    if not f.exists():
        import subprocess

        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "edgar_earnings.py"), "--symbols", sym],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if not f.exists():
            return {"error": f"EDGAR ingest failed for {sym}: {r.stdout[-200:]} {r.stderr[-200:]}"}
    df = pd.read_parquet(f).sort_values("accepted_at").tail(args.tail)
    events = [
        {
            "accepted_at": str(r["accepted_at"]),
            "form": r["form"],
            "period_end": str(r["period_end"])[:10] if pd.notna(r["period_end"]) else None,
            "eps": None if pd.isna(r["eps"]) else float(r["eps"]),
        }
        for _, r in df.iterrows()
    ]
    return {"symbol": sym, "source": "SEC EDGAR 8-K item 2.02 + XBRL", "events": events}


def cmd_universe(args: argparse.Namespace) -> dict:
    from edgestack.data.universe_pit import PitSP500Universe

    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
    members = PitSP500Universe(ROOT / "data" / "cache" / "universe", earliest=date(2000, 1, 3))
    syms = members.members(as_of)
    return {
        "as_of": as_of.isoformat(),
        "count": len(syms),
        "members": list(syms) if args.full else [*list(syms)[:25], "..."],
        "caveat": "reconstructed from the public change log; PIT approximation before ~2005",
    }


def cmd_research_summary(_args: argparse.Namespace) -> dict:
    out: dict[str, Any] = {"disclaimer": DISCLAIMER}
    zoo = ARTIFACTS / "strategy_zoo.json"
    if zoo.exists():
        rows = json.loads(zoo.read_text())
        if isinstance(rows, dict):
            rows = rows.get("results", [])
        surv = [r for r in rows if r.get("SURVIVOR")]
        out["strategy_zoo"] = {
            "trials": len(rows),
            "survivors": [f"{r['symbol']}:{r['rule']}" for r in surv],
        }
    for name, key in [
        ("unified_book", "unified_book.json"),
        ("tranche_policy_backtest", "tranche_policy_backtest.json"),
        ("pead_study", "pead_study.json"),
        ("execution_sensitivity", "execution_sensitivity.json"),
    ]:
        f = ARTIFACTS / key
        if f.exists():
            out[name] = json.loads(f.read_text())
    out["docs"] = "full narratives: docs/strategy-zoo.md, artifacts/institutional_report_*.md"
    return out


def _screened_symbols() -> list[str]:
    """Curated symbols passing the corruption / penny / liquidity screens."""
    out = []
    for f in PRICES.glob("*.parquet"):
        df = pd.read_parquet(f, columns=["close", "volume", "adj_close"])
        if len(df) < 300:
            continue
        if float(df["adj_close"].pct_change().abs().max()) > 2.0:
            continue
        tail = df.tail(60)
        if float(tail["adj_close"].min()) < 1.0:
            continue
        if float((tail["close"] * tail["volume"]).median()) < 5e6:
            continue
        out.append(f.stem)
    return out


def cmd_vol_screen(args: argparse.Namespace) -> dict:
    rows = []
    for sym in _screened_symbols():
        df = _load(sym)
        ret = df["adj_close"].pct_change()
        vol = float(ret.rolling(20).std().iloc[-1]) * float(np.sqrt(252))
        if not np.isfinite(vol):
            continue
        c = df["close"]
        dd = float(c.iloc[-1] / c.rolling(252).max().iloc[-1] - 1)
        rows.append(
            {
                "symbol": sym,
                "vol20_annualized": round(vol, 3),
                "drawdown_from_52w_high": round(dd, 3),
                "close": round(float(c.iloc[-1]), 2),
                "worst_day_1y": round(float(ret.tail(252).min()), 3),
            }
        )
    rows.sort(key=lambda r: -r["vol20_annualized"])
    return {
        "as_of": rows[0].get("as_of", date.today().isoformat()) if rows else None,
        "screens": "no corrupted series, price>=$1, median $ volume>=$5M (last 60 sessions)",
        "top": rows[: args.top],
        "warning": (
            "High volatility is a cost, not an edge: the cross-sectional studies in this "
            "repo found high-vol names have the WORST risk-adjusted forward returns. "
            "Run leverage-check before sizing anything."
        ),
    }


def cmd_leverage_check(args: argparse.Namespace) -> dict:
    """Historical survival of an L-times leveraged CFD-style position.

    Model: static position at L x notional, forced close-out when equity
    falls to 50% of initial margin (i.e. an adverse excursion of 50%/L
    from entry, intraday low counts). Entries: every session, and dip-
    trigger sessions (RSI2<10 or 3 down days or IBS<0.2). Horizon
    args.horizon sessions. No costs, no overnight financing — reality is
    strictly worse than these numbers.
    """
    lev = args.leverage
    df = _load(args.symbol)
    c, low, h = df["close"], df["low"], df["high"]
    adj = df["adj_close"]
    ret = adj.pct_change()
    factor = adj / c
    alow = low * factor
    r2 = _rsi(c, 2)
    ibs = ((c - low) / (h - low).replace(0, np.nan)).fillna(0.5)
    down3 = (ret < 0) & (ret.shift() < 0) & (ret.shift(2) < 0)
    dip = (r2 < 10) | down3 | (ibs < 0.2)
    s200 = c.rolling(200).mean()
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    calm = (c > s200) & (vol20 < 0.30)  # validated trend + low-vol regime gate
    liq_move = 0.5 / lev  # adverse move that halves the margin -> close-out
    H = args.horizon

    def survival(entry_idx: list[int]) -> dict:
        n_liq = 0
        finals = []
        for i in entry_idx:
            if i + 1 + H >= len(df):
                continue
            e = float(adj.iloc[i + 1])  # enter next session (zoo timing)
            window_low = float(alow.iloc[i + 1 : i + 1 + H].min())
            if window_low / e - 1 <= -liq_move:
                n_liq += 1
                finals.append(-0.5)  # closed out at half the margin
            else:
                finals.append(lev * (float(adj.iloc[i + 1 + H]) / e - 1))
        if not finals:
            return {"entries": 0}
        s = pd.Series(finals)
        return {
            "entries": len(finals),
            "liquidated_pct": round(n_liq / len(finals), 3),
            "median_return_on_margin": round(float(s.median()), 3),
            "p10_return_on_margin": round(float(s.quantile(0.1)), 3),
        }

    def mae95_of(entry_idx: list[int], step: int = 1) -> float:
        maes = []
        for i in entry_idx[::step]:
            if i + 1 + H >= len(df):
                continue
            e = float(adj.iloc[i + 1])
            maes.append(1 - float(alow.iloc[i + 1 : i + 1 + H].min()) / e)
        return float(np.quantile(maes, 0.95)) if maes else float("nan")

    all_idx = list(range(len(df)))
    calm_idx = list(np.flatnonzero(calm.to_numpy()))
    mae95 = mae95_of(all_idx, step=5)
    mae95_calm = mae95_of(calm_idx)
    return {
        "symbol": args.symbol.upper(),
        "leverage": lev,
        "close_out_move": round(liq_move, 3),
        "horizon_sessions": H,
        "all_entries": survival(all_idx),
        "calm_regime_entries": survival(calm_idx),
        "calm_gate": "close > SMA200 and 20d vol < 30% (validated trend+low-vol regime)",
        "max_leverage_95pct_survival_calm": (
            round(0.5 / mae95_calm, 2) if np.isfinite(mae95_calm) and mae95_calm > 0 else None
        ),
        "dip_trigger_entries": survival(list(np.flatnonzero(dip.to_numpy()))),
        "max_leverage_95pct_survival": round(0.5 / mae95, 2) if np.isfinite(mae95) else None,
        "notes": (
            "Close-out at 50% of initial margin, intraday lows count, zero costs/"
            "financing modeled — live results are worse. 'max_leverage_95pct_survival' "
            "is the leverage that would have avoided close-out in 95% of historical "
            f"{H}-session windows; it is descriptive, not a guarantee."
        ),
        "disclaimer": DISCLAIMER,
    }


def cmd_quote(args: argparse.Namespace) -> dict:
    """Provider-neutral indicative quotes with explicit source and freshness."""
    from edgestack.data.quote_service import build_live_quote_service

    batch = build_live_quote_service().fetch(args.symbols, provider=args.provider)
    if not batch.quotes:
        attempted = ", ".join(batch.providers_attempted) or args.provider
        raise RuntimeError(f"no quotes available; attempted: {attempted}")
    return batch.model_dump(mode="json")


def cmd_go_backtest(args: argparse.Namespace) -> dict:
    """Validate the tranche-watch GO score: forward returns by score bucket."""
    from tranche_watch import BASKET, GO_ALERT_THRESHOLD, go_score

    sym = args.symbol.upper()
    df = _load(sym)
    c, low, h = df["close"], df["low"], df["high"]
    adj = df["adj_close"]
    ret = adj.pct_change()
    idx = df.index

    # basket level + breadth (aligned to this symbol's dates)
    adjs, above = {}, {}
    for b in BASKET:
        bdf = _load(b)
        adjs[b] = bdf["adj_close"]
        above[b] = (bdf["close"] > bdf["close"].rolling(50).mean()).astype(float)
    basket = pd.DataFrame(adjs).dropna()
    basket_level = (basket / basket.iloc[0]).mean(axis=1)
    breadth = pd.DataFrame(above).reindex(idx).sum(axis=1)

    s200 = c.rolling(200).mean()
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    r2 = _rsi(c, 2)
    ibs = ((c - low) / (h - low).replace(0, np.nan)).fillna(0.5)
    down3 = (ret < 0) & (ret.shift() < 0) & (ret.shift(2) < 0)
    t1 = (r2 < 10) | down3 | (ibs < 0.2)
    calm = (c > s200) & (vol20 < 0.30)
    ratio = (adj / basket_level.reindex(idx)).dropna()
    rz = ((ratio - ratio.rolling(60).mean()) / ratio.rolling(60).std()).reindex(idx)
    rel_recross = (rz > -1) & (rz.shift() <= -1)

    tdom = pd.Series(range(len(idx)), index=idx).groupby(idx.to_period("M")).cumcount() + 1
    seasonal = ((idx.month == 11) & (tdom <= 3).to_numpy()) | ((idx.month == 6) & (idx.day >= 23))
    if sym == "CTSH":
        seasonal |= (idx.month == 12) & ((tdom >= 10) & (tdom <= 15)).to_numpy()

    post_earnings = pd.Series(False, index=idx)
    ev_file = EVENTS / f"{sym}.parquet"
    events_used = False
    if ev_file.exists():
        acc = pd.to_datetime(pd.read_parquet(ev_file)["accepted_at"]).dt.tz_convert(
            "America/New_York"
        )
        for ts in acc.dt.normalize().dt.tz_localize(None).unique():
            pos = idx.searchsorted(ts)
            post_earnings.iloc[pos : pos + 6] = True
        events_used = True

    scores = pd.Series(
        [
            go_score(
                t1=bool(t1.iloc[i]),
                near_dip=bool((r2.iloc[i] < 20) or (ibs.iloc[i] < 0.3)),
                calm=bool(calm.iloc[i]),
                low_vol=bool(vol20.iloc[i] < 0.30) if np.isfinite(vol20.iloc[i]) else False,
                rel_recross=bool(rel_recross.iloc[i]) if np.isfinite(rz.iloc[i]) else False,
                rel_ok=bool(rz.iloc[i] > -1) if np.isfinite(rz.iloc[i]) else False,
                breadth=int(breadth.iloc[i]),
                seasonal=bool(seasonal[i]),
                post_earnings=bool(post_earnings.iloc[i]),
            )
            for i in range(len(idx))
        ],
        index=idx,
    )

    out: dict[str, Any] = {
        "symbol": sym,
        "events_component": events_used,
        "threshold": GO_ALERT_THRESHOLD,
        "score_distribution": {
            str(q): int(v) for q, v in scores.quantile([0.25, 0.5, 0.75, 0.95]).items()
        },
    }
    splits = {
        "dev": ("2011-01-01", "2017-12-31"),
        "val": ("2018-01-01", "2023-12-31"),
        "holdout": ("2024-01-01", "2026-12-31"),
    }
    for horizon in (5, 20, 60):
        fwd = adj.shift(-1 - horizon) / adj.shift(-1) - 1
        buckets = pd.cut(scores, [-1, 19, 39, 59, 100], labels=["0-19", "20-39", "40-59", "60+"])
        tbl = fwd.groupby(buckets, observed=True).agg(["mean", "count"])
        out[f"h{horizon}_by_bucket"] = {
            str(k): {"mean_fwd": round(float(v["mean"]), 4), "n": int(v["count"])}
            for k, v in tbl.iterrows()
        }
        uncond = float(fwd.mean())
        consistent = 0
        for lo, hi_ in splits.values():
            m = (scores.index >= lo) & (scores.index <= hi_)
            hi_scores = fwd[m & (scores >= GO_ALERT_THRESHOLD)]
            if len(hi_scores) >= 5 and float(hi_scores.mean()) > float(fwd[m].mean()):
                consistent += 1
        out[f"h{horizon}_threshold_beats_unconditional_splits"] = consistent
        out[f"h{horizon}_unconditional"] = round(uncond, 4)
    out["gate"] = (
        "PASS: enable GO alerts"
        if out["h20_threshold_beats_unconditional_splits"] >= 2
        and out["h20_by_bucket"].get("60+", {}).get("mean_fwd", -1)
        > out["h20_by_bucket"].get("0-19", {}).get("mean_fwd", 0)
        else "FAIL: keep score display-only (set GO_ALERTS_ENABLED=False)"
    )
    out["disclaimer"] = DISCLAIMER
    return out


def cmd_watch(_args: argparse.Namespace) -> dict:
    import subprocess

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "tranche_watch.py")],
        capture_output=True,
        text=True,
        timeout=300,
    )
    f = ARTIFACTS / "tranche_watch.json"
    if not f.exists():
        return {"error": "tranche watcher produced no status file"}
    return json.loads(f.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("manual")
    sub.add_parser("status")
    p = sub.add_parser("snapshot")
    p.add_argument("symbol")
    p = sub.add_parser("bars")
    p.add_argument("symbol")
    p.add_argument("--tail", type=int, default=10)
    p = sub.add_parser("analyze")
    p.add_argument("symbol")
    p.add_argument("--entry-date", default="")
    p = sub.add_parser("edge-check")
    p.add_argument("symbol")
    p.add_argument("--top", type=int, default=10)
    p = sub.add_parser("earnings")
    p.add_argument("symbol")
    p.add_argument("--tail", type=int, default=8)
    p = sub.add_parser("universe")
    p.add_argument("--as-of", default="")
    p.add_argument("--full", action="store_true")
    sub.add_parser("research-summary")
    sub.add_parser("watch")
    p = sub.add_parser("vol-screen")
    p.add_argument("--top", type=int, default=15)
    p = sub.add_parser("leverage-check")
    p.add_argument("symbol")
    p.add_argument("--leverage", type=float, default=5.0)
    p.add_argument("--horizon", type=int, default=60)
    p = sub.add_parser("go-backtest")
    p.add_argument("symbol")
    p = sub.add_parser("quote")
    p.add_argument("symbols", help="comma-separated, e.g. EPAM,CTSH,SPY")
    p.add_argument(
        "--provider",
        choices=("auto", "finnhub", "twelvedata", "alpaca", "yahoo"),
        default="auto",
        help="force one provider for diagnostics (default: failover automatically)",
    )
    args = parser.parse_args()

    handlers = {
        "manual": cmd_manual,
        "status": cmd_status,
        "snapshot": cmd_snapshot,
        "bars": cmd_bars,
        "analyze": cmd_analyze,
        "edge-check": cmd_edge_check,
        "earnings": cmd_earnings,
        "universe": cmd_universe,
        "research-summary": cmd_research_summary,
        "watch": cmd_watch,
        "vol-screen": cmd_vol_screen,
        "leverage-check": cmd_leverage_check,
        "go-backtest": cmd_go_backtest,
        "quote": cmd_quote,
    }
    try:
        result = handlers[args.command](args)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 1 if isinstance(result, dict) and "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
