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
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402 — path setup must precede repo imports
import pandas as pd  # noqa: E402

PRICES = ROOT / "data" / "curated" / "prices"
EVENTS = ROOT / "data" / "curated" / "events"
ARTIFACTS = ROOT / "artifacts"

DISCLAIMER = "Historical research on previously-accessed data; not investment advice."


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


def cmd_status(_args: argparse.Namespace) -> dict:
    out: dict[str, Any] = {"today": date.today().isoformat()}
    freshness = {}
    for sym in ("SPY", "ACN", "CTSH", "EPAM"):
        try:
            freshness[sym] = str(_load(sym).index.max().date())
        except FileNotFoundError:
            freshness[sym] = None
    out["last_bar"] = freshness
    cur = ARTIFACTS / "recommendations" / "current.json"
    if cur.exists():
        pub = json.loads(cur.read_text())
        out["publication"] = {k: pub.get(k) for k in ("run_id", "session", "published_at")}
    watch = ARTIFACTS / "tranche_watch.json"
    if watch.exists():
        w = json.loads(watch.read_text())
        out["tranche_watch"] = {
            "run_date": w.get("run_date"),
            "fired": {
                s["symbol"]: [t for t in ("T1", "T2", "T3", "REL") if s[t]["fired"]]
                for s in w.get("symbols", [])
            },
        }
    snaps = sorted((ROOT / "data" / "curated" / "universe_snapshots").glob("*.json"))
    out["universe_snapshots"] = {
        "count": len(snaps),
        "latest": snaps[-1].stem if snaps else None,
    }
    out["events_symbols"] = len(list(EVENTS.glob("*.parquet"))) if EVENTS.exists() else 0
    return out


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
