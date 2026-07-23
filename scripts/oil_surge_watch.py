"""Oil geopolitical-shock reaction watcher (CL=F detection, Brent-ETC proxy tickets).

Born from the 2026-07 Bab al-Mandab closure: crude jumped ~10% and kept
climbing while the nightly watchers looked at IT-services stocks. This
watcher REACTS to oil price shocks — it predicts nothing. A strait closure,
an OPEC surprise and a short squeeze all look identical to the trigger; the
news headlines quoted in the alert are corroboration for the human, never a
trigger input.

What it does:
  SHOCK   CL=F front-month up >= +4% on the day, or >= +8% over 5 sessions.
          Alert fires immediately (descriptive fact). Intraday polls (Task
          Scheduler "EdgeStack Oil Surge Watch", every 30 min) send a
          PROVISIONAL shock heads-up from the live partial bar; the nightly
          EOD pass on the curated close is canonical.
  DIP     While a shock regime is armed (10 sessions, or until price
          round-trips below the pre-shock close), a pullback of >= 1.5%
          from the post-shock closing high fires a dip-entry alert.
          DISPLAY-ONLY until scripts/oil_shock_study.py records a PASS in
          artifacts/oil_shock_study_verdict.json AND a human flips
          DIP_TICKETS_ENABLED — the GO-score precedent (tranche_watch.py).

Honest caveats:
- scripts/uso_entry_study.py already showed the equity tranche triggers do
  NOT transfer to oil; these shock/dip rules are separate and start unproven.
- CL=F is a continuous front-month future with roll artifacts and is not
  itself tradeable here. BNO is the Brent data proxy for paper fills; EU
  retail cannot buy USO/BNO under PRIIPs — a real purchase would be a UCITS
  Brent ETC (for example WisdomTree Brent Crude Oil). Alerts say so.
- The oil fresh-entry service (edgestack oil check) treats SHIPPING_DISRUPTION
  as a hard veto for fresh CFD entries. That polarity is intentional and
  unchanged: this module is a separate, study-gated dip program on a proxy
  ETC, not a licence to open leveraged CFDs into chaos.
- No live orders ever (repo covenant). Paper book: artifacts/paper_oil.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PRICES = ROOT / "data" / "curated" / "prices"
STATE_PATH = ROOT / "artifacts" / "oil_surge_state.json"
VERDICT_PATH = ROOT / "artifacts" / "oil_shock_study_verdict.json"
PLAN_PATH = ROOT / "artifacts" / "tranche_plan.json"
PAPER_PATH = ROOT / "artifacts" / "paper_oil.json"
LOG_PATH = ROOT / "logs" / "oil_surge_watch.log"
NEWS_CACHE_DIR = ROOT / "data" / "cache" / "news"
TOAST_SCRIPT = ROOT / "scripts" / "notify_toast.ps1"

SHOCK_SYMBOL = "CL=F"  # detection series (longest free history)
PROXY_SYMBOL = "BNO"  # Brent data proxy for paper fills; see PRIIPs caveat above
CONFIRM_SYMBOL = "BNO"
VOL_SYMBOL = "^OVX"
NEWS_SYMBOLS = ("BNO", "USO")

SHOCK_DAY_RETURN = 0.04  # single-session close-to-close
SHOCK_WINDOW_RETURN = 0.08  # cumulative over SHOCK_WINDOW_SESSIONS
SHOCK_WINDOW_SESSIONS = 5
REGIME_SESSIONS = 10  # dip watch expires this many sessions after the (re)shock
DIP_PULLBACK = 0.015  # from post-shock closing high; R3 in oil_shock_study.py
DIP_COOLDOWN_SESSIONS = 2
POST_REGIME_COOLDOWN = 5  # sessions of suppression after a regime expires
WATCH_RULE = "R3"  # the study rule this watcher's live dip threshold implements

# GO-score precedent: tickets stay off until scripts/oil_shock_study.py
# records a PASS naming WATCH_RULE in artifacts/oil_shock_study_verdict.json
# AND a human flips this after reviewing the report. Either alone does nothing.
DIP_TICKETS_ENABLED = False

# Corroboration vocabulary only — never a trigger input.
GEOPOLITICAL_WORDS = frozenset(
    {
        "strait",
        "hormuz",
        "mandab",
        "mandeb",
        "suez",
        "red",
        "blockade",
        "closure",
        "closed",
        "attack",
        "attacks",
        "strike",
        "strikes",
        "missile",
        "drone",
        "tanker",
        "tankers",
        "houthi",
        "houthis",
        "opec",
        "embargo",
        "sanctions",
        "pipeline",
        "disruption",
        "escalation",
        "iran",
        "israel",
        "russia",
        "ukraine",
    }
)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

DEFAULT_STATE: dict = {
    "phase": "IDLE",  # IDLE | WATCHING | COOLDOWN
    "last_session": None,  # last curated CL=F session date processed by --eod
    "cooldown_left": 0,
    "episode": None,
    "intraday": {},  # dedupe keys for provisional intraday alerts
}


def load(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(PRICES / f"{symbol}.parquet")
    return (
        df.assign(date=pd.to_datetime(df["date"]))
        .set_index("date")
        .sort_index()[["open", "high", "low", "close", "adj_close"]]
    )


def load_state() -> dict:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
        for key, value in DEFAULT_STATE.items():
            state.setdefault(key, json.loads(json.dumps(value)))
        return state
    return json.loads(json.dumps(DEFAULT_STATE))


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def detect_shock(closes: pd.Series) -> dict | None:
    """Shock on the LAST bar of `closes`: day jump or 5-session window jump."""
    if len(closes) < 2:
        return None
    ret1 = float(closes.iloc[-1] / closes.iloc[-2] - 1)
    if ret1 >= SHOCK_DAY_RETURN:
        return {"kind": "DAY", "ret": ret1, "pre_close": float(closes.iloc[-2])}
    if len(closes) > SHOCK_WINDOW_SESSIONS:
        ret5 = float(closes.iloc[-1] / closes.iloc[-1 - SHOCK_WINDOW_SESSIONS] - 1)
        if ret5 >= SHOCK_WINDOW_RETURN:
            return {
                "kind": "WINDOW",
                "ret": ret5,
                "pre_close": float(closes.iloc[-1 - SHOCK_WINDOW_SESSIONS]),
            }
    return None


def step_session(state: dict, day: str, closes: pd.Series) -> list[dict]:
    """Advance the state machine by ONE session (the last bar of `closes`).

    Mutates `state`; returns machine events (SHOCK/RESHOCK/DIP/EXPIRED).
    Pure with respect to I/O — no files, no network — so tests can drive it.
    """
    events: list[dict] = []
    close = float(closes.iloc[-1])
    state["last_session"] = day

    if state["phase"] == "COOLDOWN":
        state["cooldown_left"] = max(0, int(state["cooldown_left"]) - 1)
        if state["cooldown_left"] == 0:
            state["phase"] = "IDLE"
            state["episode"] = None
        return events

    if state["phase"] == "WATCHING":
        ep = state["episode"]
        ep["sessions"] = int(ep["sessions"]) + 1
        ep["dip_cooldown"] = max(0, int(ep["dip_cooldown"]) - 1)
        ret1 = float(closes.iloc[-1] / closes.iloc[-2] - 1) if len(closes) >= 2 else 0.0
        if ret1 >= SHOCK_DAY_RETURN:
            # fresh leg: extend the regime instead of stacking a new episode
            ep["sessions"] = 0
            events.append({"type": "RESHOCK", "date": day, "ret": ret1})
        if close > float(ep["post_shock_high"]):
            ep["post_shock_high"] = close
        pullback = close / float(ep["post_shock_high"]) - 1
        if pullback <= -DIP_PULLBACK and ep["dip_cooldown"] == 0:
            ep["dip_cooldown"] = DIP_COOLDOWN_SESSIONS
            ep["dips"] = int(ep.get("dips", 0)) + 1
            events.append(
                {
                    "type": "DIP",
                    "date": day,
                    "pullback": pullback,
                    "session": ep["sessions"],
                    "shock_date": ep["shock_date"],
                    "shock_ret": ep["shock_ret"],
                }
            )
        if close < float(ep["pre_shock_close"]):
            events.append({"type": "EXPIRED", "date": day, "reason": "round-trip"})
        elif ep["sessions"] >= REGIME_SESSIONS:
            events.append({"type": "EXPIRED", "date": day, "reason": "regime elapsed"})
        if events and events[-1]["type"] == "EXPIRED":
            state["phase"] = "COOLDOWN"
            state["cooldown_left"] = POST_REGIME_COOLDOWN
        return events

    shock = detect_shock(closes)
    if shock:
        state["phase"] = "WATCHING"
        state["episode"] = {
            "shock_date": day,
            "shock_kind": shock["kind"],
            "shock_ret": shock["ret"],
            "pre_shock_close": shock["pre_close"],
            "post_shock_high": close,
            "sessions": 0,
            "dip_cooldown": 0,
            "dips": 0,
        }
        events.append({"type": "SHOCK", "date": day, "kind": shock["kind"], "ret": shock["ret"]})
    return events


def tickets_enabled() -> bool:
    """Double gate: constant flipped by a human AND a study PASS naming WATCH_RULE."""
    if not DIP_TICKETS_ENABLED or not VERDICT_PATH.exists():
        return False
    try:
        verdict = json.loads(VERDICT_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if verdict.get("verdict") != "PASS":
        return False
    return any(f"/{WATCH_RULE}@" in rule for rule in verdict.get("rules_passed", []))


def order_ticket() -> str:
    if not tickets_enabled() or not PLAN_PATH.exists():
        return ""
    plan = json.loads(PLAN_PATH.read_text())
    eur = plan.get(PROXY_SYMBOL, {}).get("OILDIP")
    return f" -> ORDER: buy ~EUR {eur} {PROXY_SYMBOL} (Brent-ETC proxy) at next open" if eur else ""


def paper_autopilot(fired: list[str]) -> list[str]:
    """PAPER execution of ticketed dip alerts (research only, no live orders).

    Same contract as tranche_watch.paper_autopilot but with its own ledger so
    oil P&L never mixes into the equity tranche book. Fills at PROXY_SYMBOL's
    actual next-session adjusted open. No-ops entirely while tickets are off.
    """
    book = json.loads(PAPER_PATH.read_text()) if PAPER_PATH.exists() else {"trades": []}
    lines: list[str] = []
    for tr in book["trades"]:
        if tr.get("fill_price") is None:
            df = load(tr["symbol"])
            after = df[df.index > pd.Timestamp(tr["signal_date"])]
            if len(after):
                row = after.iloc[0]
                aopen = float(row["open"] * row["adj_close"] / row["close"])
                tr["fill_price"] = round(aopen, 4)
                tr["fill_date"] = str(after.index[0].date())
                tr["shares"] = round(tr["eur"] / aopen, 4)
                lines.append(
                    f"PAPER filled {tr['symbol']} {tr['trigger']} EUR {tr['eur']} "
                    f"@ {aopen:.2f} ({tr['fill_date']})"
                )
    if tickets_enabled() and PLAN_PATH.exists():
        plan = json.loads(PLAN_PATH.read_text())
        eur = plan.get(PROXY_SYMBOL, {}).get("OILDIP")
        for sig_date in fired:
            if not eur:
                continue
            if any(
                t["symbol"] == PROXY_SYMBOL and t["signal_date"] == sig_date for t in book["trades"]
            ):
                continue
            book["trades"].append(
                {
                    "symbol": PROXY_SYMBOL,
                    "trigger": "OILDIP",
                    "eur": eur,
                    "signal_date": sig_date,
                    "fill_price": None,
                }
            )
            lines.append(f"PAPER intent {PROXY_SYMBOL} OILDIP EUR {eur} (fills at next open)")
    cost = value = 0.0
    for tr in book["trades"]:
        if tr.get("fill_price"):
            last = float(load(tr["symbol"])["adj_close"].iloc[-1])
            cost += tr["eur"]
            value += tr["shares"] * last
    if cost:
        lines.append(
            f"PAPER oil book: cost EUR {cost:.0f} value EUR {value:.0f} ({value / cost - 1:+.1%})"
        )
    if book["trades"] or PAPER_PATH.exists():
        book["_note"] = "Automatic paper execution of oil dip tickets; research only."
        PAPER_PATH.write_text(json.dumps(book, indent=2))
    return lines


def notify(alerts: list[str]) -> None:
    """Windows toast + Telegram, mirroring tranche_watch.notify with an oil title."""
    body = "; ".join(alerts)[:250]
    try:
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(TOAST_SCRIPT),
                "-Title",
                "EdgeStack oil alert",
                "-Body",
                body,
            ],
            timeout=30,
            capture_output=True,
            check=False,
        )
    except Exception as exc:
        print(f"WARN toast failed: {exc}")
    try:
        import notify_telegram

        notify_telegram.send("EdgeStack oil alert\n" + "\n".join(alerts)[:3800])
    except Exception as exc:
        print(f"WARN telegram failed: {exc}")


def geopolitical_headlines(max_items: int = 2) -> list[str]:
    """Best-effort: recent cached/RSS headlines whose tokens hit the geo vocabulary."""
    try:
        from edgestack.recommendation.news import gather_news_evidence

        evidence = gather_news_evidence(NEWS_SYMBOLS, NEWS_CACHE_DIR)
    except Exception as exc:
        print(f"WARN news unavailable: {exc}")
        return []
    cutoff = datetime.now(UTC) - timedelta(hours=48)
    hits: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        headline = item.headline
        if headline in seen or item.published_at < cutoff:
            continue
        tokens = {tok.strip(".,:;!?'\"()").lower() for tok in headline.split()}
        if tokens & GEOPOLITICAL_WORDS:
            seen.add(headline)
            hits.append(headline.encode("ascii", "replace").decode("ascii"))
        if len(hits) >= max_items:
            break
    return hits


def context_suffix(daily: dict[str, pd.DataFrame]) -> str:
    parts: list[str] = []
    confirm = daily.get(CONFIRM_SYMBOL)
    if confirm is not None and len(confirm) >= 2:
        cret = float(confirm["close"].iloc[-1] / confirm["close"].iloc[-2] - 1)
        parts.append(f"{CONFIRM_SYMBOL} {cret:+.1%}")
    vol = daily.get(VOL_SYMBOL)
    if vol is not None and len(vol):
        parts.append(f"OVX {float(vol['close'].iloc[-1]):.0f}")
    return f" ({', '.join(parts)})" if parts else ""


def shock_alert_text(event: dict, suffix: str, headlines: list[str], provisional: bool) -> str:
    tag = "provisional, live bar - nightly close confirms" if provisional else "EOD confirmed"
    kind = "today" if event.get("kind") == "DAY" else f"over {SHOCK_WINDOW_SESSIONS} sessions"
    text = (
        f"OIL SHOCK ({tag}): {SHOCK_SYMBOL} {event['ret']:+.1%} {kind}{suffix}. "
        f"Reacts to the move, predicts nothing. Dip watch armed for {REGIME_SESSIONS} "
        "sessions. EU note: real purchase = UCITS Brent ETC (PRIIPs blocks USO/BNO). "
        "If shipping is disrupted, consider the SHIPPING_DISRUPTION flag for "
        "`edgestack oil check` (hard veto there)."
    )
    if headlines:
        text += " News: " + " | ".join(f"'{h}'" for h in headlines)
    return text


def dip_alert_text(event: dict, provisional: bool) -> str:
    ticket = order_ticket()
    prefix = "" if ticket else "DISPLAY-ONLY: "
    tag = " (provisional, live bar - nightly close confirms)" if provisional else ""
    return (
        f"OIL DIP {prefix}pullback {event['pullback']:+.1%} from post-shock high, "
        f"session {event['session']} of surge regime (shock {event['shock_date']} "
        f"{event['shock_ret']:+.1%}){tag}."
        + (ticket or " Entry rule not validated by oil_shock_study - no ticket.")
    )


def fetch_daily_with_today(symbol: str) -> pd.DataFrame:
    """Recent daily bars via Yahoo v8 chart; last row is the live partial bar.

    Local copy of the intraday_precheck pattern: quote_service rejects
    symbols containing '=' or '^', the chart endpoint takes any string.
    """
    resp = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={"range": "3mo", "interval": "1d", "includePrePost": "false"},
        headers=UA,
        timeout=20,
    )
    resp.raise_for_status()
    res = resp.json()["chart"]["result"][0]
    quote = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(res["timestamp"], unit="s", utc=True),
            "close": quote["close"],
        }
    ).dropna()
    df["et_date"] = df["ts"].dt.tz_convert(ZoneInfo("America/New_York")).dt.date
    return df


def run_intraday(dry_run: bool) -> int:
    state = load_state()
    alerts: list[str] = []
    try:
        live = fetch_daily_with_today(SHOCK_SYMBOL)
    except Exception as exc:
        print(f"WARN {SHOCK_SYMBOL}: live fetch failed ({exc}); offline is not an error")
        return 0
    if len(live) < 2:
        print("not enough live bars; skipping")
        return 0
    bar_date = str(live["et_date"].iloc[-1])
    price = float(live["close"].iloc[-1])
    ret1 = price / float(live["close"].iloc[-2]) - 1

    suffix = ""
    try:
        confirm = fetch_daily_with_today(CONFIRM_SYMBOL)
        if len(confirm) >= 2:
            cret = float(confirm["close"].iloc[-1] / confirm["close"].iloc[-2] - 1)
            suffix = f" ({CONFIRM_SYMBOL} {cret:+.1%})"
    except Exception:
        pass

    print(f"{SHOCK_SYMBOL} live {price:.2f} ({ret1:+.2%} vs prior close), phase {state['phase']}")

    if state["phase"] in ("IDLE",) and ret1 >= SHOCK_DAY_RETURN:
        key = f"ISHOCK:{bar_date}"
        if key not in state["intraday"]:
            state["intraday"][key] = 1
            event = {"kind": "DAY", "ret": ret1}
            alerts.append(shock_alert_text(event, suffix, geopolitical_headlines(), True))
    elif state["phase"] == "WATCHING":
        ep = state["episode"]
        high = max(float(ep["post_shock_high"]), price)
        pullback = price / high - 1
        if pullback <= -DIP_PULLBACK:
            key = f"IDIP:{bar_date}"
            if key not in state["intraday"]:
                state["intraday"][key] = 1
                event = {
                    "pullback": pullback,
                    "session": ep["sessions"],
                    "shock_date": ep["shock_date"],
                    "shock_ret": ep["shock_ret"],
                }
                alerts.append(dip_alert_text(event, True))

    if alerts and not dry_run:
        notify(alerts)
        save_state(state)
        LOG_PATH.parent.mkdir(exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            stamp = datetime.now(UTC).isoformat(timespec="seconds")
            for alert in alerts:
                fh.write(f"{stamp} INTRADAY {alert}\n")
        print("\n".join(alerts))
    elif alerts:
        print("[dry-run] would send:\n" + "\n".join(alerts))
    else:
        print("no intraday oil alerts")
    return 0


def run_eod(dry_run: bool) -> int:
    state = load_state()
    daily = {}
    for symbol in (SHOCK_SYMBOL, CONFIRM_SYMBOL, VOL_SYMBOL):
        try:
            daily[symbol] = load(symbol)
        except Exception as exc:
            print(f"WARN {symbol}: load failed ({exc})")
    if SHOCK_SYMBOL not in daily:
        print(f"{SHOCK_SYMBOL} missing from the catalog; run `edgestack oil refresh`")
        return 1
    df = daily[SHOCK_SYMBOL]
    closes = df["close"]

    last_done = state.get("last_session")
    todo = [d for d in df.index if last_done is None or str(d.date()) > last_done]
    if last_done is None:
        # first ever run: replay the trailing window so an already-running
        # surge regime is reconstructed instead of missed
        todo = todo[-(REGIME_SESSIONS + SHOCK_WINDOW_SESSIONS) :]
    # alert only on events from the freshest sessions; older ones (backfill
    # after downtime) rebuild state silently and land in the log
    fresh = {str(ts.date()) for ts in df.index[-2:]}

    lines: list[str] = []
    alerts: list[str] = []
    dip_signal_dates: list[str] = []
    age = (date.today() - df.index[-1].date()).days
    if age > 5:
        lines.append(f"WARN {SHOCK_SYMBOL}: last bar {df.index[-1].date()} is {age}d old - stale")
    for ts in todo:
        upto = closes.loc[:ts]
        day = str(ts.date())
        for event in step_session(state, day, upto):
            if event["type"] == "SHOCK":
                if day in fresh:
                    suffix = context_suffix(daily)
                    alerts.append(shock_alert_text(event, suffix, geopolitical_headlines(), False))
                else:
                    lines.append(f"{day} SHOCK {event['ret']:+.1%} (backfill, no alert)")
            elif event["type"] == "RESHOCK":
                lines.append(f"{day} RESHOCK {event['ret']:+.1%} - regime extended")
            elif event["type"] == "DIP":
                if day in fresh:
                    alerts.append(dip_alert_text(event, False))
                    dip_signal_dates.append(day)
                else:
                    lines.append(f"{day} DIP {event['pullback']:+.1%} (backfill, no alert)")
            elif event["type"] == "EXPIRED":
                lines.append(f"{day} regime expired ({event['reason']}); cooldown")

    if state["phase"] == "WATCHING":
        ep = state["episode"]
        lines.append(
            f"WATCHING session {ep['sessions']}/{REGIME_SESSIONS} since shock "
            f"{ep['shock_date']} ({ep['shock_ret']:+.1%}); post-shock high "
            f"{ep['post_shock_high']:.2f}; close {closes.iloc[-1]:.2f}"
        )
    else:
        lines.append(f"phase {state['phase']}; close {closes.iloc[-1]:.2f}")

    # prune intraday dedupe keys older than ~2 weeks
    cutoff = (date.today() - timedelta(days=14)).isoformat()
    state["intraday"] = {k: v for k, v in state["intraday"].items() if k[-10:] >= cutoff}

    lines += paper_autopilot(dip_signal_dates)

    for line in lines:
        print(line)
    if alerts and not dry_run:
        print("\n" + "\n".join(alerts))
        notify(alerts)
    elif alerts:
        print("[dry-run] would send:\n" + "\n".join(alerts))
    else:
        print("no new oil alerts")
    if not dry_run:
        save_state(state)
        LOG_PATH.parent.mkdir(exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"===== oil_surge_watch EOD {date.today().isoformat()} =====\n")
            for line in lines + alerts:
                fh.write(line + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--intraday", action="store_true", help="30-min live poll (Task Scheduler)")
    mode.add_argument("--eod", action="store_true", help="nightly pass on curated closes")
    parser.add_argument("--dry-run", action="store_true", help="evaluate but never send/persist")
    args = parser.parse_args()
    if args.intraday:
        return run_intraday(args.dry_run)
    return run_eod(args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
