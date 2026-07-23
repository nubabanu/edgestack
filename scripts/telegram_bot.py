"""Read-only Telegram command bot for the Depot Alerts chat.

Long-polls getUpdates and answers slash commands with the same artifacts the
watchers publish. Kept alive by scripts/telegram_bot.bat (watchdog loop) via
a Windows Startup entry, mirroring the API server pattern.

Commands: /status (data + nightly health), /watch (tranche triggers),
/oil (oil surge state), /paper (paper books), /help.

Honest caveats:
- READ-ONLY. The bot never places orders, never mutates watcher state, and
  its replies are display-only research output, not investment advice.
- Auth is per-chat: only messages from the chat id(s) in
  artifacts/telegram_config.json get answers; everything else is silently
  ignored (the offset still advances). In a group chat that means anyone in
  the group can query — acceptable for the two-member Depot Alerts group.
- Only ONE getUpdates consumer per bot token may run. A second instance (for
  example a manual debug run beside the Startup one) causes HTTP 409; the
  loop backs off 60 s and retries. Stop the Startup instance before running
  --once by hand. notify_telegram.py only calls sendMessage, never conflicts.
- Replies are plain ASCII (cp1252 console/log discipline), max 4000 chars.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

CONFIG_PATH = ROOT / "artifacts" / "telegram_config.json"  # shared with notify_telegram.py
STATE_PATH = ROOT / "artifacts" / "telegram_bot_state.json"
WATCH_PATH = ROOT / "artifacts" / "tranche_watch.json"
OIL_STATE_PATH = ROOT / "artifacts" / "oil_surge_state.json"
SWING_ZONES_PATH = ROOT / "artifacts" / "swing_zones.json"
OIL_VERDICT_PATH = ROOT / "artifacts" / "oil_shock_study_verdict.json"
PAPER_TRANCHE_PATH = ROOT / "artifacts" / "paper_tranche.json"
PAPER_OIL_PATH = ROOT / "artifacts" / "paper_oil.json"
PRICES_DIR = ROOT / "data" / "curated" / "prices"

POLL_TIMEOUT_S = 50
CONFLICT_BACKOFF_S = 60
ERROR_BACKOFF_S = 15
MAX_REPLY_CHARS = 4000
TRIGGER_KEYS = ("T1", "T2", "T3", "REL")


def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")[:MAX_REPLY_CHARS]


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        return None


def _age_line(path: Path) -> str:
    if not path.exists():
        return "never updated"
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    hours = (datetime.now(UTC) - mtime).total_seconds() / 3600
    stamp = mtime.astimezone().strftime("%Y-%m-%d %H:%M")
    return f"updated {stamp} ({hours:.0f}h ago)" if hours >= 1 else f"updated {stamp} (<1h ago)"


def load_config() -> dict | None:
    return _read_json(CONFIG_PATH)


def allowed_chat_ids(cfg: dict) -> frozenset[int]:
    raw = cfg.get("chat_ids") or [cfg.get("chat_id")]
    return frozenset(int(cid) for cid in raw if cid is not None)


def load_offset() -> int:
    state = _read_json(STATE_PATH) or {}
    return int(state.get("offset", 0))


def save_offset(offset: int) -> None:
    from edgestack.data.catalog import atomic_write_bytes

    STATE_PATH.parent.mkdir(exist_ok=True)
    atomic_write_bytes(STATE_PATH, json.dumps({"offset": offset}, indent=2).encode("utf-8"))


def _last_close(symbol: str) -> tuple[str, float] | None:
    import pandas as pd

    path = PRICES_DIR / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path, columns=["date", "close"])
        last = frame.iloc[-1]
        return str(pd.Timestamp(last["date"]).date()), round(float(last["close"]), 2)
    except (OSError, ValueError, KeyError, IndexError):
        return None


def fmt_status() -> str:
    from agent_toolkit import build_status_report

    report = build_status_report()
    lines = [f"STATUS: {report.get('health', 'unknown')} (as of {report.get('today')})"]
    for issue in report.get("issues", []):
        lines.append(f"! {issue.get('code')}: {issue.get('message')}")
    pub = report.get("publication") or {}
    if pub:
        lines.append(
            f"publication: session {pub.get('session')} "
            f"({'STALE' if pub.get('stale') else 'fresh'}, {pub.get('age_hours', '?')}h old)"
        )
    market = report.get("market_data") or {}
    stale_syms = [s for s, m in (market.get("symbols") or {}).items() if m.get("stale")]
    lines.append("data: " + (f"STALE {','.join(stale_syms)}" if stale_syms else "all bars fresh"))
    tranche = report.get("tranche_watch") or {}
    fired = {s: t for s, t in (tranche.get("fired") or {}).items() if t}
    lines.append(
        f"tranche watch: {tranche.get('status', '?')} run {tranche.get('run_date', '?')}"
        + (f"; fired {fired}" if fired else "; nothing fired")
    )
    oil = _read_json(OIL_STATE_PATH) or {}
    lines.append(f"oil surge: phase {oil.get('phase', 'never ran')}")
    return _ascii("\n".join(lines))


def fmt_watch() -> str:
    payload = _read_json(WATCH_PATH)
    if payload is None:
        return "no tranche watcher status yet (artifacts/tranche_watch.json missing)"
    lines = [f"TRANCHE WATCH run {payload.get('run_date', '?')} ({_age_line(WATCH_PATH)})"]
    breadth = payload.get("breadth") or {}
    if breadth:
        lines.append(
            f"breadth {breadth.get('count')}/{breadth.get('total')} ({breadth.get('names', '')})"
        )
    for status in payload.get("symbols", []):
        sym = status.get("symbol", "?")
        fired = []
        for key in TRIGGER_KEYS:
            info = status.get(key) or {}
            if info.get("fired"):
                fired.append(f"{key} [{info.get('detail', '')}]")
        if status.get("CAL"):
            fired.append(f"CAL [{status['CAL']}]")
        if status.get("window_open"):
            fired.append("WINDOW open")
        state = "; ".join(fired) if fired else "waiting"
        go = status.get("GO")
        go_note = f" | GO {go}/100 (display-only, failed backtest)" if go is not None else ""
        lines.append(f"{sym} close {status.get('close', '?')}: {state}{go_note}")
    return _ascii("\n".join(lines))


def fmt_oil() -> str:
    state = _read_json(OIL_STATE_PATH)
    if state is None:
        return "no oil surge state yet (run scripts/oil_surge_watch.py --eod)"
    lines = [f"OIL SURGE phase {state.get('phase', '?')} ({_age_line(OIL_STATE_PATH)})"]
    episode = state.get("episode")
    if episode:
        lines.append(
            f"shock {episode.get('shock_date')} ({episode.get('shock_ret', 0):+.1%}), "
            f"session {episode.get('sessions')}/10, post-shock high "
            f"{episode.get('post_shock_high', 0):.2f}, dips fired {episode.get('dips', 0)}"
        )
    for symbol in ("CL=F", "BNO"):
        close = _last_close(symbol)
        if close:
            lines.append(f"{symbol}: {close[1]} ({close[0]})")
    verdict = _read_json(OIL_VERDICT_PATH)
    if verdict:
        lines.append(
            f"study verdict: {verdict.get('verdict')} - dip alerts "
            + ("TICKETED" if state.get("tickets_enabled") else "DISPLAY-ONLY")
        )
    else:
        lines.append("study not run yet - dip alerts DISPLAY-ONLY")
    lines.append("Read-only research context, not investment advice.")
    return _ascii("\n".join(lines))


def _book_lines(name: str, path: Path) -> list[str]:
    book = _read_json(path)
    if not book or not book.get("trades"):
        return [f"{name}: no paper trades"]
    lines = [f"{name}:"]
    cost = value = 0.0
    for tr in book["trades"]:
        if tr.get("fill_price"):
            close = _last_close(tr["symbol"])
            last = close[1] if close else tr["fill_price"]
            pnl = tr["shares"] * last - tr["eur"]
            cost += tr["eur"]
            value += tr["shares"] * last
            lines.append(
                f"  {tr['symbol']} {tr.get('trigger', '?')} EUR {tr['eur']} "
                f"@ {tr['fill_price']} -> {pnl:+.0f} EUR"
            )
        else:
            lines.append(
                f"  {tr['symbol']} {tr.get('trigger', '?')} EUR {tr['eur']} (pending fill)"
            )
    if cost:
        lines.append(f"  total: cost {cost:.0f} value {value:.0f} ({value / cost - 1:+.1%})")
    return lines


def fmt_paper() -> str:
    lines = _book_lines("TRANCHE BOOK", PAPER_TRANCHE_PATH)
    lines += _book_lines("OIL BOOK", PAPER_OIL_PATH)
    lines.append("Paper autopilot only - no live orders ever (repo covenant).")
    return _ascii("\n".join(lines))


def fmt_zones() -> str:
    zones = _read_json(SWING_ZONES_PATH)
    if not zones or not zones.get("entries"):
        return "no swing zone watchlist yet (run scripts/swing_cycle_scan.py)"
    lines = [f"SWING BUY ZONES ({_age_line(SWING_ZONES_PATH)})"]
    for entry in zones["entries"]:
        sym = entry["symbol"]
        close = _last_close(sym)
        if close is None:
            continue
        dip = float(entry["trough_zone"])
        buy_at = dip * 1.05  # tested entry level
        stop_at = dip * 0.85
        price = close[1]
        if stop_at <= price <= buy_at:
            status = "BUY WINDOW NOW"
        elif price < stop_at:
            status = "zone broken - do not buy"
        else:
            status = f"waiting ({price / buy_at - 1:+.0%} above buy level)"
        lines.append(f"{sym}: {price} | buy <= {buy_at:.2f} | {status}")
    lines.append(
        "Levels refresh nightly. Edge is small and unproven live - no guarantees, "
        "your decision. Full playbook arrives as a BUY WINDOW alert when triggered."
    )
    return _ascii("\n".join(lines))


def fmt_help() -> str:
    return _ascii(
        "EdgeStack bot (read-only, never places orders):\n"
        "/status - data + nightly pipeline health\n"
        "/watch - tranche trigger status (ACN/CTSH/EPAM/SPY)\n"
        "/oil - oil surge watcher state + dip watch\n"
        "/zones - swing buy zones: is it buy time yet, and at what price\n"
        "/paper - paper trade books + P&L\n"
        "/help - this list"
    )


def route(text: str) -> str | None:
    """Map an incoming message to a reply; None = stay silent (non-commands)."""
    cmd = text.strip().split()[0].lower() if text.strip() else ""
    if not cmd.startswith("/"):
        return None
    cmd = cmd.split("@")[0]  # groups deliver /watch@BotName
    handlers = {
        "/status": fmt_status,
        "/watch": fmt_watch,
        "/oil": fmt_oil,
        "/zones": fmt_zones,
        "/paper": fmt_paper,
        "/help": fmt_help,
        "/start": fmt_help,
    }
    handler = handlers.get(cmd, fmt_help)
    try:
        return handler()
    except Exception as exc:
        return _ascii(f"WARN: {cmd} failed ({type(exc).__name__}: {exc})")


def handle_update(update: dict, allowed: frozenset[int]) -> tuple[int, str] | None:
    """(chat_id, reply) for an authorized command, else None. Silent on strangers."""
    message = update.get("message") or update.get("edited_message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    text = message.get("text")
    if chat_id is None or chat_id not in allowed or not text:
        return None
    reply = route(text)
    return (chat_id, reply) if reply else None


def fetch_updates(cfg: dict, offset: int) -> tuple[list[dict], bool]:
    """(updates, conflict). conflict=True means another poller holds the token."""
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{cfg['bot_token']}/getUpdates",
            params={"offset": offset, "timeout": POLL_TIMEOUT_S},
            timeout=POLL_TIMEOUT_S + 15,
        )
        if resp.status_code == 409:
            return [], True
        resp.raise_for_status()
        body = resp.json()
        return (body.get("result", []) if body.get("ok") else []), False
    except requests.exceptions.RequestException as exc:
        print(f"WARN getUpdates failed: {exc}")
        return [], False


def send_reply(cfg: dict, chat_id: int, text: str) -> None:
    try:
        requests.post(
            f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage",
            json={"chat_id": chat_id, "text": text[:MAX_REPLY_CHARS]},
            timeout=15,
        ).raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"WARN sendMessage failed: {exc}")


def poll_once(cfg: dict, allowed: frozenset[int]) -> bool:
    """One getUpdates pass; True if a conflict backoff is needed."""
    offset = load_offset()
    updates, conflict = fetch_updates(cfg, offset)
    if conflict:
        return True
    max_id = offset - 1
    for update in updates:
        max_id = max(max_id, int(update.get("update_id", max_id)))
        try:
            result = handle_update(update, allowed)
        except Exception as exc:
            print(f"WARN update handling failed: {exc}")
            continue
        if result:
            send_reply(cfg, *result)
    if updates:
        save_offset(max_id + 1)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", metavar="CMD", help="print the reply for CMD, no network")
    parser.add_argument("--once", action="store_true", help="single getUpdates pass, then exit")
    args = parser.parse_args()

    if args.dry_run:
        print(route(args.dry_run) or "(no reply - not a command)")
        return 0

    cfg = load_config()
    if cfg is None or not cfg.get("bot_token"):
        print(f"WARN no telegram config at {CONFIG_PATH}; sleeping before exit")
        time.sleep(300)
        return 1
    allowed = allowed_chat_ids(cfg)
    if not allowed:
        print("WARN no chat_id/chat_ids configured; sleeping before exit")
        time.sleep(300)
        return 1
    print(f"telegram bot up; answering {len(allowed)} chat(s); long-poll {POLL_TIMEOUT_S}s")

    while True:
        try:
            conflict = poll_once(cfg, allowed)
        except Exception as exc:
            print(f"WARN poll loop error: {exc}")
            time.sleep(ERROR_BACKOFF_S)
            continue
        if conflict:
            print("WARN getUpdates 409 conflict (another poller?); backing off")
            time.sleep(CONFLICT_BACKOFF_S)
        if args.once:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
