"""Nightly ACN/CTSH tranche-entry watcher.

Watches the staged-entry triggers from the 2026-07 buy-timing research
(see docs/strategy-zoo.md caveats — historical research, not advice):

  T1 dip     ACN: RSI(2) < 10.  CTSH: 3 consecutive down days OR IBS < 0.2.
             Suppressed within 3 calendar days before a known earnings date
             (a dip into the print is an earnings bet, not mean reversion).
  T2 repair  2 of 3 held for 5 straight sessions:
             close > SMA50, MACD line > signal, 20d ann. vol < 30%.
  T3 trend   close > SMA200.
  REL        ratio to the equal-weight IT-services peer basket: 60d z-score
             crosses up through -1 (the best non-calendar ACN trigger).
  SECTOR     peer-basket breadth (share of names above SMA50) crosses from
             minority to >= 4 of 6 — sector-wide repair confirmation.
  CAL        first 3 sessions of November (both), Dec sessions 10-15 (CTSH),
             late June (both).

Exit half: positions recorded in artifacts/tranche_positions.json get
REVIEW alerts (not sell signals) when a repair/trend entry loses the SMA50
for 5 sessions, relative strength makes a 120d low, vol re-spikes above
50%, or the position is down 20% from entry.

Reads bars the nightly orchestrator already refreshed in
data/curated/prices/. Appends to logs/tranche_watch.log, publishes status
to artifacts/tranche_watch.json and a dashboard to
artifacts/tranche_status.html, fires a Windows toast when anything newly
triggers (per-trigger cooldowns prevent nightly re-alerting).
"""

from __future__ import annotations

import contextlib
import html
import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from edgestack.data.catalog import atomic_write_bytes

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "data" / "curated" / "prices"
STATE_PATH = ROOT / "artifacts" / "tranche_watch_state.json"
STATUS_PATH = ROOT / "artifacts" / "tranche_watch.json"
POSITIONS_PATH = ROOT / "artifacts" / "tranche_positions.json"
EARNINGS_CACHE = ROOT / "artifacts" / "earnings_cache.json"
DASHBOARD_PATH = ROOT / "artifacts" / "tranche_status.html"
LOG_PATH = ROOT / "logs" / "tranche_watch.log"
TOAST_SCRIPT = ROOT / "scripts" / "notify_toast.ps1"

SYMBOLS = ("ACN", "CTSH", "EPAM", "SPY")  # SPY: calm+dip entries for the core
BASKET = ("ACN", "CTSH", "EPAM", "DXC", "IBM", "IT")
STALE_AFTER_DAYS = 5
EARNINGS_BLACKOUT_DAYS = 3
EARNINGS_REFRESH_DAYS = 7
COOLDOWN_SESSIONS = {"T1": 5, "T2": 21, "T3": 21, "REL": 21, "REVIEW": 10, "GO": 5}
GO_ALERT_THRESHOLD = 60
# `agent_toolkit.py go-backtest` verdict 2026-07-19: FAIL on all three names
# (60+ bucket does not beat unconditional forward returns; CTSH 60+ negative).
# The composite is display-only context; the individual validated triggers
# (T1/T2/T3/REL/WINDOW) remain the actionable alerts. Do not enable without a
# fresh go-backtest PASS.
GO_ALERTS_ENABLED = False
POST_EARNINGS_WINDOW_SESSIONS = 5
PUBLIC_TRIGGER_TYPES = (
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
POSITIONS_TEMPLATE = {
    "_howto": (
        "Record each actual buy here so the watcher can run exit/review checks. "
        'Example entry: {"symbol": "ACN", "tranche": "T2", "date": "2026-08-03", '
        '"price": 161.50}. tranche is one of T1/T2/T3/CAL. Remove an entry when '
        "the position is closed."
    ),
    "positions": [],
}
PLAN_PATH = ROOT / "artifacts" / "tranche_plan.json"
PAPER_PATH = ROOT / "artifacts" / "paper_tranche.json"
# Per-symbol euro sizes per trigger; alerts turn these into ready order tickets
# and the paper autopilot executes them at the actual next open. Edit freely.
PLAN_TEMPLATE = {
    "_howto": (
        "Euro amounts the alerts should turn into order tickets, keyed by "
        "symbol then trigger (T1/T2/T3/REL/CAL/WINDOW). Missing keys = no "
        "ticket, alert only. The paper autopilot fills every ticketed alert "
        "at the next session's open automatically."
    ),
    "CTSH": {"T1": 300, "T3": 200, "CAL": 100},
    "EPAM": {"T1": 200, "T3": 100, "CAL": 100},
    "SPY": {"T1": 500},
}


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def load(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(PRICES / f"{symbol}.parquet")
    return (
        df.assign(date=pd.to_datetime(df["date"]))
        .set_index("date")
        .sort_index()[["open", "high", "low", "close", "adj_close"]]
    )


_YAHOO_SESSION: requests.Session | None = None
_YAHOO_CRUMB: str = ""


def _yahoo_session() -> tuple[requests.Session, str]:
    """Cookie+crumb session Yahoo's quoteSummary endpoint requires."""
    global _YAHOO_SESSION, _YAHOO_CRUMB
    if _YAHOO_SESSION is None:
        s = requests.Session()
        s.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        # Any response (even an error page) sets the cookie.
        with contextlib.suppress(requests.exceptions.RequestException):
            s.get("https://fc.yahoo.com", timeout=10)
        _YAHOO_CRUMB = s.get(
            "https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=10
        ).text.strip()
        _YAHOO_SESSION = s
    return _YAHOO_SESSION, _YAHOO_CRUMB


def next_earnings(symbol: str, today: date) -> str | None:
    """Next earnings date via Yahoo calendarEvents, cached; None if unknown."""
    cache = json.loads(EARNINGS_CACHE.read_text()) if EARNINGS_CACHE.exists() else {}
    entry = cache.get(symbol)
    if entry and (today - date.fromisoformat(entry["fetched"])).days < EARNINGS_REFRESH_DAYS:
        return entry["date"]
    try:
        session, crumb = _yahoo_session()
        resp = session.get(
            f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}",
            params={"modules": "calendarEvents", "crumb": crumb},
            timeout=15,
        )
        resp.raise_for_status()
        result = resp.json()["quoteSummary"]["result"][0]["calendarEvents"]
        stamps = result.get("earnings", {}).get("earningsDate", [])
        dates = [date.fromtimestamp(s["raw"]).isoformat() for s in stamps if "raw" in s]
        earn = min((d for d in dates if d >= today.isoformat()), default=None)
    except Exception as exc:
        print(f"WARN {symbol}: earnings fetch failed ({exc}); keeping stale cache")
        return entry["date"] if entry else None
    cache[symbol] = {"date": earn, "fetched": today.isoformat()}
    EARNINGS_CACHE.write_text(json.dumps(cache, indent=2))
    return earn


def basket_series() -> tuple[pd.Series, pd.DataFrame]:
    """Equal-weight peer-basket adj level and per-name close>SMA50 flags."""
    adjs, above = {}, {}
    for sym in BASKET:
        df = load(sym)
        adjs[sym] = df["adj_close"]
        above[sym] = df["close"] > df["close"].rolling(50).mean()
    adj = pd.DataFrame(adjs).dropna()
    level = (adj / adj.iloc[0]).mean(axis=1)
    return level, pd.DataFrame(above).dropna()


def rel_zscore(adj: pd.Series, basket: pd.Series) -> pd.Series:
    ratio = (adj / basket).dropna()
    return (ratio - ratio.rolling(60).mean()) / ratio.rolling(60).std()


def evaluate(symbol: str, df: pd.DataFrame, basket: pd.Series, earnings: str | None) -> dict:
    c, h, low = df["close"], df["high"], df["low"]
    ret = df["adj_close"].pct_change()
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    r2 = _rsi(c, 2)
    ibs = ((c - low) / (h - low).replace(0, np.nan)).fillna(0.5)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    macd = _ema(c, 12) - _ema(c, 26)
    macd_sig = _ema(macd, 9)
    rz = rel_zscore(df["adj_close"], basket)

    down3 = (ret < 0) & (ret.shift() < 0) & (ret.shift(2) < 0)
    if symbol == "SPY":
        # validated index rule: any of RSI2<10 / 3 down days / weak close,
        # but only inside the calm regime (>200DMA, vol<30%) — the entry
        # with the best compound growth in the whole research program.
        in_calm = bool(c.iloc[-1] > s200.iloc[-1]) and bool(vol20.iloc[-1] < 0.30)
        dip_now = bool(r2.iloc[-1] < 10) or bool(down3.iloc[-1]) or bool(ibs.iloc[-1] < 0.2)
        t1 = in_calm and dip_now
        t1_detail = (
            f"calm={in_calm} RSI2={r2.iloc[-1]:.0f} down3={bool(down3.iloc[-1])} "
            f"IBS={ibs.iloc[-1]:.2f} (calm AND any-dip fires)"
        )
    elif symbol == "ACN":
        t1 = bool(r2.iloc[-1] < 10)
        t1_detail = f"RSI2={r2.iloc[-1]:.0f} (<10 fires)"
    else:
        t1 = bool(down3.iloc[-1]) or bool(ibs.iloc[-1] < 0.2)
        t1_detail = f"down3={bool(down3.iloc[-1])} IBS={ibs.iloc[-1]:.2f} (<0.2 fires)"
    t1_suppressed = False
    if t1 and earnings:
        days_to = (date.fromisoformat(earnings) - df.index[-1].date()).days
        if 0 <= days_to <= EARNINGS_BLACKOUT_DAYS:
            t1_suppressed = True
            t1_detail += f" [SUPPRESSED: earnings {earnings}]"

    repair = (
        (c > s50).astype(int) + (macd > macd_sig).astype(int) + (vol20 < 0.30).astype(int)
    ) >= 2
    t2 = bool(repair.iloc[-5:].all())
    t2_detail = (
        f"c>SMA50={bool((c > s50).iloc[-1])} macd_up={bool((macd > macd_sig).iloc[-1])} "
        f"vol20={vol20.iloc[-1]:.0%} (2of3 for 5 sessions fires)"
    )
    if t2:
        # Cohort study 2026-07: in idiosyncratic -40% episodes the first T2
        # firing was followed by a NEW low 61% of the time (median +0.4%
        # fwd 6m). Half-weight it or wait for REL/SECTOR co-confirmation.
        t2_detail += " [CAUTION: 61% historical whipsaw rate in this regime]"

    t3 = bool((c > s200).iloc[-1])
    if symbol == "SPY":
        # For the index, above-200DMA is the steady state; alert only on the
        # actual reclaim cross (for the crashed names, "still above" is news).
        t3 = t3 and not bool((c > s200).iloc[-2])
    t3_detail = (
        f"close={c.iloc[-1]:.2f} SMA200={s200.iloc[-1]:.2f} ({c.iloc[-1] / s200.iloc[-1] - 1:+.1%})"
    )

    rel = bool(len(rz) > 2 and rz.iloc[-1] > -1 and rz.iloc[-2] <= -1)
    rel_detail = f"rel-z={rz.iloc[-1]:+.2f} (up-cross of -1 fires)"

    idx = df.index
    tdom = (
        int(pd.Series(range(len(idx)), index=idx).groupby(idx.to_period("M")).cumcount().iloc[-1])
        + 1
    )
    last = idx[-1]
    cal = None
    if last.month == 11 and tdom <= 3:
        cal = f"November window session {tdom} — seasonal add window (both names)"
    elif symbol == "CTSH" and last.month == 12 and 10 <= tdom <= 15:
        cal = f"Mid-December tax-loss window session {tdom}"
    elif last.month == 6 and last.day >= 23:
        cal = "Late-June window — July seasonal starts next sessions"

    return {
        "symbol": symbol,
        "as_of": str(last.date()),
        "close": round(float(c.iloc[-1]), 2),
        "sma50": round(float(s50.iloc[-1]), 2),
        "sma200": round(float(s200.iloc[-1]), 2),
        "vol20": round(float(vol20.iloc[-1]), 3),
        "rel_z": round(float(rz.iloc[-1]), 2),
        "rsi2": round(float(r2.iloc[-1]), 1),
        "ibs": round(float(ibs.iloc[-1]), 2),
        "calm": bool((c.iloc[-1] > s200.iloc[-1]) and (vol20.iloc[-1] < 0.30)),
        "earnings": earnings,
        "T1": {"fired": t1 and not t1_suppressed, "detail": t1_detail},
        "T2": {"fired": t2, "detail": t2_detail},
        "T3": {"fired": t3, "detail": t3_detail},
        "REL": {"fired": rel, "detail": rel_detail},
        "CAL": cal,
        "_series": {"c": c, "s50": s50, "rz": rz, "vol20": vol20},
    }


def review_positions(statuses: dict[str, dict]) -> list[tuple[str, str]]:
    """(key, message) review alerts for recorded positions."""
    if not POSITIONS_PATH.exists():
        POSITIONS_PATH.write_text(json.dumps(POSITIONS_TEMPLATE, indent=2))
        return []
    positions = json.loads(POSITIONS_PATH.read_text()).get("positions", [])
    out: list[tuple[str, str]] = []
    for pos in positions:
        sym = pos.get("symbol")
        if sym not in statuses:
            continue
        s = statuses[sym]["_series"]
        entry_price = float(pos.get("price", 0)) or None
        tranche = pos.get("tranche", "?")
        tag = f"{sym} {tranche}@{pos.get('date', '?')}"
        below50 = (s["c"] < s["s50"]).iloc[-5:].all()
        if tranche in ("T2", "T3") and bool(below50):
            out.append((f"REVIEW:{sym}:{tranche}:sma50", f"{tag}: lost SMA50 for 5 sessions"))
        if len(s["rz"]) > 120 and s["rz"].iloc[-1] <= s["rz"].iloc[-120:].min():
            out.append((f"REVIEW:{sym}:rellow", f"{tag}: relative strength at 120d low"))
        if float(s["vol20"].iloc[-1]) > 0.50:
            out.append(
                (f"REVIEW:{sym}:vol", f"{tag}: vol20 {s['vol20'].iloc[-1]:.0%} > 50% — re-spike")
            )
        if entry_price and float(s["c"].iloc[-1]) < 0.80 * entry_price:
            out.append(
                (
                    f"REVIEW:{sym}:loss20",
                    f"{tag}: down {s['c'].iloc[-1] / entry_price - 1:.0%} from entry",
                )
            )
    return out


def load_plan() -> dict:
    if not PLAN_PATH.exists():
        PLAN_PATH.write_text(json.dumps(PLAN_TEMPLATE, indent=2))
    return json.loads(PLAN_PATH.read_text())


def order_ticket(plan: dict, symbol: str, trig: str) -> str:
    eur = plan.get(symbol, {}).get(trig)
    # ASCII arrow: the nightly console/log is cp1252 and chokes on U+2192
    return f" -> ORDER: buy ~EUR {eur} {symbol} at next open" if eur else ""


def paper_autopilot(plan: dict, fired: list[tuple[str, str, str]]) -> list[str]:
    """Automatic PAPER execution of ticketed alerts (research only, no live
    orders — the repo covenant). Intents recorded when a ticketed trigger
    fires; filled at the actual next session's adjusted open on a later run;
    running P&L reported every night."""
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
                    f"PAPER filled {tr['symbol']} {tr['trigger']} €{tr['eur']} "
                    f"@ {aopen:.2f} ({tr['fill_date']})"
                )
    for symbol, trig, sig_date in fired:
        eur = plan.get(symbol, {}).get(trig)
        if not eur:
            continue
        if any(
            t["symbol"] == symbol and t["trigger"] == trig and t["signal_date"] == sig_date
            for t in book["trades"]
        ):
            continue
        book["trades"].append(
            {
                "symbol": symbol,
                "trigger": trig,
                "eur": eur,
                "signal_date": sig_date,
                "fill_price": None,
            }
        )
        lines.append(f"PAPER intent {symbol} {trig} €{eur} (fills at next open)")
    cost = value = 0.0
    for tr in book["trades"]:
        if tr.get("fill_price"):
            last = float(load(tr["symbol"])["adj_close"].iloc[-1])
            cost += tr["eur"]
            value += tr["shares"] * last
    if cost:
        lines.append(f"PAPER book: cost €{cost:.0f} value €{value:.0f} ({value / cost - 1:+.1%})")
    book["_note"] = "Automatic paper execution of ticketed alerts; research only, no live orders."
    PAPER_PATH.write_text(json.dumps(book, indent=2))
    return lines


def notify(alerts: list[str]) -> None:
    """Push alerts to every configured channel: Windows toast + Telegram."""
    body = "; ".join(a.removeprefix("ALERT ") for a in alerts)[:250]
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
                "EdgeStack tranche alert",
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

        notify_telegram.send("EdgeStack tranche alert\n" + "\n".join(alerts)[:3800])
    except Exception as exc:
        print(f"WARN telegram failed: {exc}")


def go_score(
    *,
    t1: bool,
    near_dip: bool,
    calm: bool,
    low_vol: bool,
    rel_recross: bool,
    rel_ok: bool,
    breadth: int,
    seasonal: bool,
    post_earnings: bool,
) -> int:
    """0-100 composite entry score. Weights are HAND-SET from the measured
    effects in this repo's research (docs/strategy-zoo.md and the 2026-07
    buy-timing studies) — deliberately not fitted, to avoid overfitting.
    Validated by `agent_toolkit.py go-backtest` before the alert threshold
    was enabled."""
    score = 0
    score += 30 if t1 else (12 if near_dip else 0)  # dip freshness
    score += 20 if calm else (8 if low_vol else 0)  # regime
    score += 15 if rel_recross else (7 if rel_ok else 0)  # relative strength
    score += 10 if breadth >= 4 else (5 if breadth >= 2 else 0)  # peers
    score += 10 if seasonal else 0  # calendar window
    score += 15 if post_earnings else 0  # event uncertainty resolved
    return score


def write_dashboard(statuses: list[dict], breadth: dict, alerts: list[str], stamp: str) -> None:
    def badge(fired: bool) -> str:
        return "<span class='b on'>FIRED</span>" if fired else "<span class='b off'>waiting</span>"

    recent = ""
    if LOG_PATH.exists():
        tail = LOG_PATH.read_text(encoding="utf-8").splitlines()[-40:]
        recent = "\n".join(html.escape(x) for x in tail if "ALERT" in x or "=====" in x)

    positions = []
    if POSITIONS_PATH.exists():
        positions = json.loads(POSITIONS_PATH.read_text()).get("positions", [])

    rows = ""
    for st in statuses:
        trig_rows = "".join(
            f"<tr><td>{label}</td><td>{badge(st[key]['fired'])}</td>"
            f"<td>{html.escape(st[key]['detail'])}</td></tr>"
            for label, key in (
                ("T1 dip", "T1"),
                ("T2 repair", "T2"),
                ("T3 trend", "T3"),
                ("REL repair", "REL"),
            )
        )
        cal_row = (
            f"<tr><td>Calendar</td><td>{badge(bool(st['CAL']))}</td>"
            f"<td>{html.escape(st['CAL'] or 'no window active')}</td></tr>"
        )
        go = st.get("GO", 0)
        go_cls = "on" if go >= GO_ALERT_THRESHOLD else "off"
        rows += f"""
      <h2>{st["symbol"]} <small>close {st["close"]} · as of {st["as_of"]}
        · earnings {st["earnings"] or "unknown"}</small>
        <span class="b {go_cls}">GO {go}/100</span></h2>
      <table>
        <tr><th>Trigger</th><th>State</th><th>Detail</th></tr>
        {trig_rows}{cal_row}
      </table>
      <p class="dist">distance to SMA50 {st["close"] / st["sma50"] - 1:+.1%} ·
         to SMA200 {st["close"] / st["sma200"] - 1:+.1%} ·
         vol20 {st["vol20"]:.0%} · rel-z {st["rel_z"]:+.2f}</p>"""

    pos_html = (
        "<ul>"
        + "".join(
            f"<li>{html.escape(str(p.get('symbol')))} {html.escape(str(p.get('tranche')))} "
            f"bought {html.escape(str(p.get('date')))} @ {p.get('price')}</li>"
            for p in positions
        )
        + "</ul>"
        if positions
        else "<p>none recorded — add buys to artifacts/tranche_positions.json</p>"
    )

    paper_html = "<p>no paper trades yet</p>"
    if PAPER_PATH.exists():
        trades = json.loads(PAPER_PATH.read_text()).get("trades", [])
        if trades:
            paper_html = (
                "<table><tr><th>Symbol</th><th>Trigger</th><th>€</th><th>Signal</th>"
                "<th>Fill</th></tr>"
                + "".join(
                    f"<tr><td>{t['symbol']}</td><td>{t['trigger']}</td><td>{t['eur']}</td>"
                    f"<td>{t['signal_date']}</td>"
                    f"<td>{t.get('fill_price') or 'pending next open'}</td></tr>"
                    for t in trades
                )
                + "</table>"
            )

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ACN/CTSH tranche watch</title><style>
body{{font-family:system-ui,sans-serif;max-width:780px;margin:2rem auto;padding:0 1rem;
     background:#fff;color:#1a1a1a}}
@media(prefers-color-scheme:dark){{body{{background:#14161a;color:#e6e6e6}}
  table,th,td{{border-color:#3a3f47!important}}pre{{background:#1e2126!important}}}}
h1{{font-size:1.3rem}}h2{{font-size:1.05rem;margin-top:1.6rem}}small{{font-weight:normal;opacity:.7}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:.35rem .5rem;
  text-align:left;font-size:.9rem}}
.b{{padding:.1rem .45rem;border-radius:.6rem;font-size:.75rem;font-weight:600}}
.on{{background:#1d7a3e;color:#fff}}.off{{background:#888;color:#fff;opacity:.55}}
.dist{{font-size:.85rem;opacity:.8}}pre{{background:#f4f4f4;padding:.7rem;overflow-x:auto;
  font-size:.78rem}}footer{{margin-top:2rem;font-size:.75rem;opacity:.6}}</style></head><body>
<h1>ACN / CTSH tranche watch <small>run {stamp}</small></h1>
<p>Peer-basket breadth (close &gt; SMA50): <b>{breadth["count"]} of {breadth["total"]}</b>
 ({html.escape(breadth["names"])})</p>
{rows}
<h2>Recorded positions</h2>{pos_html}
<h2>Paper autopilot (auto-executed tickets, research only)</h2>{paper_html}
<h2>Alerts this run</h2>
<pre>{html.escape(chr(10).join(alerts)) if alerts else "none"}</pre>
<h2>Recent alert history</h2><pre>{recent or "none yet"}</pre>
<footer>Historical pattern research on previously-accessed data — not investment advice.
Generated nightly by scripts/tranche_watch.py.</footer></body></html>"""
    DASHBOARD_PATH.write_text(doc, encoding="utf-8")


def main() -> int:
    LOG_PATH.parent.mkdir(exist_ok=True)
    STATE_PATH.parent.mkdir(exist_ok=True)
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    today = date.today()
    plan = load_plan()
    lines: list[str] = []
    alerts: list[str] = []
    statuses: list[dict] = []
    fired: list[tuple[str, str, str]] = []
    events: list[dict[str, str]] = []

    basket, above50 = basket_series()
    breadth_count = int(above50.iloc[-1].sum())
    breadth = {
        "count": breadth_count,
        "total": len(BASKET),
        "names": ", ".join(f"{s}{'+' if above50.iloc[-1][s] else '-'}" for s in BASKET),
    }
    lines.append(f"SECTOR breadth {breadth_count}/{len(BASKET)} above SMA50 ({breadth['names']})")
    prev_breadth = state.get("sector_breadth", breadth_count)
    if breadth_count >= 4 and prev_breadth < 4:
        alerts.append(f"ALERT SECTOR: breadth crossed to {breadth_count}/{len(BASKET)} above SMA50")
        events.append(
            {
                "symbol": "BASKET",
                "trigger": "SECTOR",
                "as_of": today.isoformat(),
                "detail": f"breadth crossed to {breadth_count}/{len(BASKET)} above SMA50",
            }
        )
    state["sector_breadth"] = breadth_count

    peer_note = (
        f" [peer-confirmed: {breadth_count}/{len(BASKET)} above SMA50]"
        if breadth_count >= 2
        else ""
    )
    for symbol in SYMBOLS:
        df = load(symbol)
        earnings = None if symbol == "SPY" else next_earnings(symbol, today)  # ETFs don't report
        status = evaluate(symbol, df, basket, earnings)
        age = (today - date.fromisoformat(status["as_of"])).days
        if age > STALE_AFTER_DAYS:
            lines.append(f"WARN {symbol}: last bar {status['as_of']} is {age}d old — data stale")

        # post-earnings window: first sessions ON/AFTER a known print date
        as_of = date.fromisoformat(status["as_of"])
        window_open = False
        if earnings:
            edate = date.fromisoformat(earnings)
            days_since = (as_of - edate).days
            if 0 <= days_since <= POST_EARNINGS_WINDOW_SESSIONS + 2:
                window_open = True
                wkey = f"{symbol}:WINDOW:{earnings}"
                lines.append(f"{symbol} WINDOW post-earnings window open (printed {earnings})")
                if wkey not in state:
                    alerts.append(
                        f"ALERT {symbol} WINDOW: post-earnings window open (printed "
                        f"{earnings}) — dip triggers re-armed; tranche 1 is now legal"
                        + order_ticket(plan, symbol, "WINDOW")
                    )
                    state[wkey] = 1
                    fired.append((symbol, "WINDOW", status["as_of"]))
                    events.append(
                        {
                            "symbol": symbol,
                            "trigger": "WINDOW",
                            "as_of": status["as_of"],
                            "detail": f"post-earnings window open (printed {earnings})",
                        }
                    )
        status["window_open"] = window_open

        session_no = len(df)
        # REL is relative strength vs the IT-services basket — meaningless for SPY
        trigs = ("T1", "T2", "T3") if symbol == "SPY" else ("T1", "T2", "T3", "REL")
        for trig in trigs:
            info = status[trig]
            flag = "FIRED" if info["fired"] else "-"
            lines.append(f"{symbol} {trig:<4} {flag:<5} {info['detail']}")
            if info["fired"]:
                key = f"{symbol}:{trig}"
                if session_no - state.get(key, -(10**9)) >= COOLDOWN_SESSIONS[trig]:
                    alerts.append(
                        f"ALERT {symbol} {trig}: {info['detail']}{peer_note}"
                        + order_ticket(plan, symbol, trig)
                    )
                    state[key] = session_no
                    fired.append((symbol, trig, status["as_of"]))
                    events.append(
                        {
                            "symbol": symbol,
                            "trigger": trig,
                            "as_of": status["as_of"],
                            "detail": str(info["detail"]),
                        }
                    )
        if status["CAL"]:
            key = f"{symbol}:CAL:{today.year}-{today.month}"
            lines.append(f"{symbol} CAL   {status['CAL']}")
            if key not in state:
                alerts.append(
                    f"ALERT {symbol} CAL: {status['CAL']}" + order_ticket(plan, symbol, "CAL")
                )
                state[key] = 1
                fired.append((symbol, "CAL", status["as_of"]))
                events.append(
                    {
                        "symbol": symbol,
                        "trigger": "CAL",
                        "as_of": status["as_of"],
                        "detail": str(status["CAL"]),
                    }
                )

        score = go_score(
            t1=status["T1"]["fired"],
            near_dip=(status["rsi2"] < 20) or (status["ibs"] < 0.3),
            calm=status["calm"],
            low_vol=status["vol20"] < 0.30,
            rel_recross=status["REL"]["fired"],
            rel_ok=status["rel_z"] > -1,
            breadth=breadth_count,
            seasonal=bool(status["CAL"]),
            post_earnings=window_open,
        )
        status["GO"] = score
        lines.append(f"{symbol} GO   {score}/100 (alert at >={GO_ALERT_THRESHOLD})")
        if GO_ALERTS_ENABLED and score >= GO_ALERT_THRESHOLD:
            key = f"{symbol}:GO"
            if session_no - state.get(key, -(10**9)) >= COOLDOWN_SESSIONS["GO"]:
                alerts.append(
                    f"ALERT {symbol} GO: entry score {score}/100 — plan says act this week"
                )
                state[key] = session_no
                events.append(
                    {
                        "symbol": symbol,
                        "trigger": "GO",
                        "as_of": status["as_of"],
                        "detail": f"entry score {score}/100",
                    }
                )
        statuses.append(status)

    lines += paper_autopilot(plan, fired)

    session_no = max(len(load(s)) for s in SYMBOLS)
    for key, msg in review_positions({s["symbol"]: s for s in statuses}):
        lines.append(msg)
        if session_no - state.get(key, -(10**9)) >= COOLDOWN_SESSIONS["REVIEW"]:
            alerts.append(f"ALERT {msg}")
            state[key] = session_no
            parts = key.split(":")
            events.append(
                {
                    "symbol": parts[1] if len(parts) > 1 else "UNKNOWN",
                    "trigger": "REVIEW",
                    "as_of": today.isoformat(),
                    "detail": msg,
                }
            )

    stamp = today.isoformat()
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"===== tranche_watch {stamp} =====\n")
        for line in lines + alerts:
            fh.write(line + "\n")
    STATE_PATH.write_text(json.dumps(state, indent=2))
    public = [{k: v for k, v in s.items() if k != "_series"} for s in statuses]
    status_payload = json.dumps(
        {
            "run_status": "ok",
            "run_date": stamp,
            "trigger_types": PUBLIC_TRIGGER_TYPES,
            "go_alerts_enabled": GO_ALERTS_ENABLED,
            "breadth": breadth,
            "symbols": public,
            "events": events,
        },
        indent=2,
    )
    atomic_write_bytes(STATUS_PATH, status_payload.encode("utf-8"))
    write_dashboard(statuses, breadth, alerts, stamp)

    for line in lines:
        print(line)
    if alerts:
        print("\n" + "\n".join(alerts))
        notify(alerts)
    else:
        print("\nno new trigger firings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
