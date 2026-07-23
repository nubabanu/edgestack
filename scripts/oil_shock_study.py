"""Oil shock dip-entry study: is buying the pullback during a crude surge an edge?

Question (2026-07-23, after the Bab al-Mandab closure surge was missed): once
CL=F has already jumped (>= +4% day or >= +8% over 5 sessions — the same
thresholds scripts/oil_surge_watch.py alerts on), do the candidate entry
rules beat unconditional entry, and does the resulting strategy clear the
repo survivor bar? Only a PASS here may ever promote the watcher's
DISPLAY-ONLY dip alerts into ticketed paper orders.

Honest caveats, stated up front:
- This study exists BECAUSE a 2026 episode was missed: selection bias is
  baked into the question itself, and the 2024+ "holdout" is
  previously-accessed history twice over. Only dev+val carry weight; nothing
  here can promote a V2 sleeve (repo convention for scripts/ campaigns).
- CL=F is a continuous front-month series with roll artifacts and is not
  tradeable; verdicts transfer to a Brent ETC only approximately.
- Yahoo silently DROPPED the negative-price sessions of April 2020
  (2020-04-20/21 are missing rows), so returns bridging that gap are
  artifacts; the window 2020-04-15..2020-05-01 is excised and logged.
- Shock episodes are rare. With a few dozen episodes across 26 years, several
  split cells land below MIN_EVENTS_PER_SPLIT and print INSUFFICIENT —
  INSUFFICIENT is not PASS, and "alerts stay display-only forever" is a
  valid end state.
- 24 trials are evaluated (2 episode definitions x 4 rules x 3 horizons).
  The repo gate is pooled t >= 2; at 24 trials a Bonferroni-deflated bar is
  ~t >= 2.9 — both are reported, and any survivor that clears 2 but not 2.9
  is flagged MARGINAL_MULTIPLICITY.

Method: signal at close t fills at close t+1 (repo convention; identical
one-buy cost either way, so the 2 bps convention cancels for the event gate
and is charged only in the survivor-bar strategy series, 2 bps per unit of
exposure change). Splits: dev 2000-2015, val 2016-2023, holdout 2024+
(previously accessed). Gates per trial:
  1. event gate — conditional mean forward return > unconditional same-split
     mean in ALL splits with >= MIN_EVENTS_PER_SPLIT non-overlapping events
     each, AND pooled non-overlapping event t >= 2 (uso_entry_study form);
  2. survivor bar — long-CL=F-for-h-sessions strategy Sharpe >= buy-and-hold
     Sharpe in ALL splits AND pooled Newey-West alpha t >= 2.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from oil_surge_watch import (  # noqa: E402
    REGIME_SESSIONS,
    SHOCK_DAY_RETURN,
    SHOCK_SYMBOL,
    SHOCK_WINDOW_RETURN,
    SHOCK_WINDOW_SESSIONS,
)

from edgestack.validation.advanced_tests import newey_west_alpha  # noqa: E402
from edgestack.validation.metrics import sharpe_ratio  # noqa: E402

PRICES = ROOT / "data" / "curated" / "prices"
REPORT_PATH = ROOT / "artifacts" / "oil_shock_study.json"
VERDICT_PATH = ROOT / "artifacts" / "oil_shock_study_verdict.json"

SPLITS = {
    "dev_2000_2015": ("2000-01-01", "2015-12-31"),
    "val_2016_2023": ("2016-01-01", "2023-12-31"),
    "holdout_2024_prev_accessed": ("2024-01-01", "2026-12-31"),
}
HORIZONS = {"5d": 5, "10d": 10, "20d": 20}
MIN_EVENTS_PER_SPLIT = 5
COST_BPS = 2  # per unit of exposure change, survivor-bar series only
CRASH_REBOUND_LOOKBACK = 20  # E2: trailing sessions
CRASH_REBOUND_RETURN = -0.10  # E2: exclude if trailing return <= this at t0
PULLBACK_R3 = 0.015
PULLBACK_R4 = 0.025
# Yahoo dropped the negative-price rows; returns across the gap are artifacts.
EXCLUDE_WINDOWS = (("2020-04-15", "2020-05-01"),)
BONFERRONI_T = 2.9  # ~0.05/24 two-sided on a normal


def load_series() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(PRICES / f"{SHOCK_SYMBOL}.parquet")
    df = df.assign(date=pd.to_datetime(df["date"])).set_index("date").sort_index()
    dropped: list[str] = []
    bad = df["close"] <= 0
    if bad.any():
        dropped += [f"close<=0 {d.date()}" for d in df.index[bad]]
        df = df[~bad]
    for lo, hi in EXCLUDE_WINDOWS:
        mask = (df.index >= lo) & (df.index <= hi)
        if mask.any():
            dropped.append(f"excised {lo}..{hi} ({int(mask.sum())} sessions, negative-price gap)")
            df = df[~mask]
    return df, dropped


def find_episodes(closes: pd.Series) -> list[int]:
    """E1 episode starts (positional indices): qualifying days clustered so a
    new episode needs REGIME_SESSIONS quiet sessions since the last start."""
    c = closes.to_numpy()
    starts: list[int] = []
    last = -(10**9)
    for i in range(1, len(c)):
        day_jump = c[i] / c[i - 1] - 1 >= SHOCK_DAY_RETURN
        window_jump = (
            i >= SHOCK_WINDOW_SESSIONS
            and c[i] / c[i - SHOCK_WINDOW_SESSIONS] - 1 >= SHOCK_WINDOW_RETURN
        )
        if (day_jump or window_jump) and i - last >= REGIME_SESSIONS:
            starts.append(i)
            last = i
    return starts


def is_crash_rebound(closes: pd.Series, t0: int) -> bool:
    if t0 < CRASH_REBOUND_LOOKBACK:
        return False
    trailing = closes.iloc[t0] / closes.iloc[t0 - CRASH_REBOUND_LOOKBACK] - 1
    return bool(trailing <= CRASH_REBOUND_RETURN)


def entry_signal_index(closes: pd.Series, t0: int, rule: str) -> int | None:
    """Positional index of the SIGNAL close for `rule` inside episode at t0.

    R1 immediate (signal = shock day); R2 first down close; R3/R4 first
    pullback of >= 1.5%/2.5% from the post-shock closing high. Signal at
    close t fills at close t+1 (handled by the caller)."""
    if rule == "R1":
        return t0
    c = closes.to_numpy()
    high = c[t0]
    end = min(t0 + REGIME_SESSIONS, len(c) - 1)
    threshold = {"R3": PULLBACK_R3, "R4": PULLBACK_R4}.get(rule)
    for t in range(t0 + 1, end + 1):
        if rule == "R2":
            if c[t] < c[t - 1]:
                return t
        else:
            high = max(high, c[t])
            if c[t] / high - 1 <= -float(threshold or 0):
                return t
    return None


def evaluate_trial(
    closes: pd.Series,
    adj: pd.Series,
    signals: list[int],
    horizon: int,
) -> dict:
    """Event gate for one (episode-def, rule, horizon) trial, uso_entry_study form."""
    fwd = adj.shift(-1 - horizon) / adj.shift(-1) - 1  # fill at close t+1
    # dedupe overlapping events: keep first signal per horizon-length cluster
    events: list[int] = []
    last = -(10**9)
    for t in sorted(signals):
        if t - last >= horizon:
            events.append(t)
            last = t
    row: dict = {"events_total": len(events)}
    beats = 0
    decidable = True
    pooled: list[float] = []
    for slab, (lo, hi) in SPLITS.items():
        in_split = (fwd.index >= lo) & (fwd.index <= hi)
        cond = fwd.iloc[[t for t in events if in_split[t]]].dropna()
        uncond = fwd[in_split].dropna()
        cell = {
            "n": len(cond),
            "mean_fwd": round(float(cond.mean()), 4) if len(cond) else None,
            "uncond_mean": round(float(uncond.mean()), 4) if len(uncond) else None,
        }
        if len(cond) < MIN_EVENTS_PER_SPLIT:
            cell["note"] = "INSUFFICIENT"
            decidable = False
        elif len(uncond) and cond.mean() > uncond.mean():
            beats += 1
        if len(cond) and len(uncond):
            pooled.extend((cond - uncond.mean()).tolist())
        row[slab] = cell
    excess = np.array(pooled)
    t_stat = (
        float(excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess))))
        if len(excess) > 2 and excess.std(ddof=1) > 0
        else 0.0
    )
    row["beats_unconditional_splits"] = beats
    row["pooled_events"] = len(excess)
    row["pooled_excess_mean"] = round(float(excess.mean()), 4) if len(excess) else None
    row["pooled_t"] = round(t_stat, 2)
    if not decidable:
        row["event_gate"] = "INSUFFICIENT"
    elif beats == len(SPLITS) and t_stat >= 2:
        row["event_gate"] = "PASS"
    else:
        row["event_gate"] = "FAIL"
    row["_events"] = events
    row["_t"] = t_stat
    return row


def strategy_returns(adj: pd.Series, events: list[int], horizon: int) -> pd.Series:
    """Daily returns of: long CL=F for `horizon` sessions after each fill
    (fill at close t+1), flat otherwise; COST_BPS per exposure change."""
    ret = adj.pct_change().fillna(0.0)
    exposure = np.zeros(len(adj))
    for t in events:
        start = t + 1  # fill at close t+1: exposed from t+2's return onward
        stop = min(start + horizon, len(adj) - 1)
        exposure[start:stop] = 1.0
    exp_s = pd.Series(exposure, index=adj.index)
    turn = exp_s.diff().abs().fillna(exp_s.abs())
    return exp_s.shift(1).fillna(0.0) * ret - turn * (COST_BPS / 10_000)


def survivor_bar(strategy: pd.Series, market: pd.Series) -> dict:
    per_split = {}
    all_ok = True
    for slab, (lo, hi) in SPLITS.items():
        mask = (strategy.index >= lo) & (strategy.index <= hi)
        s = sharpe_ratio(strategy[mask].to_numpy())
        b = sharpe_ratio(market[mask].dropna().to_numpy())
        ok = s >= b
        all_ok &= ok
        per_split[slab] = {"strategy_sharpe": round(s, 2), "bh_sharpe": round(b, 2), "ok": ok}
    nw = newey_west_alpha(strategy, market.dropna())
    alpha_t = float(nw["alpha_t"])
    return {
        "splits": per_split,
        "alpha_t": round(alpha_t, 2),
        "gate": "PASS" if (all_ok and alpha_t >= 2) else "FAIL",
    }


def main() -> int:
    df, dropped = load_series()
    closes, adj = df["close"], df["adj_close"]
    market = adj.pct_change()

    e1 = find_episodes(closes)
    e2 = [t0 for t0 in e1 if not is_crash_rebound(closes, t0)]
    episode_defs = {"E1": e1, "E2": e2}
    batch_id = uuid.uuid4().hex[:12]
    report: dict = {
        "symbol": SHOCK_SYMBOL,
        "sessions": len(df),
        "span": [str(df.index[0].date()), str(df.index[-1].date())],
        "batch_id": batch_id,
        "trials": len(episode_defs) * 4 * len(HORIZONS),
        "bonferroni_t": BONFERRONI_T,
        "dropped": dropped,
        "episodes": {
            k: {"n": len(v), "dates": [str(closes.index[t].date()) for t in v]}
            for k, v in episode_defs.items()
        },
        "splits": {k: list(v) for k, v in SPLITS.items()},
        "results": {},
        "survivors": [],
        "marginal_multiplicity": [],
    }

    for edef, starts in episode_defs.items():
        for rule in ("R1", "R2", "R3", "R4"):
            signals = [
                s for t0 in starts if (s := entry_signal_index(closes, t0, rule)) is not None
            ]
            for hlab, h in HORIZONS.items():
                key = f"{edef}/{rule}@{hlab}"
                row = evaluate_trial(closes, adj, signals, h)
                events = row.pop("_events")
                t_stat = row.pop("_t")
                if row["event_gate"] == "PASS":
                    row["survivor_bar"] = survivor_bar(strategy_returns(adj, events, h), market)
                    if row["survivor_bar"]["gate"] == "PASS":
                        report["survivors"].append(key)
                        if t_stat < BONFERRONI_T:
                            report["marginal_multiplicity"].append(key)
                report["results"][key] = row

    report["verdict"] = (
        f"survivors at the repo bar: {report['survivors']} "
        f"(marginal under 24-trial Bonferroni: {report['marginal_multiplicity']}); "
        "human review required before enabling DIP_TICKETS_ENABLED"
        if report["survivors"]
        else (
            "no shock dip-entry rule clears the event gate plus survivor bar; "
            "oil_surge_watch DIP alerts stay DISPLAY-ONLY"
        )
    )
    report["disclaimer"] = (
        "Historical research on previously-accessed data; the study question was "
        "selected after observing a 2026 episode; holdout previously accessed; "
        "not investment advice."
    )

    from edgestack.data.catalog import atomic_write_bytes

    atomic_write_bytes(REPORT_PATH, json.dumps(report, indent=2).encode("utf-8"))
    verdict = {
        "verdict": "PASS" if report["survivors"] else "FAIL",
        "rules_passed": report["survivors"],
        "marginal_multiplicity": report["marginal_multiplicity"],
        "batch_id": batch_id,
        "trials": report["trials"],
        "generated_from_span": report["span"],
    }
    atomic_write_bytes(VERDICT_PATH, json.dumps(verdict, indent=2).encode("utf-8"))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
