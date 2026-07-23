"""Swing-zone strategy study: walk-forward test of buying dip zones of oscillators.

Question (2026-07-23, user request): build the best strategy exploiting the
fuzzy-zone swing behavior (swing_cycle_scan) and test it in simulation.

Strategy (pre-registered, one rule family, three scales as the only knob):
  At each session, using ONLY past data: confirmed zigzag pivots (theta) on
  the trailing window define the current dip zone (median of last 3 confirmed
  troughs) and top zone (median of last 3 confirmed peaks). The asset is an
  "oscillator" when >= 4 pivots confirmed in the trailing ~3y and the zones
  are meaningfully apart. Entry: oscillator AND close <= dip_zone * 1.05.
  Exit: close >= top_zone * 0.95 (target), close < dip_zone * 0.85 (zone
  break - the regime is dead), or timeout (2x the scale's typical swing
  time). Signal at close t, fill at close t+1, costs charged per side.

Honest caveats, stated up front:
- Walk-forward removes zone look-ahead, but the QUESTION itself (test swing
  zones) was chosen after seeing the scans - name-level selection bias is
  gone (the rule runs on the whole universe), question-level bias remains.
- Trials: 3 thetas x 2 cost levels reported; gates applied at the repo
  convention cost (2 bps/side) with the retail cost (20 bps/side) shown.
- ^VIX/^OVX are reported separately as UNTRADEABLE references (no spot
  vehicle; VIX products bleed roll). Futures results carry roll caveats.
- Random-entry null: same symbols, same number of trades, same holding
  lengths, random entry dates - what timing-free luck produces.
- The stock_range_study precedent (zero floor-bounce edge) is the prior.
- Previously-accessed data; dev 2000-2015 / val 2016-2023 / holdout 2024+
  (previously accessed). Not investment advice; no tickets without PASS.

Gate (survivor bar, uso_entry_study form): pooled non-overlapping net trade
return t >= 2 AND mean net trade return > random-entry null 95th pct AND
positive mean net in dev AND val (holdout reported, previously accessed).
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from swing_cycle_scan import load_prices  # noqa: E402  (same universe screens)

PRICES = ROOT / "data" / "curated" / "prices"
REPORT_PATH = ROOT / "artifacts" / "swing_zone_strategy_study.json"

THETAS = {0.05: 21, 0.15: 63, 0.30: 126}  # theta -> timeout sessions
ENTRY_TOL = 1.05
TARGET_TOL = 0.95
STOP_TOL = 0.85
MIN_PIVOTS_TRAILING = 4
TRAILING = 756
MIN_ZONE_GAP = 1.10  # top zone must be >= 10% above dip zone to bother
COST_SIDE_BPS = 2  # repo convention; 20 bps retail sensitivity also reported
RETAIL_SIDE_BPS = 20
NULL_DRAWS = 200
SPLITS = {
    "dev_2000_2015": ("2000-01-01", "2015-12-31"),
    "val_2016_2023": ("2016-01-01", "2023-12-31"),
    "holdout_2024_prev_accessed": ("2024-01-01", "2026-12-31"),
}


def zigzag_confirmed(log_px: np.ndarray, theta: float) -> list[tuple[int, int, str]]:
    """(pivot_idx, confirm_idx, kind) with kind in {'P','T'}; confirm is causal."""
    out: list[tuple[int, int, str]] = []
    hi = lo = 0
    direction = 0
    for i in range(1, len(log_px)):
        if log_px[i] > log_px[hi]:
            hi = i
        if log_px[i] < log_px[lo]:
            lo = i
        if direction >= 0 and log_px[hi] - log_px[i] >= theta:
            out.append((hi, i, "P"))
            direction = -1
            hi = lo = i
        elif direction <= 0 and log_px[i] - log_px[lo] >= theta:
            out.append((lo, i, "T"))
            direction = 1
            hi = lo = i
    return out


def run_symbol(px: pd.Series, theta: float, timeout: int) -> list[dict]:
    """Walk-forward trades for one symbol at one theta. Causal by construction."""
    log_px = np.log(px.to_numpy(dtype=float))
    n = len(log_px)
    pivots = zigzag_confirmed(log_px, theta)
    trades: list[dict] = []
    pos_entry_i: int | None = None
    # walk confirmed pivots forward with pointers
    confirmed: list[tuple[int, int, str]] = []
    ptr = 0
    for t in range(TRAILING, n - 1):
        while ptr < len(pivots) and pivots[ptr][1] <= t:
            confirmed.append(pivots[ptr])
            ptr += 1
        recent = [pv for pv in confirmed if pv[1] > t - TRAILING]
        troughs = [pv[0] for pv in recent if pv[2] == "T"][-3:]
        peaks = [pv[0] for pv in recent if pv[2] == "P"][-3:]
        if len(recent) < MIN_PIVOTS_TRAILING or not troughs or not peaks:
            continue
        dip_zone = float(np.exp(np.median(log_px[troughs])))
        top_zone = float(np.exp(np.median(log_px[peaks])))
        if top_zone / dip_zone < MIN_ZONE_GAP:
            continue
        price = float(np.exp(log_px[t]))
        if pos_entry_i is None:
            if price <= dip_zone * ENTRY_TOL:
                pos_entry_i = t + 1  # fill at next close
        else:
            held = t - pos_entry_i
            exit_reason = None
            if price >= top_zone * TARGET_TOL:
                exit_reason = "target"
            elif price < dip_zone * STOP_TOL:
                exit_reason = "zone_break"
            elif held >= timeout:
                exit_reason = "timeout"
            if exit_reason and pos_entry_i < n - 1:
                fill_out = min(t + 1, n - 1)
                gross = float(np.exp(log_px[fill_out] - log_px[pos_entry_i]) - 1)
                trades.append(
                    {
                        "entry_i": pos_entry_i,
                        "exit_i": fill_out,
                        "entry_date": str(px.index[pos_entry_i].date()),
                        "gross": gross,
                        "held": fill_out - pos_entry_i,
                        "reason": exit_reason,
                    }
                )
                pos_entry_i = None
    return trades


def net(gross: float, side_bps: float) -> float:
    return gross - 2 * side_bps / 10_000


def pooled_stats(rets: list[float]) -> dict:
    arr = np.array(rets)
    if len(arr) < 3:
        return {"n": len(arr)}
    t = float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr))))
    return {
        "n": len(arr),
        "mean_net": round(float(arr.mean()), 4),
        "win_rate": round(float((arr > 0).mean()), 3),
        "t": round(t, 2),
    }


def main() -> int:
    rng = np.random.default_rng(20260723)
    prices: dict[str, pd.Series] = {}
    for path in sorted(PRICES.glob("*.parquet")):
        try:
            px = load_prices(path.stem)
        except Exception:
            continue
        if px is not None:
            prices[path.stem] = px

    report: dict = {
        "batch_id": uuid.uuid4().hex[:12],
        "universe": len(prices),
        "trials": len(THETAS),
        "thetas": {},
        "disclaimer": (
            "Walk-forward simulation; question-level selection bias remains; "
            "futures ignore roll; ^ indices untradeable; previously-accessed "
            "data; not investment advice."
        ),
    }

    for theta, timeout in THETAS.items():
        all_trades: dict[str, list[dict]] = {}
        for sym, px in prices.items():
            trades = run_symbol(px, theta, timeout)
            if trades:
                all_trades[sym] = trades

        def bucket(sym: str) -> str:
            if sym.startswith("^"):
                return "untradeable_index"
            if "=" in sym:
                return "futures"
            return "stocks"

        block: dict = {"timeout": timeout}
        for group in ("stocks", "futures", "untradeable_index"):
            rets: list[float] = []
            retail: list[float] = []
            per_split: dict[str, list[float]] = {k: [] for k in SPLITS}
            reasons: dict[str, int] = {}
            symbol_trades: list[tuple[str, dict]] = []
            for sym, trades in all_trades.items():
                if bucket(sym) != group:
                    continue
                for tr in trades:
                    rets.append(net(tr["gross"], COST_SIDE_BPS))
                    retail.append(net(tr["gross"], RETAIL_SIDE_BPS))
                    reasons[tr["reason"]] = reasons.get(tr["reason"], 0) + 1
                    symbol_trades.append((sym, tr))
                    for slab, (lo, hi) in SPLITS.items():
                        if lo <= tr["entry_date"] <= hi:
                            per_split[slab].append(net(tr["gross"], COST_SIDE_BPS))

            # random-entry null: same symbols, same holding lengths, random dates
            null_means: list[float] = []
            if symbol_trades:
                for _ in range(NULL_DRAWS):
                    sample = []
                    for sym, tr in symbol_trades:
                        lp = np.log(prices[sym].to_numpy(dtype=float))
                        h = max(tr["held"], 1)
                        j = int(rng.integers(TRAILING, len(lp) - h - 1))
                        sample.append(net(float(np.exp(lp[j + h] - lp[j]) - 1), COST_SIDE_BPS))
                    null_means.append(float(np.mean(sample)))
            null_p95 = round(float(np.percentile(null_means, 95)), 4) if null_means else None
            null_p50 = round(float(np.percentile(null_means, 50)), 4) if null_means else None

            stats = pooled_stats(rets)
            split_stats = {k: pooled_stats(v) for k, v in per_split.items()}
            passed = bool(
                stats.get("n", 0) >= 30
                and stats.get("t", 0) >= 2
                and null_p95 is not None
                and stats.get("mean_net", -1) > null_p95
                and split_stats["dev_2000_2015"].get("mean_net", -1) > 0
                and split_stats["val_2016_2023"].get("mean_net", -1) > 0
            )
            block[group] = {
                "pooled": stats,
                "pooled_retail_costs": pooled_stats(retail),
                "splits": split_stats,
                "exit_reasons": reasons,
                "random_entry_null": {"p50": null_p50, "p95": null_p95},
                "gate": "PASS" if passed else "FAIL",
            }
        report["thetas"][f"{int(theta * 100)}pct"] = block

    from edgestack.data.catalog import atomic_write_bytes

    atomic_write_bytes(REPORT_PATH, json.dumps(report, indent=2).encode("utf-8"))

    for tkey, block in report["thetas"].items():
        print(f"\n===== theta {tkey} (timeout {block['timeout']}d) =====")
        for group in ("stocks", "futures", "untradeable_index"):
            g = block[group]
            if g["pooled"].get("n", 0) < 3:
                print(f"  {group}: <3 trades")
                continue
            print(
                f"  {group}: {g['gate']}  n={g['pooled']['n']} "
                f"mean {g['pooled']['mean_net']:+.2%}/trade win {g['pooled']['win_rate']:.0%} "
                f"t={g['pooled']['t']} | retail-cost mean "
                f"{g['pooled_retail_costs']['mean_net']:+.2%} | null p50/p95 "
                f"{g['random_entry_null']['p50']}/{g['random_entry_null']['p95']} | "
                f"exits {g['exit_reasons']}"
            )
            for slab, s in g["splits"].items():
                if s.get("n", 0) >= 3:
                    print(f"    {slab:28s} n={s['n']:>4} mean {s['mean_net']:+.2%} t={s['t']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
