"""Audit the legacy market WIND score without turning it into a trading signal.

The original 2026-07-21 campaign used five price/calendar features, six rule
trials, and data through 2026-07-15.  The results were already viewed.  This
script freezes that cutoff, reproduces the legacy arithmetic under neutral
component names, and adds explicitly post-hoc diagnostics that can never count
as promotion evidence.

Historical conclusion: score-based selective entry failed; exact score means
were not strictly monotonic; only the QQQ daily switching rule that omitted
negative-score sessions cleared the old survivor bar.  That switching result
does not test whether delaying an already-planned purchase improves its fill.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from edgestack.data.catalog import atomic_write_bytes
from edgestack.research.market_wind import (
    COMPONENTS,
    LEGACY_CUTOFF,
    bars_fingerprint,
    component_votes,
    score_bucket_report,
)
from edgestack.validation.advanced_tests import newey_west_alpha
from edgestack.validation.metrics import max_drawdown, sharpe_ratio

ROOT = Path(__file__).resolve().parents[1]
ZOO_CACHE = ROOT / "data" / "cache" / "zoo"
OUT = ROOT / "artifacts" / "confluence_timing.json"

PRIMARY_COST = 0.0002
COST_GRID = (0.0, 0.0002, 0.0005, 0.0010)
VEHICLES = ("SPY", "QQQ")
SPLITS = {
    "dev_1999_2015": ("1999-01-01", "2015-12-31"),
    "val_2016_2023": ("2016-01-01", "2023-12-31"),
    "holdout_2024_prev_accessed": ("2024-01-01", LEGACY_CUTOFF.isoformat()),
}


def load_bars(symbol: str) -> pd.DataFrame:
    raw = pd.read_parquet(ZOO_CACHE / f"{symbol}.parquet")
    bars = (
        raw.assign(date=raw["dt"].dt.tz_localize(None).dt.normalize())
        .set_index("date")
        .drop(columns="dt")
        .sort_index()
        .loc[: LEGACY_CUTOFF.isoformat()]
    )
    if bars.empty or bars.index[-1].date() != LEGACY_CUTOFF:
        raise RuntimeError(f"{symbol} does not contain the frozen cutoff {LEGACY_CUTOFF}")
    return bars


def family_votes(frame: pd.DataFrame) -> pd.DataFrame:
    """Compatibility alias for callers of the original research script."""
    return component_votes(frame)


def _legacy_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def legacy_component_votes(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact original arithmetic, including its dataset-edge calendar bug.

    This exists only to reproduce the six viewed legacy trials.  Live and
    audited calculations use :func:`component_votes` and the XNYS calendar.
    """
    index = pd.DatetimeIndex(frame.index)
    returns = frame["adj"].pct_change()
    close = frame["close"]
    month = index.to_period("M")
    session_of_month = pd.Series(range(len(index)), index=index).groupby(month).cumcount() + 1
    sessions_left = (
        pd.Series(range(len(index)), index=index).groupby(month).cumcount(ascending=False)
    )
    turn_of_month = (session_of_month <= 3) | (sessions_left == 0)
    monthly_returns = frame["adj"].groupby(month).last().pct_change()
    prior_month_down = pd.Series(
        month.map(lambda period: bool(monthly_returns.get(period - 1, np.nan) < 0)), index=index
    )

    weekday = pd.Series(index.dayofweek, index=index)
    monday_reversal = (weekday == 0) & (returns.shift(1) < 0)
    volatility_20 = (returns.rolling(20).std() * np.sqrt(252)).shift(1)
    above_200 = (close > close.rolling(200).mean()).shift(1).astype(bool)
    calm = above_200 & (volatility_20 < 0.30)
    stressed = (~above_200) & (volatility_20 > 0.30)
    stressed_friday = (weekday == 4) & stressed
    rsi_2 = _legacy_rsi(close, 2).shift(1)
    down_3 = ((returns < 0) & (returns.shift(1) < 0) & (returns.shift(2) < 0)).shift(1)

    votes = pd.DataFrame(index=index)
    votes["turn_of_month"] = turn_of_month.astype(float)
    votes["post_down_month"] = ((session_of_month <= 3) & prior_month_down).astype(float)
    votes["weekday_reversal"] = monday_reversal.astype(float) - stressed_friday.astype(float)
    votes["trend_vol_regime"] = calm.astype(float) - stressed.astype(float)
    votes["short_term_dip"] = ((rsi_2 < 10) | down_3.fillna(False)).astype(float)
    return votes.loc[:, list(COMPONENTS)]


def net_returns(hold: pd.Series, returns: pd.Series, *, cost: float = PRIMARY_COST) -> pd.Series:
    return hold * returns - cost * hold.diff().abs().fillna(0.0)


def split_row(strategy: pd.Series, returns: pd.Series) -> tuple[dict, bool]:
    row: dict = {}
    passes_every_split = True
    for split, (low, high) in SPLITS.items():
        sample = strategy.loc[low:high].dropna()
        benchmark = returns.loc[sample.index].fillna(0.0)
        if len(sample) < 200:
            passes_every_split = False
            continue
        strategy_sharpe = sharpe_ratio(sample.to_numpy())
        benchmark_sharpe = sharpe_ratio(benchmark.to_numpy())
        row[split] = {
            "sharpe": round(strategy_sharpe, 2),
            "bh_sharpe": round(benchmark_sharpe, 2),
            "maxdd": round(max_drawdown(sample.to_numpy()), 3),
        }
        passes_every_split &= strategy_sharpe >= benchmark_sharpe
    return row, passes_every_split


def rule_metrics(hold: pd.Series, returns: pd.Series, *, cost: float = PRIMARY_COST) -> dict:
    strategy = net_returns(hold, returns, cost=cost)
    pooled = strategy.dropna()
    newey_west = newey_west_alpha(pooled, returns.loc[pooled.index].fillna(0.0))
    splits, all_splits = split_row(strategy, returns)
    return {
        "avg_exposure": round(float(hold.mean()), 3),
        **splits,
        "alpha_t": round(newey_west["alpha_t"], 2),
        "alpha_ann": round(newey_west["alpha_ann"], 4),
        "old_survivor_bar": bool(all_splits and newey_west["alpha_t"] >= 2.0),
    }


def _bootstrap_mean(values: np.ndarray, *, seed: int, draws: int = 4_000) -> dict:
    clean = values[np.isfinite(values)]
    if len(clean) < 2:
        return {"n": len(clean), "mean_bps": None, "ci95_bps": [None, None]}
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for index in range(draws):
        means[index] = rng.choice(clean, size=len(clean), replace=True).mean()
    return {
        "n": len(clean),
        "mean_bps": round(float(clean.mean()) * 10_000, 1),
        "ci95_bps": [
            round(float(np.quantile(means, 0.025)) * 10_000, 1),
            round(float(np.quantile(means, 0.975)) * 10_000, 1),
        ],
    }


def blocked_intervals(score: pd.Series, returns: pd.Series, *, seed: int) -> dict:
    """Year-block and contiguous-headwind-episode uncertainty summaries."""
    mask = score < 0
    skipped = returns.where(mask).dropna()
    yearly = skipped.groupby(skipped.index.year).mean().to_numpy()
    episode_id = (mask != mask.shift(fill_value=False)).cumsum()
    episode_means = returns[mask].groupby(episode_id[mask]).mean().dropna().to_numpy()
    return {
        "year_blocks": _bootstrap_mean(yearly, seed=seed),
        "headwind_episodes": _bootstrap_mean(episode_means, seed=seed + 1),
    }


def main() -> int:
    report: dict = {
        "campaign": "confluence_timing_legacy_audit_v2",
        "method_version": "legacy-wind-neutral-v2",
        "frozen_cutoff": LEGACY_CUTOFF.isoformat(),
        "components": COMPONENTS,
        "legacy_predeclared_trials": 6,
        "post_hoc_diagnostic_trials": 0,
        "previously_accessed": True,
        "promotion_eligible": False,
        "registration_note": (
            "The original rules were documented at evaluation time but were not independently "
            "timestamped before results; this is not a verified preregistration."
        ),
        "data_hashes": {},
        "monotonicity": {},
        "component_correlations": {},
        "legacy_rules": [],
        "calendar_correction": {},
        "calendar_corrected_rules": [],
        "simple_filter_benchmark": {},
        "leave_one_out": {},
        "cost_sensitivity": {},
        "blocked_return_intervals": {},
        "conclusions": [
            "Strict score-return monotonicity failed in every vehicle/split.",
            "Selective entry at score >= 2 or >= 3 failed the old survivor bar.",
            "The QQQ score<0 switching veto passed historically; SPY did not.",
            "A daily switching veto does not validate delaying an already-planned purchase.",
            "No result in this report is actionable or promotion evidence.",
        ],
        "caveats": [
            "Intraday MOC, VWAP, lunch, and options-positioning claims are not tested here.",
            "The 2024+ interval and all source-agent results were previously accessed.",
            "Feature names are empirical descriptions, not claims about unobserved flows.",
        ],
        "disclaimer": "Historical research on previously-accessed data; not investment advice.",
    }

    for vehicle_index, symbol in enumerate(VEHICLES):
        bars = load_bars(symbol)
        returns = bars["adj"].pct_change()
        legacy_votes = legacy_component_votes(bars)
        votes = component_votes(bars)
        legacy_score = legacy_votes.sum(axis=1)
        score = votes.sum(axis=1)
        report["data_hashes"][symbol] = bars_fingerprint(bars)
        changed = legacy_votes != votes
        report["calendar_correction"][symbol] = {
            "changed_rows": int(changed.any(axis=1).sum()),
            "changed_cells": int(changed.sum().sum()),
            "sessions": [str(value.date()) for value in changed.index[changed.any(axis=1)]],
            "reason": (
                "The original observed-row cumcounts treated partial dataset edges as true "
                "month boundaries; the audited engine uses the complete XNYS calendar."
            ),
        }
        report["component_correlations"][symbol] = {
            name: {other: round(float(value), 3) for other, value in row.items()}
            for name, row in votes.corr().to_dict(orient="index").items()
        }

        split_reports = {}
        for split, (low, high) in SPLITS.items():
            split_reports[split] = score_bucket_report(score.loc[low:high], returns.loc[low:high])
        report["monotonicity"][symbol] = split_reports

        legacy_rules = {
            "score_ge2": (legacy_score >= 2).astype(float),
            "score_ge3": (legacy_score >= 3).astype(float),
            "avoid_headwind": (legacy_score >= 0).astype(float),
        }
        for name, hold in legacy_rules.items():
            report["legacy_rules"].append(
                {"rule": name, "vehicle": symbol, **rule_metrics(hold, returns)}
            )
        corrected_rules = {
            "score_ge2": (score >= 2).astype(float),
            "score_ge3": (score >= 3).astype(float),
            "avoid_headwind": (score >= 0).astype(float),
        }
        for name, hold in corrected_rules.items():
            report["calendar_corrected_rules"].append(
                {"rule": name, "vehicle": symbol, **rule_metrics(hold, returns)}
            )
            report["post_hoc_diagnostic_trials"] += 1

        simple = (votes["trend_vol_regime"] >= 0).astype(float)
        report["simple_filter_benchmark"][symbol] = rule_metrics(simple, returns)
        report["post_hoc_diagnostic_trials"] += 1

        leave_one_out = {}
        for component in COMPONENTS:
            reduced_score = score - votes[component]
            leave_one_out[component] = rule_metrics((reduced_score >= 0).astype(float), returns)
            report["post_hoc_diagnostic_trials"] += 1
        report["leave_one_out"][symbol] = leave_one_out

        cost_rows = {}
        avoid = (score >= 0).astype(float)
        for cost in COST_GRID:
            cost_rows[str(round(cost * 10_000, 1))] = rule_metrics(avoid, returns, cost=cost)
            report["post_hoc_diagnostic_trials"] += 1
        report["cost_sensitivity"][symbol] = cost_rows
        report["blocked_return_intervals"][symbol] = blocked_intervals(
            score, returns, seed=19_990 + vehicle_index * 100
        )
        report["post_hoc_diagnostic_trials"] += 2

    report["total_trials_and_diagnostics"] = (
        report["legacy_predeclared_trials"] + report["post_hoc_diagnostic_trials"]
    )
    atomic_write_bytes(OUT, json.dumps(report, indent=2).encode("utf-8"))

    print(
        f"legacy trials={report['legacy_predeclared_trials']}; "
        f"post-hoc diagnostics={report['post_hoc_diagnostic_trials']}"
    )
    for symbol in VEHICLES:
        flags = {
            split: values["strict_monotonic"]
            for split, values in report["monotonicity"][symbol].items()
        }
        print(f"{symbol} strict monotonicity: {flags}")
    for row in report["legacy_rules"]:
        print(
            f"{row['vehicle']} {row['rule']}: alpha_t={row['alpha_t']:+.2f} "
            f"old_survivor={row['old_survivor_bar']}"
        )
    print(f"full audit -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
