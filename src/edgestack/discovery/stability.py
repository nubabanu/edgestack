"""Time-based stability selection over RuleFit discoveries
(Meinshausen & Bühlmann 2010, adapted to walk-forward finance data).

Protocol inside ONE training period (no outer data ever enters):

1. The first 80% of training sessions is the DISCOVERY region; the last 20%
   is the INNER VALIDATION region no discovery run may touch.
2. RuleFit runs on W expanding chronological windows x S symbol subsamples of
   the discovery region. Rules canonicalize by structural signature
   (feature + direction + threshold decile), so vol<22%/24%/25% merge.
3. Per structural rule: selection frequency, sign consistency, and a
   median-threshold representative condition.
4. Survivors (frequency >= min_freq, sign consistency >= 0.9) are then scored
   on the untouched inner-validation region: direction-consistent mean excess
   return with a Chordia-Goyal-Saretto t-hurdle, minimum trade count, and
   mean adverse excursion.

Total rules generated across ALL runs is returned as the trial count.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from edgestack.discovery.rulefit_engine import DiscoveredRule, _fast_mask, discover_rules
from edgestack.exceptions import ValidationError
from edgestack.logging import get_logger, log_event
from edgestack.types import AllOf, Condition, Predicate
from edgestack.validation.advanced_tests import passes_cgs_hurdle

log = get_logger("stability")

DISCOVERY_FRACTION = 0.8
WINDOW_FRACTIONS = (0.5, 0.65, 0.8)  # expanding, all inside the discovery region
SYMBOL_SUBSAMPLE = 0.7
N_SUBSAMPLES = 3


@dataclass(frozen=True)
class StableRule:
    signature: str
    condition: Condition
    direction: int  # +1 long-favorable, -1 avoid/short-research
    selection_freq: float
    sign_consistency: float
    n_runs_present: int
    mean_support: float
    inner_n: int
    inner_mean_excess: float  # DEMEANED: mean(selected) - mean(all inner)
    inner_abs_mean: float  # absolute mean of selected inner returns
    inner_t: float  # t-stat of the demeaned edge (coef semantics)
    inner_mean_mae: float | None
    passes: bool
    fail_reasons: tuple[str, ...]


def _median_condition(occurrences: list[DiscoveredRule]) -> Condition:
    """Representative rule: median threshold per (feature, direction)."""
    thresholds: dict[tuple[str, str], list[float]] = defaultdict(list)
    ops: dict[tuple[str, str], str] = {}
    for rule in occurrences:
        preds = (
            (rule.condition,)
            if isinstance(rule.condition, Predicate)
            else rule.condition.conditions
        )
        for p in preds:
            if not isinstance(p, Predicate):  # rules are flat conjunctions
                continue
            direction = "<" if p.op in ("<", "<=") else ">"
            thresholds[(p.feature, direction)].append(float(p.value))  # type: ignore[arg-type]
            ops[(p.feature, direction)] = p.op
    preds_out = tuple(
        Predicate(
            feature=f,
            op=ops[(f, d)],  # type: ignore[arg-type]
            value=float(np.median(vals)),
        )
        for (f, d), vals in sorted(thresholds.items())
    )
    return preds_out[0] if len(preds_out) == 1 else AllOf(conditions=preds_out)


def stability_select(
    train: pd.DataFrame,
    feature_cols: tuple[str, ...],
    target_col: str,
    *,
    seed: int = 42,
    min_selection_freq: float = 0.7,
    min_inner_n: int = 100,
    mae_col: str | None = None,
    max_rules: int = 300,
    max_fit_rows: int = 50_000,
    min_support: float = 0.005,
    min_symbols: int = 30,
) -> tuple[list[StableRule], int]:
    """Run the full stability protocol; returns (all structural rules with
    verdicts, total trial count)."""
    sessions = np.sort(train["date"].unique())
    if len(sessions) < 500:
        raise ValidationError("training period too short for stability selection")
    disc_end = sessions[int(len(sessions) * DISCOVERY_FRACTION) - 1]
    discovery = train.loc[train["date"] <= disc_end]
    inner = train.loc[train["date"] > disc_end]
    symbols = discovery["symbol"].unique()

    occurrences: dict[str, list[DiscoveredRule]] = defaultdict(list)
    trials = 0
    n_runs = 0
    rng = np.random.default_rng(seed)
    disc_sessions = np.sort(discovery["date"].unique())
    for frac in WINDOW_FRACTIONS:
        window_end = disc_sessions[int(len(disc_sessions) * frac) - 1]
        window = discovery.loc[discovery["date"] <= window_end]
        for s in range(N_SUBSAMPLES):
            keep = rng.choice(symbols, size=int(len(symbols) * SYMBOL_SUBSAMPLE), replace=False)
            sub = window.loc[window["symbol"].isin(keep)]
            try:
                rules, generated = discover_rules(
                    sub,
                    sub[target_col],
                    feature_cols,
                    seed=seed + 1000 * n_runs + s,
                    max_rules=max_rules,
                    max_fit_rows=max_fit_rows,
                    min_support=min_support,
                    min_symbols=min_symbols,
                )
            except ValidationError:
                continue
            trials += generated
            n_runs += 1
            for rule in rules:
                occurrences[rule.signature].append(rule)
    if n_runs == 0:
        raise ValidationError("no successful discovery runs")
    log_event(
        log,
        20,
        "stability runs complete",
        runs=n_runs,
        structural_rules=len(occurrences),
        trials=trials,
    )

    out: list[StableRule] = []
    inner_y = inner[target_col].to_numpy(dtype=float)
    inner_base = float(np.nanmean(inner_y))  # unconditional inner mean
    inner_mae = inner[mae_col].to_numpy(dtype=float) if mae_col and mae_col in inner else None
    inner_feats = inner[list(feature_cols)].astype(np.float64)
    inner_feats = inner_feats.fillna(discovery[list(feature_cols)].median(numeric_only=True))

    for signature, occ in sorted(occurrences.items(), key=lambda kv: -len(kv[1])):
        freq = len(occ) / n_runs
        signs = np.sign([r.coef for r in occ])
        sign_consistency = float(max((signs > 0).mean(), (signs < 0).mean()))
        direction = 1 if (signs > 0).mean() >= 0.5 else -1
        condition = _median_condition(occ)
        reasons: list[str] = []
        if freq < min_selection_freq:
            reasons.append(f"selection frequency {freq:.0%} < {min_selection_freq:.0%}")
        if sign_consistency < 0.9:
            reasons.append(f"sign consistency {sign_consistency:.0%} < 90%")

        preds = (condition,) if isinstance(condition, Predicate) else condition.conditions
        pred_list = [p for p in preds if isinstance(p, Predicate)]
        mask = _fast_mask(inner_feats, pred_list)
        sel = inner_y[mask & np.isfinite(inner_y)]
        if len(sel) >= 2:
            abs_mean = float(sel.mean())
            edge = abs_mean - inner_base  # relative edge = coef semantics
            sd = float(sel.std(ddof=1))
            t = edge / (sd / np.sqrt(len(sel))) if sd > 0 else 0.0
        else:
            abs_mean, edge, t = 0.0, 0.0, 0.0
        if len(sel) < min_inner_n:
            reasons.append(f"inner n {len(sel)} < {min_inner_n}")
        if not passes_cgs_hurdle(t):
            reasons.append(f"inner t {t:.2f} below CGS hurdle")
        if np.sign(t) != direction and len(sel) >= min_inner_n:
            reasons.append("inner direction contradicts discovery sign")
        mae_mean = (
            float(inner_mae[mask & np.isfinite(inner_mae)].mean())
            if inner_mae is not None and mask.any()
            else None
        )

        out.append(
            StableRule(
                signature=signature,
                condition=condition,
                direction=direction,
                selection_freq=freq,
                sign_consistency=sign_consistency,
                n_runs_present=len(occ),
                mean_support=float(np.mean([r.support for r in occ])),
                inner_n=len(sel),
                inner_mean_excess=edge,
                inner_abs_mean=abs_mean,
                inner_t=float(t),
                inner_mean_mae=mae_mean,
                passes=not reasons,
                fail_reasons=tuple(reasons),
            )
        )
    return out, trials
