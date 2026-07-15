"""Constrained RuleFit rule discovery (Friedman & Popescu 2008).

Implemented in-house for speed and control: shallow gradient-boosted trees
generate candidate rules (every root-to-leaf conjunction), and a bounded-path
Lasso selects a sparse subset. (imodels' RuleFitRegressor performs an
unbounded alpha search that takes minutes per fit — measured, not assumed —
so it is used only for distillation elsewhere.)

Constraints per the brief: shallow rules (<= max_conditions joined terms),
strong sparsity, minimum support in rows AND unique symbols, no future-derived
columns. Every rule the ensemble ever produces is counted as a trial.

Discovered rules are converted into EdgeStack's Condition AST so the entire
downstream stack (validation, engine replay, persistence) applies unchanged.
Each threshold also gets a train-quantile-decile signature so nearby
thresholds canonicalize to the same structural rule during stability selection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from edgestack.exceptions import ValidationError
from edgestack.types import AllOf, Condition, Predicate

_TERM_RE = re.compile(r"^\s*(?P<feat>[A-Za-z0-9_]+)\s*(?P<op><=|>=|<|>)\s*"
                      r"(?P<thr>-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)\s*$")

MAX_FIT_ROWS = 80_000


def _tree_rules(tree, feature_names: list[str],
                max_conditions: int) -> list[tuple[tuple[str, str, float], ...]]:
    """All root-to-leaf conjunctions of one fitted sklearn tree."""
    t = tree.tree_
    out: list[tuple[tuple[str, str, float], ...]] = []

    def simplify(path: tuple[tuple[str, str, float], ...]
                 ) -> tuple[tuple[str, str, float], ...]:
        """Collapse repeated same-feature bounds to the tightest one."""
        upper: dict[str, float] = {}
        lower: dict[str, float] = {}
        for feat, op, thr in path:
            if op == "<=":
                upper[feat] = min(thr, upper.get(feat, np.inf))
            else:
                lower[feat] = max(thr, lower.get(feat, -np.inf))
        terms = [(f, "<=", v) for f, v in upper.items()]
        terms += [(f, ">", v) for f, v in lower.items()]
        return tuple(sorted(terms))

    def walk(node: int, path: tuple[tuple[str, str, float], ...]) -> None:
        # Friedman-Popescu: EVERY node's path prefix is a candidate rule, not
        # just leaves — otherwise simple one-condition rules can never emerge
        # from fully-grown trees and stability selection starves.
        if path:
            simplified = simplify(path)
            if 0 < len(simplified) <= max_conditions:
                out.append(simplified)
        if t.children_left[node] == -1:  # leaf
            return
        feat = feature_names[t.feature[node]]
        thr = float(t.threshold[node])
        walk(t.children_left[node], (*path, (feat, "<=", thr)))
        walk(t.children_right[node], (*path, (feat, ">", thr)))

    walk(0, ())
    return out


@dataclass(frozen=True)
class DiscoveredRule:
    condition: Condition
    coef: float                    # sign = predicted direction of excess return
    support: float                 # fraction of training rows matched
    n_symbols: int
    raw: str
    signature: str                 # canonical structural identity


def parse_rule_string(raw: str, feature_cols: set[str]) -> list[Predicate] | None:
    """imodels rule strings look like 'mom_60 <= 0.08 and natr_14 > 0.02'."""
    preds: list[Predicate] = []
    for term in raw.split(" and "):
        m = _TERM_RE.match(term)
        if not m or m.group("feat") not in feature_cols:
            return None
        preds.append(Predicate(feature=m.group("feat"),
                               op=m.group("op"),  # type: ignore[arg-type]
                               value=float(m.group("thr"))))
    return preds or None


def rule_signature(preds: list[Predicate], decile_of: dict[str, np.ndarray],
                   coef: float) -> str:
    """Structural identity: feature+direction+threshold decile (+rule sign).

    vol<0.22, vol<0.24 and vol<0.25 land in the same decile bucket and merge.
    """
    parts = []
    for p in sorted(preds, key=lambda p: p.feature):
        edges = decile_of[p.feature]
        bucket = int(np.searchsorted(edges, float(p.value)))  # type: ignore[arg-type]
        direction = "<" if p.op in ("<", "<=") else ">"
        parts.append(f"{p.feature}{direction}d{bucket}")
    return "&".join(parts) + ("|+" if coef > 0 else "|-")


def discover_rules(
    features: pd.DataFrame,
    target: pd.Series,
    feature_cols: tuple[str, ...],
    *,
    seed: int = 42,
    max_rules: int = 300,
    max_conditions: int = 3,
    min_support: float = 0.005,
    min_symbols: int = 30,
    max_fit_rows: int = MAX_FIT_ROWS,
) -> tuple[list[DiscoveredRule], int]:
    """Fit constrained RuleFit; return (surviving rules, TOTAL rules generated).

    The second element is the honest trial count: every rule the ensemble
    produced, including ones filtered for depth/support, was a look at the data.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import ElasticNetCV

    cols = list(feature_cols)
    frame = features[cols].astype(np.float64)
    y = target.to_numpy(dtype=float)
    symbols = features["symbol"].to_numpy()
    ok = np.isfinite(y)
    frame, y, symbols = frame.loc[ok], y[ok], symbols[ok]
    frame = frame.fillna(frame.median(numeric_only=True))
    if len(frame) > max_fit_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(frame), size=max_fit_rows, replace=False)
        frame, y, symbols = frame.iloc[idx], y[idx], symbols[idx]
    if len(frame) < 2000:
        raise ValidationError("need >=2000 rows for RuleFit")

    # 1) Rule generation: shallow boosted trees on subsamples (Friedman-Popescu).
    n_estimators = max(40, max_rules // (2 ** (max_conditions - 1)))
    gbr = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_conditions,
        learning_rate=0.1, subsample=0.7, max_features=0.8, random_state=seed,
    )
    x_np = frame.to_numpy()
    gbr.fit(x_np, y)
    raw_rules: list[tuple[tuple[str, str, float], ...]] = []
    for est in gbr.estimators_.ravel():
        raw_rules.extend(_tree_rules(est, cols, max_conditions))
    total_generated = len(raw_rules)
    unique_rules = list(dict.fromkeys(raw_rules))[: 2 * max_rules]
    if not unique_rules:
        return [], total_generated

    # 2) Rule indicator matrix + bounded-path Lasso for sparse selection.
    preds_per_rule = [
        [Predicate(feature=f, op=op, value=thr)  # type: ignore[arg-type]
         for f, op, thr in rule]
        for rule in unique_rules
    ]
    indicator = np.column_stack([
        _fast_mask(frame, preds).astype(np.float32) for preds in preds_per_rule
    ])
    lasso = ElasticNetCV(
        l1_ratio=1.0, alphas=20, cv=3, tol=1e-3, max_iter=3000,
        selection="random", random_state=seed, n_jobs=-1,
    )
    lasso.fit(indicator, y - y.mean())
    coefs = lasso.coef_

    decile_of = {
        c: np.quantile(frame[c].to_numpy(), np.linspace(0.1, 0.9, 9)) for c in cols
    }
    order = np.argsort(-np.abs(coefs))
    out: list[DiscoveredRule] = []
    seen: set[str] = set()
    for j in order:
        coef = float(coefs[j])
        if coef == 0.0 or len(out) >= max_rules:
            break
        preds = preds_per_rule[j]
        # Prune near-vacuous companion conditions (a predicate matching ~90%
        # of rows filters nothing and only fragments the structural identity).
        if len(preds) > 1:
            kept = [p for p in preds
                    if float(_fast_mask(frame, [p]).mean()) < 0.9]
            preds = kept or preds
        mask = _fast_mask(frame, preds)
        support = float(mask.mean())
        n_syms = int(pd.unique(symbols[mask]).size)
        if support < min_support or n_syms < min_symbols:
            continue
        sig = rule_signature(preds, decile_of, coef)
        if sig in seen:
            continue
        seen.add(sig)
        cond: Condition = preds[0] if len(preds) == 1 else AllOf(conditions=tuple(preds))
        raw = " and ".join(f"{p.feature} {p.op} {float(p.value):.6g}" for p in preds)
        out.append(DiscoveredRule(condition=cond, coef=coef, support=support,
                                  n_symbols=n_syms, raw=raw, signature=sig))
    return out, total_generated


def prune_vacuous(preds: list[Predicate], frame: pd.DataFrame) -> list[Predicate]:
    """Public helper mirroring the in-fit pruning (used by tests)."""
    if len(preds) <= 1:
        return preds
    kept = [p for p in preds if float(_fast_mask(frame, [p]).mean()) < 0.9]
    return kept or preds


def _fast_mask(frame: pd.DataFrame, preds: list[Predicate]) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for p in preds:
        col = frame[p.feature].to_numpy()
        thr = float(p.value)  # type: ignore[arg-type]
        if p.op == "<":
            mask &= col < thr
        elif p.op == "<=":
            mask &= col <= thr
        elif p.op == ">":
            mask &= col > thr
        else:
            mask &= col >= thr
    return mask
