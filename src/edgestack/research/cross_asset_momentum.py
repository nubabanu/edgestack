"""Frozen cross-asset momentum campaign with next-open execution.

The historical interval used by this study has already been accessed.  A
historical qualifier may therefore open an isolated paper-shadow book, but it
can never become a promoted sleeve from this run.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, date, datetime
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog, atomic_write_bytes, safe_child_path
from edgestack.discovery.multiple_testing import benjamini_hochberg
from edgestack.exceptions import DataError
from edgestack.recommendation.growth import (
    REQUIRED_GROWTH_COMPARATORS,
    annualized_log_growth,
    financed_risk_matched_spy,
)
from edgestack.recommendation.hashing import stable_hash
from edgestack.recommendation.manifests import TrialKind, TrialRecordV2, TrialStatus
from edgestack.research.schemas import CampaignLifecycle, CampaignSummaryV1, ShadowStrategyV1
from edgestack.research.store import ResearchStore
from edgestack.validation.advanced_tests import spa_test, stepm_superior
from edgestack.validation.clustered import stationary_bootstrap_indices
from edgestack.validation.metrics import deflated_sharpe_ratio, max_drawdown, sharpe_ratio


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CrossAssetGrid(_FrozenModel):
    lookback_sessions: tuple[int, ...]
    skip_sessions: tuple[int, ...]
    top_k: tuple[int, ...]
    trend_lookback_sessions: tuple[int, ...]
    volatility_window_sessions: tuple[int, ...]
    maximum_asset_weight: tuple[float, ...]

    @property
    def trial_count(self) -> int:
        dimensions = (
            self.lookback_sessions,
            self.skip_sessions,
            self.top_k,
            self.trend_lookback_sessions,
            self.volatility_window_sessions,
            self.maximum_asset_weight,
        )
        return int(np.prod([len(values) for values in dimensions]))


class CrossAssetPortfolioDefinition(_FrozenModel):
    rebalance: str
    weighting: str
    missingness: str
    overlap: str
    leverage: float = Field(gt=0, le=1)


class CrossAssetMomentumManifest(_FrozenModel):
    schema_version: Literal[1]
    campaign_id: str
    created_at: datetime
    name: str
    family: Literal["cross_asset_momentum"]
    previously_accessed: bool
    promotion_eligible: bool
    promotion_blockers: tuple[str, ...]
    research_start: date
    research_end: date
    final_holdout_start: date
    symbols: tuple[str, ...]
    benchmark_symbol: str
    diversified_baseline_symbols: tuple[str, ...]
    parameter_grid: CrossAssetGrid
    portfolio_definition: CrossAssetPortfolioDefinition
    base_roundtrip_cost_bps: float = Field(ge=0)
    stress_roundtrip_cost_bps: float = Field(ge=0)
    cash_yield_series: str
    financing_spread_bps: float = Field(ge=0)
    validation_folds: int = Field(ge=2)
    bootstrap_samples: int = Field(ge=100)
    bootstrap_block_sessions: int = Field(ge=1)
    fdr_alpha: float = Field(gt=0, lt=1)
    minimum_positive_folds: int = Field(ge=1)
    minimum_deflated_sharpe_probability: float = Field(ge=0.5, le=1)
    maximum_drawdown: float = Field(gt=0, le=1)
    seed: int
    required_growth_comparators: tuple[str, ...]

    @model_validator(mode="after")
    def _honest_historical_study(self) -> CrossAssetMomentumManifest:
        if self.research_end >= self.final_holdout_start:
            raise ValueError("research interval must end before the final holdout")
        if not self.previously_accessed or self.promotion_eligible:
            raise ValueError("this accessed historical study must remain promotion-ineligible")
        if self.required_growth_comparators != REQUIRED_GROWTH_COMPARATORS:
            raise ValueError("all canonical log-growth comparators are required in order")
        if self.parameter_grid.trial_count > 1_500:
            raise ValueError("campaign exceeds the frozen 1,500-trial ceiling")
        if self.minimum_positive_folds > self.validation_folds:
            raise ValueError("minimum positive folds exceeds validation folds")
        if self.stress_roundtrip_cost_bps < self.base_roundtrip_cost_bps:
            raise ValueError("stress costs cannot be below base costs")
        return self

    @property
    def manifest_hash(self) -> str:
        return stable_hash(self.model_dump(mode="json"))

    def candidates(self) -> tuple[dict[str, int | float], ...]:
        grid = self.parameter_grid
        return tuple(
            {
                "lookback_sessions": lookback,
                "skip_sessions": skip,
                "top_k": top_k,
                "trend_lookback_sessions": trend,
                "volatility_window_sessions": vol_window,
                "maximum_asset_weight": max_weight,
            }
            for lookback, skip, top_k, trend, vol_window, max_weight in product(
                grid.lookback_sessions,
                grid.skip_sessions,
                grid.top_k,
                grid.trend_lookback_sessions,
                grid.volatility_window_sessions,
                grid.maximum_asset_weight,
            )
        )


def load_cross_asset_manifest(path: Path) -> CrossAssetMomentumManifest:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return CrossAssetMomentumManifest.model_validate(payload)


def _adjusted_open_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    factor = (panel["adj_close"] / panel["close"]).replace([np.inf, -np.inf], np.nan)
    if bool(factor.isna().any()) or bool((factor <= 0).any()):
        raise DataError("cross-asset study has invalid adjusted-open factors")
    prepared = panel.assign(adjusted_open=panel["open"] * factor)
    return prepared.pivot(index="date", columns="symbol", values="adjusted_open").sort_index()


def _next_open_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Return from each indexed open to the following session's open."""
    return prices.shift(-1).div(prices).sub(1.0).iloc[:-1]


def build_monthly_momentum_weights(
    adjusted_close: pd.DataFrame,
    parameters: dict[str, int | float],
) -> pd.DataFrame:
    """Compute month-end-close targets, effective only at the next open."""
    close = adjusted_close.astype(float).sort_index()
    lookback = int(parameters["lookback_sessions"])
    skip = int(parameters["skip_sessions"])
    top_k = int(parameters["top_k"])
    trend_lookback = int(parameters["trend_lookback_sessions"])
    vol_window = int(parameters["volatility_window_sessions"])
    max_weight = float(parameters["maximum_asset_weight"])

    if skip:
        score = close.shift(skip).div(close.shift(skip + lookback)).sub(1.0)
    else:
        score = close.div(close.shift(lookback)).sub(1.0)
    volatility = close.pct_change(fill_method=None).rolling(vol_window).std()
    trend_ok = (
        close.gt(close.rolling(trend_lookback).mean())
        if trend_lookback
        else pd.DataFrame(True, index=close.index, columns=close.columns)
    )

    month_codes = pd.DatetimeIndex(close.index).to_period("M")
    month_ends = close.groupby(month_codes).tail(1).index
    targets: dict[pd.Timestamp, pd.Series] = {}
    locations = pd.Series(np.arange(len(close.index)), index=close.index)
    for signal_date in month_ends:
        signal_location = int(locations.loc[signal_date])
        if signal_location + 1 >= len(close.index):
            continue
        execution_date = pd.Timestamp(close.index[signal_location + 1])
        scores = score.loc[signal_date]
        vol = volatility.loc[signal_date]
        eligible = scores.gt(0) & trend_ok.loc[signal_date] & vol.gt(0)
        eligible &= scores.notna() & vol.notna()
        selected = scores.loc[eligible].nlargest(top_k).index
        target = pd.Series(0.0, index=close.columns, dtype=float)
        if len(selected):
            inverse_vol = 1.0 / vol.loc[selected]
            normalized = inverse_vol / inverse_vol.sum()
            target.loc[selected] = normalized.clip(upper=max_weight)
        targets[execution_date] = target

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for execution_date, target in targets.items():
        weights.loc[execution_date] = target
    rebalance_dates = pd.Series(False, index=weights.index)
    if targets:
        rebalance_dates.loc[list(targets)] = True
    weights.loc[~rebalance_dates] = np.nan
    return weights.ffill().fillna(0.0)


def cross_asset_portfolio_returns(
    adjusted_close: pd.DataFrame,
    adjusted_open: pd.DataFrame,
    cash_yield: pd.Series,
    parameters: dict[str, int | float],
    *,
    roundtrip_cost_bps: float,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Build a full calendar return stream including idle-cash carry and costs."""
    weights = build_monthly_momentum_weights(adjusted_close, parameters)
    asset_returns = _next_open_frame(adjusted_open)
    weights = weights.reindex(asset_returns.index).fillna(0.0)
    daily_cash = cash_yield.reindex(asset_returns.index).ffill().fillna(0.0) / 252.0
    risky_weight = weights.sum(axis=1).clip(lower=0.0, upper=1.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    half_roundtrip_cost = roundtrip_cost_bps / 20_000.0
    returns = (
        (weights * asset_returns).sum(axis=1)
        + (1.0 - risky_weight) * daily_cash
        - turnover * half_roundtrip_cost
    )
    returns.name = stable_hash(parameters)[:16]
    return returns, weights, turnover


def _cash_yield(catalog: DataCatalog, manifest: CrossAssetMomentumManifest) -> pd.Series:
    path = catalog.data_dir / "curated" / "macro" / f"{manifest.cash_yield_series}.parquet"
    if not path.exists():
        raise DataError(f"missing cash-yield data: {path}")
    frame = pd.read_parquet(path)
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    frame["realtime_start"] = pd.to_datetime(frame["realtime_start"])
    return (
        frame.sort_values("realtime_start")
        .drop_duplicates("observation_date", keep="first")
        .set_index("observation_date")["value"]
        .astype(float)
        .div(100.0)
        .sort_index()
    )


def _scoped_data_hash(catalog: DataCatalog, manifest: CrossAssetMomentumManifest) -> str:
    digest = hashlib.sha256()
    paths = [
        safe_child_path(catalog.prices_dir, f"{symbol}.parquet") for symbol in manifest.symbols
    ]
    paths.append(catalog.data_dir / "curated" / "macro" / f"{manifest.cash_yield_series}.parquet")
    for path in paths:
        if not path.exists():
            raise DataError(f"missing frozen campaign input: {path}")
        relative = path.resolve().relative_to(catalog.data_dir.resolve())
        digest.update(relative.as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _trial_id(manifest: CrossAssetMomentumManifest, parameters: dict[str, Any]) -> str:
    return stable_hash({"campaign_id": manifest.campaign_id, "parameters": parameters})[:24]


def register_cross_asset_campaign(
    store: ResearchStore,
    manifest: CrossAssetMomentumManifest,
) -> CampaignSummaryV1:
    """Persist the complete family before any price return is evaluated."""
    candidates = manifest.candidates()
    current = store.campaign(manifest.campaign_id)
    now = datetime.now(UTC)
    if current is None:
        current = CampaignSummaryV1(
            campaign_id=manifest.campaign_id,
            manifest_hash=manifest.manifest_hash,
            name=manifest.name,
            family=manifest.family,
            lifecycle=CampaignLifecycle.READY,
            created_at=now,
            updated_at=now,
            trial_count=len(candidates),
            next_action="Evaluate the frozen 144-trial family without opening the holdout.",
            promotion_eligible=False,
            previously_accessed=True,
        )
        store.upsert_campaign(current)
    elif current.manifest_hash != manifest.manifest_hash:
        raise DataError("cross-asset campaign manifest changed without a version bump")

    trials = [
        TrialRecordV2(
            trial_id=_trial_id(manifest, candidate),
            experiment_id=manifest.campaign_id,
            kind=TrialKind.STANDALONE,
            family=manifest.family,
            horizon_sessions=21,
            parameters=candidate,
            status=TrialStatus.REGISTERED,
        )
        for candidate in candidates
    ]
    with store.catalog.connect() as con:
        con.executemany(
            "INSERT OR IGNORE INTO trial_ledger_v2 VALUES (?, ?, ?, ?, ?)",
            [
                [
                    trial.trial_id,
                    manifest.campaign_id,
                    now,
                    trial.status.value,
                    trial.model_dump_json(),
                ]
                for trial in trials
            ],
        )
        registered_row = con.execute(
            "SELECT COUNT(*) FROM trial_ledger_v2 WHERE experiment_id = ?",
            [manifest.campaign_id],
        ).fetchone()
        registered = int(registered_row[0]) if registered_row is not None else 0
    if registered != len(candidates):
        raise DataError(f"trial ledger contains {registered}, expected {len(candidates)}")
    return current


def _bootstrap_counts(
    sessions: int,
    *,
    samples: int,
    block_sessions: int,
    seed: int,
) -> np.ndarray:
    indices = stationary_bootstrap_indices(
        sessions,
        n_boot=samples,
        mean_block_length=block_sessions,
        seed=seed,
    )
    counts = np.zeros((samples, sessions), dtype=np.float32)
    rows = np.repeat(np.arange(samples), sessions)
    np.add.at(counts, (rows, indices.ravel()), 1.0)
    return counts


def _bootstrap_lower_bounds(
    strategies: pd.DataFrame,
    comparators: dict[str, pd.DataFrame],
    counts: np.ndarray,
) -> dict[str, np.ndarray]:
    strategy_logs = np.log1p(strategies.to_numpy(dtype=float))
    lower: dict[str, np.ndarray] = {}
    for name in REQUIRED_GROWTH_COMPARATORS:
        comparator_logs = np.log1p(comparators[name].to_numpy(dtype=float))
        differences = strategy_logs - comparator_logs
        sampled = counts @ differences / len(strategies) * 252.0
        lower[name] = np.quantile(sampled, 0.05, axis=0)
    return lower


def _stationary_pvalues(active_returns: np.ndarray, counts: np.ndarray) -> np.ndarray:
    observed = active_returns.mean(axis=0)
    centered = active_returns - observed
    null_means = counts @ centered / len(active_returns)
    return ((null_means >= observed).sum(axis=0) + 1.0) / (len(counts) + 1.0)


def _risk_matched_comparator(
    strategy: pd.Series,
    spy: pd.Series,
    cash_yield: pd.Series,
) -> pd.Series:
    strategy_vol = float(strategy.std(ddof=1))
    spy_vol = float(spy.std(ddof=1))
    scale = strategy_vol / spy_vol if spy_vol > 0 else 0.0
    daily_cash = cash_yield.reindex(spy.index).ffill().fillna(0.0) / 252.0
    return spy * scale + daily_cash * max(0.0, 1.0 - scale)


def _fold_growth(returns: pd.Series, folds: int) -> tuple[float, ...]:
    return tuple(
        annualized_log_growth(pd.Series(chunk, dtype=float))
        for chunk in np.array_split(returns.to_numpy(dtype=float), folds)
        if len(chunk)
    )


def _write_artifacts(
    catalog: DataCatalog,
    *,
    run_id: str,
    report: dict[str, Any],
) -> tuple[str, str, Path]:
    report_bytes = json.dumps(report, indent=2, sort_keys=True).encode()
    artifact_hash = hashlib.sha256(report_bytes).hexdigest()
    trial_content_hash = stable_hash(report["results"])
    destination = catalog.artifacts_dir / "research" / "cross_asset_momentum" / run_id
    report_path = destination / "report.json"
    if report_path.exists() and report_path.read_bytes() != report_bytes:
        raise DataError("deterministic cross-asset run produced different content")
    atomic_write_bytes(report_path, report_bytes)
    trial_frame = pd.DataFrame(report["results"])
    buffer = io.BytesIO()
    trial_frame.to_parquet(buffer, index=False)
    atomic_write_bytes(destination / "trials.parquet", buffer.getvalue())
    atomic_write_bytes(
        catalog.artifacts_dir / "research" / "cross_asset_momentum" / "current.json",
        json.dumps(
            {
                "run_id": run_id,
                "artifact_hash": artifact_hash,
                "trial_content_hash": trial_content_hash,
            },
            indent=2,
            sort_keys=True,
        ).encode(),
    )
    return artifact_hash, trial_content_hash, destination


def evaluate_cross_asset_campaign(
    store: ResearchStore,
    manifest: CrossAssetMomentumManifest,
) -> dict[str, Any]:
    """Evaluate every frozen combination and apply complete-family gates."""
    candidates = manifest.candidates()
    panel = store.catalog.load_panel(
        manifest.symbols,
        start=manifest.research_start,
        end=manifest.research_end,
    )
    if pd.to_datetime(panel["date"]).max().date() > manifest.research_end:
        raise DataError("cross-asset evaluator crossed the frozen research end")
    close = panel.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    adjusted_open = _adjusted_open_matrix(panel)
    common = close.index.intersection(adjusted_open.index)
    close = close.reindex(common).dropna(subset=list(manifest.symbols))
    adjusted_open = adjusted_open.reindex(close.index).dropna(subset=list(manifest.symbols))
    close = close.reindex(adjusted_open.index)
    if len(close) < 756:
        raise DataError(f"cross-asset study has only {len(close)} complete sessions")
    cash_yield = _cash_yield(store.catalog, manifest)
    open_returns = _next_open_frame(adjusted_open)
    spy = open_returns[manifest.benchmark_symbol]
    diversified = open_returns.loc[:, list(manifest.diversified_baseline_symbols)].mean(axis=1)
    mean_cash = float(cash_yield.reindex(spy.index).ffill().fillna(0.0).mean())

    base_returns: list[pd.Series] = []
    stress_returns: list[pd.Series] = []
    weights_by_candidate: list[pd.DataFrame] = []
    turnover_by_candidate: list[pd.Series] = []
    risk_matched: list[pd.Series] = []
    financed: list[pd.Series] = []
    for candidate in candidates:
        base, weights, turnover = cross_asset_portfolio_returns(
            close,
            adjusted_open,
            cash_yield,
            candidate,
            roundtrip_cost_bps=manifest.base_roundtrip_cost_bps,
        )
        stress, _, _ = cross_asset_portfolio_returns(
            close,
            adjusted_open,
            cash_yield,
            candidate,
            roundtrip_cost_bps=manifest.stress_roundtrip_cost_bps,
        )
        base_returns.append(base)
        stress_returns.append(stress)
        weights_by_candidate.append(weights)
        turnover_by_candidate.append(turnover)
        risk = _risk_matched_comparator(base, spy, cash_yield)
        risk_matched.append(risk)
        financed.append(
            financed_risk_matched_spy(
                spy,
                target_volatility=float(base.std(ddof=1) * np.sqrt(252.0)),
                financing_rate=mean_cash + manifest.financing_spread_bps / 10_000.0,
                cash_yield=mean_cash,
            )
        )

    strategy_frame = pd.concat(base_returns, axis=1)
    stress_frame = pd.concat(stress_returns, axis=1)
    strategy_frame.columns = [stable_hash(item)[:16] for item in candidates]
    stress_frame.columns = strategy_frame.columns
    risk_frame = pd.concat(risk_matched, axis=1)
    risk_frame.columns = strategy_frame.columns
    financed_frame = pd.concat(financed, axis=1)
    financed_frame.columns = strategy_frame.columns
    spy_frame = pd.DataFrame(
        np.repeat(spy.to_numpy()[:, None], len(candidates), axis=1),
        index=spy.index,
        columns=strategy_frame.columns,
    )
    diversified_frame = pd.DataFrame(
        np.repeat(diversified.to_numpy()[:, None], len(candidates), axis=1),
        index=diversified.index,
        columns=strategy_frame.columns,
    )
    counts = _bootstrap_counts(
        len(strategy_frame),
        samples=manifest.bootstrap_samples,
        block_sessions=manifest.bootstrap_block_sessions,
        seed=manifest.seed,
    )
    lower_bounds = _bootstrap_lower_bounds(
        strategy_frame,
        {
            "buy_now_spy": spy_frame,
            "risk_matched_spy": risk_frame,
            "diversified_baseline": diversified_frame,
            "financed_spy_same_risk": financed_frame,
        },
        counts,
    )
    raw_pvalues = _stationary_pvalues(strategy_frame.to_numpy() - risk_frame.to_numpy(), counts)
    global_trials = max(
        len(candidates),
        store.family_trial_count_as_of(manifest.campaign_id, manifest.family),
    )
    qvalues = benjamini_hochberg(raw_pvalues)
    spa = spa_test(
        spy,
        strategy_frame,
        reps=manifest.bootstrap_samples,
        block_size=manifest.bootstrap_block_sessions,
        seed=manifest.seed + 100,
    )
    superior = set(
        stepm_superior(
            spy,
            strategy_frame,
            reps=manifest.bootstrap_samples,
            block_size=manifest.bootstrap_block_sessions,
            size=manifest.fdr_alpha,
            seed=manifest.seed + 200,
        )
    )

    results: list[dict[str, Any]] = []
    qualifier_ids: list[str] = []
    for index, candidate in enumerate(candidates):
        candidate_id = strategy_frame.columns[index]
        strategy = strategy_frame.iloc[:, index]
        stress = stress_frame.iloc[:, index]
        risk = risk_frame.iloc[:, index]
        weights = weights_by_candidate[index].reindex(strategy.index).fillna(0.0)
        turnover = turnover_by_candidate[index].reindex(strategy.index).fillna(0.0)
        folds = _fold_growth(strategy, manifest.validation_folds)
        positive_folds = sum(value > 0 for value in folds)
        dsr = deflated_sharpe_ratio(strategy.to_numpy(), n_trials=global_trials)
        drawdown = max_drawdown(strategy.to_numpy())
        growth_bounds = {
            name: float(lower_bounds[name][index]) for name in REQUIRED_GROWTH_COMPARATORS
        }
        failures = []
        if float(qvalues[index]) > manifest.fdr_alpha:
            failures.append("complete-family FDR q-value exceeds alpha")
        if float(spa["p_consistent"]) > manifest.fdr_alpha:
            failures.append("family SPA consistent p-value exceeds alpha")
        if candidate_id not in superior:
            failures.append("candidate was not selected by StepM")
        failed_growth = [name for name, value in growth_bounds.items() if value <= 0]
        if failed_growth:
            failures.append(
                "non-positive log-growth lower bound versus: " + ", ".join(failed_growth)
            )
        if dsr < manifest.minimum_deflated_sharpe_probability:
            failures.append("deflated Sharpe probability is below the frozen threshold")
        if positive_folds < manifest.minimum_positive_folds:
            failures.append("too few positive validation folds")
        stress_growth = annualized_log_growth(stress)
        risk_growth = annualized_log_growth(risk)
        if stress_growth <= risk_growth:
            failures.append("stressed log growth does not beat risk-matched SPY")
        if drawdown < -manifest.maximum_drawdown:
            failures.append("maximum drawdown exceeds the frozen limit")
        historically_qualified = not failures
        if historically_qualified:
            qualifier_ids.append(candidate_id)
        risky = weights.sum(axis=1)
        results.append(
            {
                "trial_id": _trial_id(manifest, candidate),
                "candidate_id": candidate_id,
                **candidate,
                "sessions": len(strategy),
                "invested_sessions": int((risky > 0).sum()),
                "mean_risky_weight": float(risky.mean()),
                "maximum_realized_asset_weight": float(weights.max().max()),
                "annual_turnover": float(turnover.mean() * 252.0),
                "calendar_log_growth": annualized_log_growth(strategy),
                "stress_log_growth": stress_growth,
                "spy_log_growth": annualized_log_growth(spy),
                "risk_matched_spy_log_growth": risk_growth,
                "diversified_log_growth": annualized_log_growth(diversified),
                "financed_spy_log_growth": annualized_log_growth(financed_frame.iloc[:, index]),
                **{f"log_growth_lower_vs_{name}": value for name, value in growth_bounds.items()},
                "raw_pvalue": float(raw_pvalues[index]),
                "q_value": float(qvalues[index]),
                "deflated_sharpe_probability": float(dsr),
                "sharpe": sharpe_ratio(strategy.to_numpy()),
                "maximum_drawdown": float(drawdown),
                "fold_log_growth": list(folds),
                "positive_folds": positive_folds,
                "stepm_superior": candidate_id in superior,
                "historically_qualified": historically_qualified,
                "failure_reasons": failures,
            }
        )

    data_hash = _scoped_data_hash(store.catalog, manifest)
    run_id = stable_hash({"manifest_hash": manifest.manifest_hash, "data_hash": data_hash})[:24]
    report: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": manifest.campaign_id,
        "run_id": run_id,
        "manifest_hash": manifest.manifest_hash,
        "data_hash": data_hash,
        "research_start": manifest.research_start.isoformat(),
        "research_end": manifest.research_end.isoformat(),
        "final_holdout_accessed": False,
        "previously_accessed": True,
        "promotion_eligible": False,
        "registered_trials": len(candidates),
        "completed_trials": len(results),
        "global_family_trial_count": global_trials,
        "family_tests": {
            "spa": spa,
            "stepm_superior_candidate_ids": sorted(superior),
        },
        "historical_qualifier_ids": qualifier_ids,
        "results": results,
        "limitations": [
            *manifest.promotion_blockers,
            "DGS3MO is a current FRED snapshot rather than an ALFRED vintage archive.",
            "Historical qualification can create only an isolated prospective paper book.",
        ],
    }
    artifact_hash, trial_content_hash, destination = _write_artifacts(
        store.catalog, run_id=run_id, report=report
    )
    report["artifact_hash"] = artifact_hash
    report["trial_content_hash"] = trial_content_hash
    report["artifact_path"] = str(destination)

    now = datetime.now(UTC)
    by_trial = {item["trial_id"]: item for item in results}
    with store.catalog.connect() as con:
        rows = con.execute(
            "SELECT trial_id, payload FROM trial_ledger_v2 WHERE experiment_id = ?",
            [manifest.campaign_id],
        ).fetchall()
        for trial_id, payload in rows:
            trial = TrialRecordV2.model_validate_json(payload)
            result = by_trial[str(trial_id)]
            status = (
                TrialStatus.SUCCEEDED
                if bool(result["historically_qualified"])
                else TrialStatus.REJECTED
            )
            reason = (
                None if status is TrialStatus.SUCCEEDED else "; ".join(result["failure_reasons"])
            )
            updated = trial.model_copy(update={"status": status, "failure_reason": reason})
            con.execute(
                """UPDATE trial_ledger_v2
                   SET created_at = ?, status = ?, payload = ? WHERE trial_id = ?""",
                [now, status.value, updated.model_dump_json(), trial_id],
            )

    current = store.campaign(manifest.campaign_id)
    if current is None:
        raise DataError("cross-asset campaign disappeared during evaluation")
    lifecycle = CampaignLifecycle.PAPER_SHADOW if qualifier_ids else CampaignLifecycle.REJECTED
    best = min(
        results,
        key=lambda item: (
            len(item["failure_reasons"]),
            float(item["q_value"]),
            -float(item["calendar_log_growth"]),
            str(item["candidate_id"]),
        ),
    )
    failure_reasons = (
        ()
        if qualifier_ids
        else (
            "No candidate passed FDR, SPA, StepM, four lower-bound log-growth comparators, "
            "deflated Sharpe, fold consistency, stress, and drawdown gates.",
        )
    )
    updated_campaign = current.model_copy(
        update={
            "lifecycle": lifecycle,
            "updated_at": now,
            "completed_trials": len(results),
            "failure_reasons": failure_reasons,
            "next_action": (
                "Run historical qualifiers only in isolated prospective paper books."
                if qualifier_ids
                else "Preserve the rejection and advance to the next preregistered family."
            ),
            "artifact_hash": artifact_hash,
            "metrics": {
                "evaluated": len(results),
                "historical_qualifiers": len(qualifier_ids),
                "family_spa_pvalue": float(spa["p_consistent"]),
                "best_calendar_log_growth": max(
                    float(item["calendar_log_growth"]) for item in results
                ),
                "best_candidate_id": str(best["candidate_id"]),
                "best_q_value": float(best["q_value"]),
                "best_deflated_sharpe_probability": float(best["deflated_sharpe_probability"]),
                "best_maximum_drawdown": float(best["maximum_drawdown"]),
                "best_gate_failures": "; ".join(best["failure_reasons"]),
            },
        }
    )
    store.upsert_campaign(updated_campaign)
    for candidate_id in qualifier_ids:
        store.upsert_shadow(
            ShadowStrategyV1(
                strategy_id=f"{manifest.campaign_id}:{candidate_id}",
                campaign_id=manifest.campaign_id,
                artifact_hash=artifact_hash,
                status=CampaignLifecycle.PAPER_SHADOW,
                started_at=now,
                equity=100_000.0,
                benchmark_returns={
                    "SPY": 0.0,
                    "RISK_MATCHED_SPY": 0.0,
                    "DIVERSIFIED_BASELINE": 0.0,
                    "FINANCED_SPY_SAME_RISK": 0.0,
                },
                warnings=(
                    "Previously accessed historical interval; promotion is disabled.",
                    "Requires a new independent prospective paper period.",
                ),
            )
        )
    return report


def run_cross_asset_momentum_campaign(
    cfg: EdgeStackConfig,
    *,
    manifest_path: Path = Path("configs/cross_asset_momentum.yaml"),
    verify_recompute: bool = False,
) -> dict[str, Any]:
    manifest = load_cross_asset_manifest(manifest_path)
    if manifest.final_holdout_start != cfg.validation.final_test_start:
        raise DataError("manifest holdout boundary differs from the configured guard")
    catalog = DataCatalog(cfg)
    store = ResearchStore(catalog)
    current = register_cross_asset_campaign(store, manifest)
    if (
        current.lifecycle in {CampaignLifecycle.REJECTED, CampaignLifecycle.PAPER_SHADOW}
        and not verify_recompute
    ):
        if current.artifact_hash is None:
            raise DataError("terminal cross-asset campaign has no artifact hash")
        pointer_path = catalog.artifacts_dir / "research" / "cross_asset_momentum" / "current.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        report_path = (
            catalog.artifacts_dir
            / "research"
            / "cross_asset_momentum"
            / str(pointer["run_id"])
            / "report.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["artifact_hash"] = current.artifact_hash
        report["trial_content_hash"] = pointer["trial_content_hash"]
        report["artifact_path"] = str(report_path.parent)
        report["cached"] = True
        return report
    if current.lifecycle is CampaignLifecycle.READY:
        store.upsert_campaign(
            current.model_copy(
                update={"lifecycle": CampaignLifecycle.RUNNING, "updated_at": datetime.now(UTC)}
            )
        )
    return evaluate_cross_asset_campaign(store, manifest)
