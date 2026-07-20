"""Filing-time EPS-change event study with exact next-open availability."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, date, datetime
from itertools import product
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgestack.config import EdgeStackConfig
from edgestack.data.calendar import TradingCalendar
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
from edgestack.research.cross_asset_momentum import (
    _adjusted_open_matrix,
    _bootstrap_counts,
    _bootstrap_lower_bounds,
    _fold_growth,
    _next_open_frame,
    _risk_matched_comparator,
    _stationary_pvalues,
)
from edgestack.research.schemas import CampaignLifecycle, CampaignSummaryV1, ShadowStrategyV1
from edgestack.research.store import ResearchStore
from edgestack.validation.advanced_tests import spa_test, stepm_superior
from edgestack.validation.metrics import deflated_sharpe_ratio, max_drawdown, sharpe_ratio


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EarningsFeatureDefinition(_FrozenModel):
    event: str
    surprise: str
    availability: str
    pre_event_momentum: str
    method: Literal["seasonal_random_walk_sue"] | None = None


class EarningsGrid(_FrozenModel):
    minimum_eps_change: tuple[float, ...]
    require_positive_pre_event_momentum: tuple[bool, ...]
    horizon_sessions: tuple[int, ...]

    @property
    def trial_count(self) -> int:
        return (
            len(self.minimum_eps_change)
            * len(self.require_positive_pre_event_momentum)
            * len(self.horizon_sessions)
        )


class EarningsPortfolioDefinition(_FrozenModel):
    weighting: str
    overlap: str
    leverage: float = Field(gt=0, le=1)


class EarningsFilingManifest(_FrozenModel):
    schema_version: Literal[1]
    campaign_id: str
    created_at: datetime
    name: str
    family: Literal["earnings_filing_drift"]
    previously_accessed: bool
    promotion_eligible: bool
    promotion_blockers: tuple[str, ...]
    research_start: date
    research_end: date
    final_holdout_start: date
    symbols: tuple[str, ...]
    benchmark_symbol: str
    diversified_baseline_symbols: tuple[str, ...]
    feature_definition: EarningsFeatureDefinition
    parameter_grid: EarningsGrid
    portfolio_definition: EarningsPortfolioDefinition
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
    def _historical_only(self) -> EarningsFilingManifest:
        if self.research_end >= self.final_holdout_start:
            raise ValueError("research interval must end before the final holdout")
        if not self.previously_accessed or self.promotion_eligible:
            raise ValueError("accessed filing study must remain promotion-ineligible")
        if self.required_growth_comparators != REQUIRED_GROWTH_COMPARATORS:
            raise ValueError("all canonical growth comparators are required in order")
        if self.parameter_grid.trial_count > 1_500:
            raise ValueError("earnings campaign exceeds the 1,500-trial ceiling")
        if self.minimum_positive_folds > self.validation_folds:
            raise ValueError("minimum positive folds exceeds validation folds")
        return self

    @property
    def manifest_hash(self) -> str:
        payload = self.model_dump(mode="json")
        if self.feature_definition.method is None:
            # Keep the already-published v1 identity unchanged.  The method
            # discriminator exists only for later, globally charged mutations.
            del payload["feature_definition"]["method"]
        return stable_hash(payload)

    def candidates(self) -> tuple[dict[str, int | float | bool], ...]:
        grid = self.parameter_grid
        return tuple(
            {
                "minimum_eps_change": threshold,
                "require_positive_pre_event_momentum": require_momentum,
                "horizon_sessions": horizon,
            }
            for threshold, require_momentum, horizon in product(
                grid.minimum_eps_change,
                grid.require_positive_pre_event_momentum,
                grid.horizon_sessions,
            )
        )


def load_earnings_manifest(path: Path) -> EarningsFilingManifest:
    return EarningsFilingManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _cash_yield(catalog: DataCatalog, manifest: EarningsFilingManifest) -> pd.Series:
    path = catalog.data_dir / "curated" / "macro" / f"{manifest.cash_yield_series}.parquet"
    if not path.exists():
        raise DataError(f"missing cash-yield input: {path}")
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


def _first_open_after(
    accepted_at: pd.Timestamp,
    sessions: pd.DatetimeIndex,
    opens_ns: np.ndarray,
) -> pd.Timestamp | None:
    accepted = pd.Timestamp(accepted_at)
    if accepted.tzinfo is None:
        accepted = accepted.tz_localize("UTC")
    else:
        accepted = accepted.tz_convert("UTC")
    location = int(np.searchsorted(opens_ns, accepted.value, side="right"))
    return pd.Timestamp(sessions[location]) if location < len(sessions) else None


def seasonal_random_walk_sue(events: pd.DataFrame) -> pd.Series:
    """Scale the current seasonal EPS difference using only prior differences."""
    seasonal_gap = events["period_end"].sub(events["period_end"].shift(4)).dt.days
    seasonal_difference = (
        events["eps"].sub(events["eps"].shift(4)).where(seasonal_gap.between(330, 400))
    )
    prior_scale = seasonal_difference.shift(1).rolling(8, min_periods=6).std()
    return seasonal_difference.div(prior_scale).clip(-10.0, 10.0)


def build_filing_event_table(
    catalog: DataCatalog,
    manifest: EarningsFilingManifest,
    adjusted_close: pd.DataFrame,
) -> pd.DataFrame:
    """Create features using only facts available before each executable open."""
    sessions = pd.DatetimeIndex(adjusted_close.index)
    calendar = TradingCalendar(catalog.cfg.data.calendar)
    opens_ns = np.asarray(
        [calendar.market_open_at(session).tz_convert("UTC").value for session in sessions],
        dtype=np.int64,
    )
    locations = pd.Series(np.arange(len(sessions)), index=sessions)
    rows: list[dict[str, Any]] = []
    for symbol in manifest.symbols:
        path = catalog.data_dir / "curated" / "events" / f"{symbol}.parquet"
        if not path.exists():
            continue
        events = pd.read_parquet(path)
        required = {"accepted_at", "eps"}
        if not required.issubset(events.columns):
            continue
        columns = ["accepted_at", "eps"]
        if manifest.feature_definition.method == "seasonal_random_walk_sue":
            if "period_end" not in events:
                continue
            columns.append("period_end")
        events = events.loc[:, columns].copy()
        events["accepted_at"] = pd.to_datetime(events["accepted_at"], utc=True)
        events["eps"] = pd.to_numeric(events["eps"], errors="coerce")
        events = events.dropna().sort_values("accepted_at").drop_duplicates("accepted_at")
        if manifest.feature_definition.method == "seasonal_random_walk_sue":
            events["period_end"] = pd.to_datetime(events["period_end"])
            events = events.sort_values(["period_end", "accepted_at"]).drop_duplicates(
                "period_end", keep="first"
            )
            events["eps_change"] = seasonal_random_walk_sue(events)
        else:
            events["lag4_eps"] = events["eps"].shift(4)
            denominator = events["lag4_eps"].abs().clip(lower=0.25)
            events["eps_change"] = ((events["eps"] - events["lag4_eps"]) / denominator).clip(
                -10.0, 10.0
            )
        for event in events.itertuples(index=False):
            item = cast(Any, event)
            execution_date = _first_open_after(item.accepted_at, sessions, opens_ns)
            if execution_date is None or pd.isna(item.eps_change):
                continue
            if not (manifest.research_start <= execution_date.date() <= manifest.research_end):
                continue
            location = int(locations.loc[execution_date])
            if location < 127 or symbol not in adjusted_close:
                continue
            prior_close = adjusted_close[symbol].iloc[location - 1]
            momentum_base = adjusted_close[symbol].iloc[location - 127]
            if pd.isna(prior_close) or pd.isna(momentum_base) or momentum_base <= 0:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "accepted_at": pd.Timestamp(item.accepted_at).isoformat(),
                    "execution_date": execution_date,
                    "eps_change": float(item.eps_change),
                    "pre_event_momentum_126": float(prior_close / momentum_base - 1.0),
                }
            )
    return pd.DataFrame(rows).sort_values(["execution_date", "symbol"]).reset_index(drop=True)


def filing_portfolio_returns(
    adjusted_open: pd.DataFrame,
    cash_yield: pd.Series,
    events: pd.DataFrame,
    candidate: dict[str, int | float | bool],
    *,
    roundtrip_cost_bps: float,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    asset_returns = _next_open_frame(adjusted_open)
    entries = pd.DataFrame(False, index=asset_returns.index, columns=adjusted_open.columns)
    threshold = float(candidate["minimum_eps_change"])
    eligible = events.loc[events["eps_change"] >= threshold]
    if bool(candidate["require_positive_pre_event_momentum"]):
        eligible = eligible.loc[eligible["pre_event_momentum_126"] > 0]
    for event in eligible.itertuples(index=False):
        item = cast(Any, event)
        session = pd.Timestamp(item.execution_date)
        if session in entries.index and item.symbol in entries.columns:
            entries.loc[session, item.symbol] = True
    horizon = int(candidate["horizon_sessions"])
    active = entries.astype(float).rolling(horizon, min_periods=1).max()
    available = asset_returns.notna().astype(float)
    active *= available
    count = active.sum(axis=1)
    weights = active.div(count.replace(0.0, np.nan), axis=0).fillna(0.0)
    risky_weight = weights.sum(axis=1).clip(upper=1.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    daily_cash = cash_yield.reindex(asset_returns.index).ffill().fillna(0.0) / 252.0
    returns = (
        (weights * asset_returns.fillna(0.0)).sum(axis=1)
        + (1.0 - risky_weight) * daily_cash
        - turnover * roundtrip_cost_bps / 20_000.0
    )
    returns.name = stable_hash(candidate)[:16]
    return returns, weights, turnover


def _trial_id(manifest: EarningsFilingManifest, candidate: dict[str, Any]) -> str:
    return stable_hash({"campaign_id": manifest.campaign_id, "parameters": candidate})[:24]


def register_earnings_campaign(
    store: ResearchStore,
    manifest: EarningsFilingManifest,
) -> CampaignSummaryV1:
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
            next_action="Evaluate the frozen filing-time family without opening the holdout.",
            promotion_eligible=False,
            previously_accessed=True,
        )
        store.upsert_campaign(current)
    elif current.manifest_hash != manifest.manifest_hash:
        raise DataError("earnings campaign manifest changed without a version bump")
    trials = [
        TrialRecordV2(
            trial_id=_trial_id(manifest, candidate),
            experiment_id=manifest.campaign_id,
            kind=(
                TrialKind.INTERACTION
                if bool(candidate["require_positive_pre_event_momentum"])
                else TrialKind.STANDALONE
            ),
            family=manifest.family,
            horizon_sessions=int(candidate["horizon_sessions"]),
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
        row = con.execute(
            "SELECT COUNT(*) FROM trial_ledger_v2 WHERE experiment_id = ?",
            [manifest.campaign_id],
        ).fetchone()
    if row is None or int(row[0]) != len(candidates):
        raise DataError("earnings trial ledger is incomplete")
    return current


def _scoped_data_hash(catalog: DataCatalog, manifest: EarningsFilingManifest) -> str:
    digest = hashlib.sha256()
    paths = [
        safe_child_path(catalog.prices_dir, f"{symbol}.parquet")
        for symbol in (
            *manifest.symbols,
            manifest.benchmark_symbol,
            *manifest.diversified_baseline_symbols,
        )
    ]
    paths.extend(
        catalog.data_dir / "curated" / "events" / f"{symbol}.parquet" for symbol in manifest.symbols
    )
    paths.append(catalog.data_dir / "curated" / "macro" / f"{manifest.cash_yield_series}.parquet")
    for path in sorted(set(paths)):
        if not path.exists():
            raise DataError(f"missing frozen earnings input: {path}")
        relative = path.resolve().relative_to(catalog.data_dir.resolve())
        digest.update(relative.as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_artifact(
    catalog: DataCatalog,
    run_id: str,
    report: dict[str, Any],
) -> tuple[str, str, Path]:
    encoded = json.dumps(report, indent=2, sort_keys=True).encode()
    artifact_hash = hashlib.sha256(encoded).hexdigest()
    trial_content_hash = stable_hash(report["results"])
    namespace = (
        "earnings_filing_drift"
        if report["campaign_id"] == "earnings-filing-drift-2011-2022-v1"
        else str(report["campaign_id"])
    )
    root = catalog.artifacts_dir / "research" / namespace
    destination = root / run_id
    report_path = destination / "report.json"
    if report_path.exists() and report_path.read_bytes() != encoded:
        raise DataError("deterministic earnings rerun produced different content")
    atomic_write_bytes(report_path, encoded)
    buffer = io.BytesIO()
    pd.DataFrame(report["results"]).to_parquet(buffer, index=False)
    atomic_write_bytes(destination / "trials.parquet", buffer.getvalue())
    atomic_write_bytes(
        root / "current.json",
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


def evaluate_earnings_campaign(
    store: ResearchStore,
    manifest: EarningsFilingManifest,
) -> dict[str, Any]:
    symbols = tuple(
        dict.fromkeys(
            (
                *manifest.symbols,
                manifest.benchmark_symbol,
                *manifest.diversified_baseline_symbols,
            )
        )
    )
    panel = store.catalog.load_panel(
        symbols=symbols,
        start=manifest.research_start,
        end=manifest.research_end,
    )
    close = panel.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    adjusted_open = _adjusted_open_matrix(panel)
    benchmark_index = close[manifest.benchmark_symbol].dropna().index
    benchmark_index = benchmark_index.intersection(
        adjusted_open[manifest.benchmark_symbol].dropna().index
    )
    close = close.reindex(benchmark_index)
    adjusted_open = adjusted_open.reindex(benchmark_index)
    cash_yield = _cash_yield(store.catalog, manifest)
    events = build_filing_event_table(store.catalog, manifest, close)
    if events.empty:
        raise DataError("no filing events survived the frozen availability rules")
    candidates = manifest.candidates()
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
        base, weights, turnover = filing_portfolio_returns(
            adjusted_open,
            cash_yield,
            events,
            candidate,
            roundtrip_cost_bps=manifest.base_roundtrip_cost_bps,
        )
        stress, _, _ = filing_portfolio_returns(
            adjusted_open,
            cash_yield,
            events,
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
    columns = [stable_hash(candidate)[:16] for candidate in candidates]
    strategy_frame = pd.concat(base_returns, axis=1)
    strategy_frame.columns = columns
    stress_frame = pd.concat(stress_returns, axis=1)
    stress_frame.columns = columns
    risk_frame = pd.concat(risk_matched, axis=1)
    risk_frame.columns = columns
    financed_frame = pd.concat(financed, axis=1)
    financed_frame.columns = columns
    spy_frame = pd.DataFrame(
        np.repeat(spy.to_numpy()[:, None], len(candidates), axis=1),
        index=spy.index,
        columns=columns,
    )
    diversified_frame = pd.DataFrame(
        np.repeat(diversified.to_numpy()[:, None], len(candidates), axis=1),
        index=diversified.index,
        columns=columns,
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
        candidate_id = columns[index]
        strategy = strategy_frame.iloc[:, index]
        stress = stress_frame.iloc[:, index]
        risk = risk_frame.iloc[:, index]
        weights = weights_by_candidate[index]
        turnover = turnover_by_candidate[index]
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
                "eligible_events": int(
                    (
                        (events["eps_change"] >= float(candidate["minimum_eps_change"]))
                        & (
                            (events["pre_event_momentum_126"] > 0)
                            if bool(candidate["require_positive_pre_event_momentum"])
                            else True
                        )
                    ).sum()
                ),
                "invested_sessions": int((risky > 0).sum()),
                "mean_risky_weight": float(risky.mean()),
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
        "available_filing_events": len(events),
        "global_family_trial_count": global_trials,
        "family_tests": {
            "spa": spa,
            "stepm_superior_candidate_ids": sorted(superior),
        },
        "historical_qualifier_ids": qualifier_ids,
        "results": results,
        "limitations": [
            *manifest.promotion_blockers,
            "EPS change is a filing-time signal, not a consensus-estimate surprise.",
            "Historical qualification can create only an isolated prospective paper book.",
        ],
    }
    artifact_hash, trial_content_hash, destination = _write_artifact(store.catalog, run_id, report)
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
        raise DataError("earnings campaign disappeared during evaluation")
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
    updated_campaign = current.model_copy(
        update={
            "lifecycle": lifecycle,
            "updated_at": now,
            "completed_trials": len(results),
            "failure_reasons": (
                ()
                if qualifier_ids
                else (
                    "No filing-time candidate passed FDR, SPA, StepM, four growth bounds, "
                    "deflated Sharpe, fold, stress, and drawdown gates.",
                )
            ),
            "next_action": (
                "Run qualifiers only in isolated prospective paper books."
                if qualifier_ids
                else "Preserve the rejection and advance to the next frozen family."
            ),
            "artifact_hash": artifact_hash,
            "metrics": {
                "evaluated": len(results),
                "available_filing_events": len(events),
                "historical_qualifiers": len(qualifier_ids),
                "family_spa_pvalue": float(spa["p_consistent"]),
                "best_candidate_id": str(best["candidate_id"]),
                "best_calendar_log_growth": float(best["calendar_log_growth"]),
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
                    "Previously accessed history; promotion is disabled.",
                    "Requires a new independent filing-time prospective period.",
                ),
            )
        )
    return report


def run_earnings_filing_campaign(
    cfg: EdgeStackConfig,
    *,
    manifest_path: Path = Path("configs/earnings_filing_drift.yaml"),
    verify_recompute: bool = False,
) -> dict[str, Any]:
    manifest = load_earnings_manifest(manifest_path)
    if manifest.final_holdout_start != cfg.validation.final_test_start:
        raise DataError("earnings manifest holdout differs from the configured guard")
    catalog = DataCatalog(cfg)
    store = ResearchStore(catalog)
    current = register_earnings_campaign(store, manifest)
    if (
        current.lifecycle in {CampaignLifecycle.REJECTED, CampaignLifecycle.PAPER_SHADOW}
        and not verify_recompute
    ):
        namespace = (
            "earnings_filing_drift"
            if manifest.campaign_id == "earnings-filing-drift-2011-2022-v1"
            else manifest.campaign_id
        )
        pointer_path = catalog.artifacts_dir / "research" / namespace / "current.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        report_path = (
            catalog.artifacts_dir / "research" / namespace / str(pointer["run_id"]) / "report.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["artifact_hash"] = pointer["artifact_hash"]
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
    return evaluate_earnings_campaign(store, manifest)
