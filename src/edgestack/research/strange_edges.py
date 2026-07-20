"""Pre-registered evaluation of the user-supplied strange-edge hypotheses.

Only inputs that are actually present and legally usable are evaluated.  The
first campaign uses USAspending monthly prime-contract obligations.  Every
other proposed standalone or combination remains visible with its exact data
blocker; no price-derived substitute is allowed to masquerade as alternative
data.
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
import requests
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog, atomic_write_bytes
from edgestack.data.providers.usaspending import USAspendingContractProvider
from edgestack.discovery.multiple_testing import benjamini_hochberg
from edgestack.exceptions import DataError, ProviderError, ValidationError
from edgestack.recommendation.growth import (
    annualized_log_growth,
    financed_risk_matched_spy,
    log_growth_superiority,
)
from edgestack.recommendation.hashing import canonical_json, file_sha256, stable_hash
from edgestack.recommendation.manifests import TrialKind, TrialRecordV2, TrialStatus
from edgestack.recommendation.registry import RecommendationRegistry
from edgestack.research.schemas import (
    CampaignLifecycle,
    CampaignSummaryV1,
    CoverageState,
    DataCoverageV1,
    EvidenceGapV1,
    ShadowStrategyV1,
)
from edgestack.research.store import ResearchStore
from edgestack.validation.advanced_tests import spa_test, stepm_superior
from edgestack.validation.clustered import stationary_bootstrap_indices
from edgestack.validation.metrics import effective_sample_size, max_drawdown, sharpe_ratio


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class HypothesisSpec(_FrozenModel):
    id: str
    family: str
    status: str
    dataset: str
    reason: str


class CombinationSpec(_FrozenModel):
    id: str
    components: tuple[str, ...]
    status: str


class GovernmentContractStudy(_FrozenModel):
    acquisition_start: date
    acquisition_end: date
    evaluation_start: date
    award_type_codes: tuple[str, ...]
    symbols: dict[str, tuple[str, ...]]
    feature_methods: tuple[Literal["seasonal_mad", "trailing_mad", "yoy_log_change"], ...]
    availability_lag_days: tuple[int, ...]
    horizons: tuple[int, ...]
    selection_quantiles: tuple[float, ...]
    neutralizations: tuple[Literal["raw", "size_beta_momentum"], ...]
    feature_definitions: dict[str, str]
    neutralization_definition: str
    neutralization_ridge_alpha: float = Field(gt=0)
    selection_definition: str
    overlapping_cohorts: str
    return_definition: str
    cost_definition: str
    cash_yield_series: Literal["DGS3MO"]
    comparators: tuple[str, ...]
    diversified_baseline_symbols: tuple[str, ...]
    execution_delay_sessions: int = Field(ge=1)
    base_roundtrip_cost_bps: float = Field(ge=0)
    stress_roundtrip_cost_bps: float = Field(ge=0)
    minimum_months_history: int = Field(ge=12)
    minimum_cross_section: int = Field(ge=3)
    validation_folds: int = Field(ge=2)
    embargo_sessions: int = Field(ge=1)
    bootstrap_samples: int = Field(ge=100)
    bootstrap_block_sessions: int = Field(ge=1)
    fdr_alpha: float = Field(gt=0, lt=1)
    seeds: tuple[int, ...]
    placebos: tuple[str, ...]
    required_gates: tuple[str, ...]

    @property
    def trial_count(self) -> int:
        return int(
            np.prod(
                [
                    len(self.feature_methods),
                    len(self.availability_lag_days),
                    len(self.horizons),
                    len(self.selection_quantiles),
                    len(self.neutralizations),
                ]
            )
        )


class StrangeEdgesManifest(_FrozenModel):
    schema_version: Literal[1]
    campaign_id: str
    created_at: datetime
    source_sha256: str
    source_description: str
    trial_cap: int = Field(ge=1, le=1_500)
    final_holdout_start: date
    promotion_eligible: Literal[False]
    promotion_blockers: tuple[str, ...]
    hypotheses: tuple[HypothesisSpec, ...]
    combinations: tuple[CombinationSpec, ...]
    government_contract_study: GovernmentContractStudy

    @property
    def manifest_hash(self) -> str:
        return stable_hash(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def _complete_and_bounded(self) -> StrangeEdgesManifest:
        if len(self.hypotheses) != 9 or len(self.combinations) != 5:
            raise ValueError(
                "strange-edge manifest must retain all 9 hypotheses and 5 combinations"
            )
        if self.government_contract_study.trial_count > self.trial_cap:
            raise ValueError("government-contract grid exceeds the frozen campaign cap")
        if self.government_contract_study.acquisition_end >= self.final_holdout_start:
            raise ValueError("alternative-event acquisition may not enter the final holdout")
        if max(self.government_contract_study.horizons) > (
            self.government_contract_study.embargo_sessions
        ):
            raise ValueError("embargo must cover the longest registered horizon")
        return self


def load_strange_edges_manifest(path: Path) -> StrangeEdgesManifest:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DataError(f"could not load strange-edge manifest: {path}") from exc
    return StrangeEdgesManifest.model_validate(payload)


def registered_government_trials(manifest: StrangeEdgesManifest) -> tuple[TrialRecordV2, ...]:
    study = manifest.government_contract_study
    output: list[TrialRecordV2] = []
    axes = product(
        study.feature_methods,
        study.availability_lag_days,
        study.horizons,
        study.selection_quantiles,
        study.neutralizations,
    )
    for method, lag, horizon, quantile, neutralization in axes:
        parameters: dict[str, Any] = {
            "feature_method": method,
            "availability_lag_days": lag,
            "horizon_sessions": horizon,
            "selection_quantile": quantile,
            "neutralization": neutralization,
            "manifest_hash": manifest.manifest_hash,
        }
        output.append(
            TrialRecordV2(
                trial_id=stable_hash(parameters)[:24],
                experiment_id=manifest.campaign_id,
                kind=TrialKind.STANDALONE,
                family="government_contract_surprise",
                horizon_sessions=horizon,
                parameters=parameters,
                status=TrialStatus.REGISTERED,
            )
        )
    if len(output) != study.trial_count:
        raise DataError("expanded government-contract grid does not match the frozen manifest")
    return tuple(output)


def register_strange_edge_campaigns(
    cfg: EdgeStackConfig,
    manifest: StrangeEdgesManifest,
) -> tuple[TrialRecordV2, ...]:
    """Expose the evaluated campaign and all blocked inputs in Edge Lab."""
    catalog = DataCatalog(cfg)
    store = ResearchStore(catalog)
    registry = RecommendationRegistry(catalog)
    now = datetime.now(UTC)
    trials = registered_government_trials(manifest)
    existing_trials = {item.trial_id: item for item in registry.trials(manifest.campaign_id)}
    for trial in trials:
        prior = existing_trials.get(trial.trial_id)
        if prior is None:
            registry.register_trial(trial)
        elif (
            prior.model_copy(update={"status": TrialStatus.REGISTERED, "failure_reason": None})
            != trial
        ):
            raise DataError(f"strange-edge trial identity collision: {trial.trial_id}")
    primary = store.campaign(manifest.campaign_id)
    if primary is None:
        store.upsert_campaign(
            CampaignSummaryV1(
                campaign_id=manifest.campaign_id,
                manifest_hash=manifest.manifest_hash,
                name="Government-contract surprise",
                family="government_contract_surprise",
                lifecycle=CampaignLifecycle.READY,
                created_at=manifest.created_at,
                updated_at=now,
                trial_count=len(trials),
                data_requirements=("usaspending_monthly_contract_obligations", "daily_prices"),
                next_action="Run the frozen 108-trial exploratory campaign.",
                promotion_eligible=False,
                previously_accessed=False,
            )
        )
    items: tuple[HypothesisSpec | CombinationSpec, ...] = (
        *manifest.hypotheses[1:],
        *manifest.combinations,
    )
    lifecycle_map = {
        "BLOCKED_LICENSE": CampaignLifecycle.BLOCKED_LICENSE,
        "BLOCKED_POINT_IN_TIME": CampaignLifecycle.BLOCKED_POINT_IN_TIME,
        "BLOCKED_COMPLIANCE": CampaignLifecycle.BLOCKED_COMPLIANCE,
        "BLOCKED_COMBINATION_INPUT": CampaignLifecycle.BLOCKED_COMPONENTS,
        "BLOCKED_COMPONENTS": CampaignLifecycle.BLOCKED_COMPONENTS,
        "NEEDS_DATA": CampaignLifecycle.NEEDS_DATA,
    }
    coverage_state_map = {
        "BLOCKED_LICENSE": CoverageState.BLOCKED_LICENSE,
        "BLOCKED_POINT_IN_TIME": CoverageState.BLOCKED_POINT_IN_TIME,
        "BLOCKED_COMPLIANCE": CoverageState.BLOCKED_COMPLIANCE,
        "BLOCKED_COMBINATION_INPUT": CoverageState.BLOCKED_COMPONENTS,
        "BLOCKED_COMPONENTS": CoverageState.BLOCKED_COMPONENTS,
        "NEEDS_DATA": CoverageState.NEEDS_DATA,
    }
    for item in items:
        campaign_id = f"{manifest.campaign_id}:{item.id}"
        reason = (
            item.reason
            if isinstance(item, HypothesisSpec)
            else (
                "One or more pre-registered combination inputs are unavailable; "
                "partial proxies are prohibited."
            )
        )
        desired_lifecycle = lifecycle_map[item.status]
        summary = store.campaign(campaign_id)
        if summary is None:
            summary = CampaignSummaryV1(
                campaign_id=campaign_id,
                manifest_hash=stable_hash(
                    {"parent": manifest.manifest_hash, "item": item.model_dump(mode="json")}
                ),
                name=item.id.replace("_", " ").title(),
                family=item.family if isinstance(item, HypothesisSpec) else "alternative_ensemble",
                lifecycle=desired_lifecycle,
                created_at=manifest.created_at,
                updated_at=now,
                trial_count=0,
                completed_trials=0,
                failure_reasons=(f"{item.status}: {reason}",),
                next_action=reason,
                promotion_eligible=False,
                previously_accessed=False,
                metrics={"blocker_status": item.status},
            )
        elif summary.lifecycle is CampaignLifecycle.NEEDS_DATA and (
            desired_lifecycle is not CampaignLifecycle.NEEDS_DATA
        ):
            summary = summary.model_copy(update={"lifecycle": desired_lifecycle, "updated_at": now})
        store.upsert_campaign(summary)
        symbols = (item.dataset,) if isinstance(item, HypothesisSpec) else tuple(item.components)
        store.upsert_gap(
            EvidenceGapV1(
                requirement_id=stable_hash({"campaign_id": campaign_id, "input": symbols})[:24],
                campaign_id=campaign_id,
                dataset=item.dataset if isinstance(item, HypothesisSpec) else item.id,
                symbols=symbols,
                frequency="alternative_event",
                start=manifest.government_contract_study.acquisition_start,
                end=manifest.government_contract_study.acquisition_end,
                required_observations=1,
                observed_observations=0,
                state=coverage_state_map[item.status],
                next_action=reason,
            )
        )
    return trials


def _logical_frame_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.select_dtypes(include=["datetime", "datetimetz"]):
        normalized[column] = pd.to_datetime(normalized[column]).astype(str)
    records = normalized.replace({np.nan: None}).to_dict(orient="records")
    return stable_hash(records)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    atomic_write_bytes(path, buffer.getvalue())


def acquire_government_contract_data(
    cfg: EdgeStackConfig,
    manifest: StrangeEdgesManifest,
    *,
    refresh: bool = False,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    study = manifest.government_contract_study
    raw_dir = Path(cfg.paths.data_dir) / "raw" / "usaspending"
    provider = USAspendingContractProvider(
        raw_dir=raw_dir,
        timeout=cfg.data.request_timeout_seconds * 2,
        max_retries=cfg.data.max_retries,
        session=session,
    )
    events = provider.fetch_monthly_contract_obligations(
        study.symbols,
        study.acquisition_start,
        study.acquisition_end,
        award_type_codes=study.award_type_codes,
        refresh=refresh,
    )
    expected = len(study.symbols) * len(
        pd.date_range(study.acquisition_start, study.acquisition_end, freq="ME")
    )
    if len(events) != expected:
        raise DataError(f"USAspending normalized coverage is incomplete: {len(events)}/{expected}")
    target = (
        Path(cfg.paths.data_dir)
        / "curated"
        / "alternative_events"
        / "usaspending_contract_obligations.parquet"
    )
    _write_parquet(target, events)
    metadata = {
        "provider": provider.metadata.name,
        "point_in_time": provider.metadata.is_point_in_time,
        "raw_content_hashes": list(provider.last_raw_hashes),
        "normalized_logical_hash": _logical_frame_hash(events),
        "normalized_file_sha256": file_sha256(target),
        "rows": len(events),
        "symbols": len(study.symbols),
        "months": int(events["event_time"].nunique()),
        "missing_rows": int(events["contract_obligations"].isna().sum()),
        "limitations": list(provider.metadata.limitations),
    }
    atomic_write_bytes(target.with_suffix(".metadata.json"), canonical_json(metadata))
    return events, metadata


def acquire_cash_yield(
    cfg: EdgeStackConfig,
    study: GovernmentContractStudy,
    *,
    refresh: bool = False,
    session: requests.Session | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Acquire FRED's public graph CSV for DGS3MO, frozen to the test interval."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {
        "id": study.cash_yield_series,
        "cosd": study.acquisition_start.isoformat(),
        "coed": study.acquisition_end.isoformat(),
    }
    request_hash = stable_hash({"url": url, "params": params})
    raw_dir = Path(cfg.paths.data_dir) / "raw" / "fred_graph"
    request_path = raw_dir / "requests" / f"{request_hash}.csv"
    if request_path.exists() and not refresh:
        raw = request_path.read_bytes()
    else:
        client = session or requests.Session()
        try:
            response = client.get(url, params=params, timeout=cfg.data.request_timeout_seconds * 2)
        except requests.RequestException as exc:
            raise ProviderError("FRED cash-yield request failed (network)") from exc
        if response.status_code >= 400:
            raise ProviderError(f"FRED cash-yield request failed (HTTP {response.status_code})")
        raw = response.content
        digest = hashlib.sha256(raw).hexdigest()
        atomic_write_bytes(raw_dir / "content" / f"{digest}.csv", raw)
        atomic_write_bytes(request_path, raw)
    digest = hashlib.sha256(raw).hexdigest()
    frame = pd.read_csv(io.BytesIO(raw))
    if "observation_date" not in frame or study.cash_yield_series not in frame:
        raise ProviderError("FRED cash-yield CSV has an unexpected schema")
    dates = pd.to_datetime(frame["observation_date"], errors="coerce")
    values = pd.to_numeric(frame[study.cash_yield_series], errors="coerce") / 100.0
    series = pd.Series(values.to_numpy(), index=dates, name="cash_yield").dropna().sort_index()
    series = series.loc[
        (series.index >= pd.Timestamp(study.acquisition_start))
        & (series.index <= pd.Timestamp(study.acquisition_end))
    ]
    if series.empty:
        raise ProviderError("FRED cash-yield series has no observations in the frozen interval")
    return series, {
        "provider": "FRED",
        "series": study.cash_yield_series,
        "raw_content_hash": digest,
        "observations": len(series),
        "start": series.index.min().date().isoformat(),
        "end": series.index.max().date().isoformat(),
        "point_in_time": False,
        "limitations": ["Current FRED graph snapshot; no historical vintage timestamps."],
    }


def _record_strange_data_coverage(
    cfg: EdgeStackConfig,
    events: pd.DataFrame,
    event_metadata: dict[str, Any],
    cash_yield: pd.Series,
    cash_metadata: dict[str, Any],
) -> None:
    store = ResearchStore(DataCatalog(cfg))
    now = datetime.now(UTC)
    raw_hashes = tuple(str(item) for item in event_metadata["raw_content_hashes"])
    for symbol, group in events.groupby("symbol", sort=True):
        missing = int(group["contract_obligations"].isna().sum())
        store.upsert_coverage(
            DataCoverageV1(
                dataset_id=f"government_contracts:usaspending:monthly:{symbol}",
                dataset="government_contracts",
                provider="usaspending",
                feed="spending_over_time_prime_contracts",
                symbol=str(symbol),
                frequency="monthly",
                start=pd.Timestamp(group["event_time"].min()).to_pydatetime().replace(tzinfo=UTC),
                end=pd.Timestamp(group["event_time"].max()).to_pydatetime().replace(tzinfo=UTC),
                observation_count=int(group["contract_obligations"].notna().sum()),
                missing_sessions=missing,
                quality_status="PASS" if missing == 0 else "GAPS",
                point_in_time=False,
                content_hash=str(event_metadata["normalized_logical_hash"]),
                raw_content_hashes=raw_hashes,
                retrieved_at=now,
                quality_results={
                    "returned_months": len(group),
                    "missing_months": missing,
                },
                limitations=tuple(str(item) for item in event_metadata["limitations"]),
            )
        )
    store.upsert_coverage(
        DataCoverageV1(
            dataset_id="macro:fred:daily:DGS3MO:strange-edges",
            dataset="macro",
            provider="fred",
            feed="fred_graph_current_snapshot",
            symbol="DGS3MO",
            frequency="daily",
            start=cash_yield.index.min().to_pydatetime().replace(tzinfo=UTC),
            end=cash_yield.index.max().to_pydatetime().replace(tzinfo=UTC),
            observation_count=len(cash_yield),
            missing_sessions=0,
            quality_status="PASS",
            point_in_time=False,
            content_hash=str(cash_metadata["raw_content_hash"]),
            raw_content_hashes=(str(cash_metadata["raw_content_hash"]),),
            retrieved_at=now,
            quality_results={"returned_observations": len(cash_yield)},
            limitations=tuple(str(item) for item in cash_metadata["limitations"]),
        )
    )


def _signed_log(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    transformed = np.sign(numeric.to_numpy()) * np.log1p(np.abs(numeric.to_numpy()))
    return pd.Series(transformed, index=values.index, dtype=float)


def build_contract_feature(
    events: pd.DataFrame,
    *,
    method: str,
    minimum_months_history: int,
) -> pd.DataFrame:
    """Build a past-only surprise score without filling absent source periods."""
    required = {"symbol", "event_time", "contract_obligations"}
    if missing := required - set(events):
        raise DataError(f"government-contract events missing columns: {sorted(missing)}")
    frames: list[pd.DataFrame] = []
    for _, raw_group in events.groupby("symbol", sort=True):
        group = raw_group.sort_values("event_time").copy()
        transformed = _signed_log(group["contract_obligations"])
        if method == "trailing_mad":
            prior = transformed.shift(1)
            center = prior.rolling(24, min_periods=24).median()
            scale = prior.rolling(24, min_periods=24).apply(
                lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True
            )
            score = (transformed - center) / (1.4826 * scale.replace(0.0, np.nan))
        elif method == "seasonal_mad":
            score = pd.Series(np.nan, index=group.index, dtype=float)
            month_numbers = pd.to_datetime(group["event_time"]).dt.month
            for month in range(1, 13):
                positions = group.index[month_numbers == month]
                seasonal = transformed.loc[positions]
                center = seasonal.shift(1).expanding(min_periods=3).median()
                scale = (
                    seasonal.shift(1)
                    .expanding(min_periods=3)
                    .apply(lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True)
                )
                score.loc[positions] = (seasonal - center) / (1.4826 * scale.replace(0.0, np.nan))
        elif method == "yoy_log_change":
            score = transformed - transformed.shift(12)
        else:
            raise DataError(f"unknown government-contract feature method: {method}")
        history = np.arange(len(group))
        group["score"] = score.where(history >= minimum_months_history)
        group["feature_method"] = method
        frames.append(group)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["event_time", "symbol"])
        .reset_index(drop=True)
    )


def _adjusted_price_inputs(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prepared = panel.copy()
    factor = (prepared["adj_close"] / prepared["close"]).replace([np.inf, -np.inf], np.nan)
    if bool(factor.isna().any()) or bool((factor <= 0).any()):
        raise DataError("adjusted-open factors are missing or invalid")
    prepared["adjusted_open"] = prepared["open"] * factor
    opens = prepared.pivot(index="date", columns="symbol", values="adjusted_open").sort_index()
    closes = prepared.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    volumes = prepared.pivot(index="date", columns="symbol", values="volume").sort_index()
    return opens, closes, volumes


def _control_panels(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    *,
    symbols: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    returns = closes.pct_change(fill_method=None)
    spy = returns["SPY"]
    rolling_variance = spy.rolling(252, min_periods=126).var()
    beta = pd.DataFrame(index=closes.index, columns=symbols, dtype=float)
    for symbol in symbols:
        beta[symbol] = returns[symbol].rolling(252, min_periods=126).cov(spy) / rolling_variance
    dollar_volume = closes.loc[:, list(symbols)] * volumes.loc[:, list(symbols)]
    log_adv = dollar_volume.rolling(60, min_periods=40).median().apply(np.log)
    momentum = closes.loc[:, list(symbols)].pct_change(126, fill_method=None)
    return {"log_adv60": log_adv, "beta252": beta, "momentum126": momentum}


def _ridge_residual(
    frame: pd.DataFrame,
    *,
    alpha: float,
    minimum_cross_section: int,
) -> pd.Series:
    controls = ["log_adv60", "beta252", "momentum126"]
    valid = frame[["score", *controls]].replace([np.inf, -np.inf], np.nan).dropna()
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    if len(valid) < max(minimum_cross_section, len(controls) + 2):
        return output
    x_raw = valid[controls].to_numpy(dtype=float)
    means = x_raw.mean(axis=0)
    scales = x_raw.std(axis=0, ddof=1)
    scales = np.where(scales > 0, scales, 1.0)
    x = np.column_stack([np.ones(len(valid)), (x_raw - means) / scales])
    penalty = np.diag([0.0, alpha, alpha, alpha])
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ valid["score"].to_numpy(dtype=float))
    output.loc[valid.index] = valid["score"].to_numpy(dtype=float) - x @ beta
    return output


def align_feature_availability(
    feature: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    controls: dict[str, pd.DataFrame],
    *,
    lag_days: int,
    neutralization: str,
    ridge_alpha: float,
    minimum_cross_section: int,
) -> pd.DataFrame:
    """Attach observable dates, next-open entries, and point-in-time controls."""
    output = feature.copy()
    output["available_at"] = pd.to_datetime(output["event_time"]) + pd.to_timedelta(
        lag_days, unit="D"
    )
    all_positions = sessions.searchsorted(pd.DatetimeIndex(output["available_at"]), side="left")
    valid = all_positions < len(sessions) - 1
    output = output.loc[valid].copy()
    positions = np.asarray(all_positions[valid], dtype=np.int64)
    output["signal_date"] = sessions[positions].to_numpy()
    output["entry_date"] = sessions[positions + 1].to_numpy()
    for name, panel in controls.items():
        output[name] = [
            panel.at[signal_date, symbol]
            if signal_date in panel.index and symbol in panel.columns
            else np.nan
            for signal_date, symbol in zip(output["signal_date"], output["symbol"], strict=True)
        ]
    if neutralization == "raw":
        output["selection_score"] = output["score"]
    elif neutralization == "size_beta_momentum":
        residuals = pd.Series(np.nan, index=output.index, dtype=float)
        for _, group in output.groupby("signal_date", sort=True):
            residuals.loc[group.index] = _ridge_residual(
                group,
                alpha=ridge_alpha,
                minimum_cross_section=minimum_cross_section,
            )
        output["selection_score"] = residuals
    else:
        raise DataError(f"unknown neutralization: {neutralization}")
    return output.sort_values(["signal_date", "symbol"]).reset_index(drop=True)


def _portfolio_path(
    aligned_feature: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cash_daily: pd.Series,
    *,
    horizon: int,
    quantile: float,
    roundtrip_cost_bps: float,
    minimum_cross_section: int,
    placebo: str | None = None,
    seed: int = 42,
) -> tuple[pd.Series, pd.DataFrame]:
    dates = asset_returns.index
    symbols = tuple(asset_returns.columns)
    accumulated = np.zeros((len(dates), len(symbols)), dtype=float)
    active_cohorts = np.zeros(len(dates), dtype=float)
    rng = np.random.default_rng(seed)
    symbol_location = {symbol: index for index, symbol in enumerate(symbols)}
    for _, raw_group in aligned_feature.groupby("signal_date", sort=True):
        group = raw_group.dropna(subset=["selection_score", "entry_date"]).copy()
        group = group.loc[group["symbol"].isin(symbol_location)]
        if len(group) < minimum_cross_section:
            continue
        entry = pd.Timestamp(group["entry_date"].iloc[0])
        start = int(dates.searchsorted(entry, side="left"))
        if placebo == "timestamp_shift_one_month":
            start += 21
        if start + horizon >= len(dates):
            continue
        count = max(1, int(np.ceil(quantile * len(group))))
        if placebo == "company_shuffle":
            group["selection_score"] = rng.permutation(group["selection_score"].to_numpy())
        if placebo == "random_portfolio":
            selected = group.iloc[rng.choice(len(group), size=count, replace=False)]
        else:
            selected = group.nlargest(count, "selection_score", keep="first")
        weight = 1.0 / len(selected)
        for symbol in selected["symbol"].astype(str):
            accumulated[start : start + horizon, symbol_location[symbol]] += weight
        active_cohorts[start : start + horizon] += 1.0
    weights = np.divide(
        accumulated,
        active_cohorts[:, None],
        out=np.zeros_like(accumulated),
        where=active_cohorts[:, None] > 0,
    )
    weight_frame = pd.DataFrame(weights, index=dates, columns=symbols)
    gross = (weight_frame * asset_returns).sum(axis=1)
    risky_weight = weight_frame.sum(axis=1).clip(0.0, 1.0)
    gross += (1.0 - risky_weight) * cash_daily.reindex(dates).fillna(0.0)
    change = weight_frame.diff().fillna(weight_frame)
    one_way = roundtrip_cost_bps / 20_000.0
    costs = change.abs().sum(axis=1) * one_way
    return (gross - costs).rename("strategy"), weight_frame


def _risk_matched_spy(
    strategy: pd.Series,
    spy: pd.Series,
    cash_daily: pd.Series,
) -> pd.Series:
    target = float(strategy.std(ddof=1))
    benchmark = float(spy.std(ddof=1))
    scale = min(5.0, target / benchmark) if benchmark > 0 else 0.0
    return scale * spy + max(0.0, 1.0 - scale) * cash_daily


def _candidate_metrics(
    strategy: pd.Series,
    stress: pd.Series,
    spy: pd.Series,
    risk_spy: pd.Series,
    weights: pd.DataFrame,
    *,
    folds: int,
) -> dict[str, Any]:
    excess = strategy - risk_spy
    fold_values = [float(excess.loc[index].mean()) for index in np.array_split(excess.index, folds)]
    return {
        "sessions": len(strategy),
        "invested_sessions": int((weights.sum(axis=1) > 0).sum()),
        "effective_sample_size": float(effective_sample_size(excess.to_numpy(dtype=float))),
        "annual_log_growth": annualized_log_growth(strategy),
        "spy_annual_log_growth": annualized_log_growth(spy),
        "risk_matched_spy_annual_log_growth": annualized_log_growth(risk_spy),
        "annual_excess_mean": float(excess.mean() * 252.0),
        "annual_volatility": float(strategy.std(ddof=1) * np.sqrt(252.0)),
        "annual_sharpe": float(sharpe_ratio(strategy.to_numpy(dtype=float))),
        "max_drawdown": float(max_drawdown(strategy.to_numpy(dtype=float))),
        "stress_annual_log_growth": annualized_log_growth(stress),
        "positive_folds": int(sum(item > 0 for item in fold_values)),
        "fold_means": fold_values,
        "fold_consistency": float(sum(item > 0 for item in fold_values) / folds),
    }


def _selected_price_hashes(cfg: EdgeStackConfig, symbols: tuple[str, ...]) -> dict[str, str]:
    directory = Path(cfg.paths.data_dir) / "curated" / "prices"
    output: dict[str, str] = {}
    for symbol in sorted(set(symbols)):
        path = directory / f"{symbol}.parquet"
        if not path.exists():
            raise DataError(f"required frozen price series is missing: {symbol}")
        output[symbol] = file_sha256(path)
    return output


def _placebo_diagnostics(
    candidate: TrialRecordV2,
    aligned_feature: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cash_daily: pd.Series,
    *,
    study: GovernmentContractStudy,
) -> dict[str, Any]:
    values: dict[str, list[float]] = {name: [] for name in study.placebos}
    for placebo, seed in product(study.placebos, study.seeds):
        path, _ = _portfolio_path(
            aligned_feature,
            asset_returns,
            cash_daily,
            horizon=candidate.horizon_sessions,
            quantile=float(candidate.parameters["selection_quantile"]),
            roundtrip_cost_bps=study.base_roundtrip_cost_bps,
            minimum_cross_section=study.minimum_cross_section,
            placebo=placebo,
            seed=seed,
        )
        values[placebo].append(annualized_log_growth(path))
    flattened = [value for collection in values.values() for value in collection]
    return {
        "annual_log_growth": values,
        "maximum_placebo_log_growth": max(flattened) if flattened else float("inf"),
    }


def _persist_campaign_result(
    cfg: EdgeStackConfig,
    manifest: StrangeEdgesManifest,
    trials: tuple[TrialRecordV2, ...],
    report: dict[str, Any],
    artifact_hash: str,
) -> None:
    catalog = DataCatalog(cfg)
    store = ResearchStore(catalog)
    registry = RecommendationRegistry(catalog)
    for trial in trials:
        registry.update_trial(trial.model_copy(update={"status": TrialStatus.SUCCEEDED}))
    qualified = tuple(str(item) for item in report["historically_qualified_trial_ids"])
    lifecycle = CampaignLifecycle.PAPER_SHADOW if qualified else CampaignLifecycle.REJECTED
    current = store.campaign(manifest.campaign_id)
    if current is None:
        raise DataError("strange-edge campaign was not registered")
    if current.lifecycle in {
        CampaignLifecycle.READY,
        CampaignLifecycle.RUNNING,
        lifecycle,
    }:
        if current.lifecycle is CampaignLifecycle.READY:
            current = current.model_copy(
                update={"lifecycle": CampaignLifecycle.RUNNING, "updated_at": datetime.now(UTC)}
            )
            store.upsert_campaign(current)
        reasons = tuple(report["campaign_failure_reasons"])
        store.upsert_campaign(
            current.model_copy(
                update={
                    "lifecycle": lifecycle,
                    "updated_at": datetime.now(UTC),
                    "completed_trials": len(trials),
                    "failure_reasons": reasons,
                    "next_action": (
                        "Begin a new, current-date prospective shadow; historical "
                        "results remain promotion-ineligible."
                        if qualified
                        else (
                            "Acquire a new point-in-time input or register a new economic "
                            "hypothesis; do not retune this holdout."
                        )
                    ),
                    "artifact_hash": artifact_hash,
                    "promotion_eligible": False,
                    "metrics": {
                        "historically_qualified": len(qualified),
                        "spa_p_consistent": report["family_tests"]["spa_p_consistent"],
                        "completed_trials": len(trials),
                    },
                }
            )
        )
    if qualified:
        best = qualified[0]
        existing = {item.strategy_id for item in store.shadows()}
        strategy_id = f"gov-contract:{best}"
        if strategy_id not in existing:
            store.upsert_shadow(
                ShadowStrategyV1(
                    strategy_id=strategy_id,
                    campaign_id=manifest.campaign_id,
                    artifact_hash=artifact_hash,
                    status=CampaignLifecycle.PAPER_SHADOW,
                    started_at=datetime.now(UTC),
                    sessions=0,
                    trades=0,
                    effective_resolved_outcomes=0,
                    equity=cfg.paper.initial_cash,
                    warnings=tuple(manifest.promotion_blockers),
                )
            )


def run_strange_edges_campaign(
    cfg: EdgeStackConfig,
    *,
    manifest_path: Path = Path("configs/strange_edges.yaml"),
    refresh: bool = False,
) -> dict[str, Any]:
    """Acquire, evaluate, freeze, and expose the complete strange-edge campaign."""
    manifest = load_strange_edges_manifest(manifest_path)
    trials = register_strange_edge_campaigns(cfg, manifest)
    study = manifest.government_contract_study
    events, event_metadata = acquire_government_contract_data(cfg, manifest, refresh=refresh)
    cash_yield, cash_metadata = acquire_cash_yield(cfg, study, refresh=refresh)
    _record_strange_data_coverage(
        cfg,
        events,
        event_metadata,
        cash_yield,
        cash_metadata,
    )

    research_symbols = tuple(sorted(study.symbols))
    all_symbols = tuple(
        dict.fromkeys((*research_symbols, *study.diversified_baseline_symbols, "SPY"))
    )
    price_hashes = _selected_price_hashes(cfg, all_symbols)
    panel = DataCatalog(cfg).load_panel(
        all_symbols,
        start=study.acquisition_start,
        end=study.acquisition_end,
    )
    if panel["date"].max() >= pd.Timestamp(manifest.final_holdout_start):
        raise ValidationError("final-holdout prices entered the strange-edge study")
    opens, closes, volumes = _adjusted_price_inputs(panel)
    complete = opens.loc[:, list(all_symbols)].dropna()
    open_returns = complete.shift(-1).div(complete).sub(1.0).dropna()
    open_returns = open_returns.loc[
        (open_returns.index >= pd.Timestamp(study.evaluation_start))
        & (open_returns.index <= pd.Timestamp(study.acquisition_end))
    ]
    if len(open_returns) < 1_000:
        raise ValidationError("government-contract study has insufficient common price sessions")
    cash_daily = cash_yield.reindex(open_returns.index).ffill().fillna(0.0).clip(lower=0.0) / 252.0
    asset_returns = open_returns.loc[:, list(research_symbols)]
    spy = open_returns["SPY"].rename("buy_now_spy")
    diversified = open_returns.loc[:, list(study.diversified_baseline_symbols)].mean(axis=1)
    controls = _control_panels(closes, volumes, symbols=research_symbols)

    code_hash = stable_hash(
        {
            "study": file_sha256(Path(__file__)),
            "provider": file_sha256(
                Path(__file__).parents[1] / "data" / "providers" / "usaspending.py"
            ),
        }
    )
    input_hashes = {
        "events": event_metadata["normalized_logical_hash"],
        "cash": cash_metadata["raw_content_hash"],
        "prices": stable_hash(price_hashes),
    }
    run_id = stable_hash(
        {
            "manifest_hash": manifest.manifest_hash,
            "input_hashes": input_hashes,
            "code_hash": code_hash,
        }
    )[:24]
    run_dir = Path(cfg.paths.artifacts_dir) / "research" / "strange_edges" / run_id
    report_path = run_dir / "report.json"
    if report_path.exists() and not refresh:
        cached_report = json.loads(report_path.read_text(encoding="utf-8"))
        artifact_hash = hashlib.sha256(canonical_json(cached_report)).hexdigest()
        _persist_campaign_result(cfg, manifest, trials, cached_report, artifact_hash)
        return cached_report

    feature_cache: dict[str, pd.DataFrame] = {
        method: build_contract_feature(
            events,
            method=method,
            minimum_months_history=study.minimum_months_history,
        )
        for method in study.feature_methods
    }
    aligned_cache: dict[tuple[str, int, str], pd.DataFrame] = {}
    return_paths: dict[str, pd.Series] = {}
    risk_benchmarks: dict[str, pd.Series] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for trial in trials:
        method = str(trial.parameters["feature_method"])
        lag = int(trial.parameters["availability_lag_days"])
        neutralization = str(trial.parameters["neutralization"])
        cache_key = (method, lag, neutralization)
        if cache_key not in aligned_cache:
            aligned_cache[cache_key] = align_feature_availability(
                feature_cache[method],
                pd.DatetimeIndex(open_returns.index),
                controls,
                lag_days=lag,
                neutralization=neutralization,
                ridge_alpha=study.neutralization_ridge_alpha,
                minimum_cross_section=study.minimum_cross_section,
            )
        aligned = aligned_cache[cache_key]
        strategy, weights = _portfolio_path(
            aligned,
            asset_returns,
            cash_daily,
            horizon=trial.horizon_sessions,
            quantile=float(trial.parameters["selection_quantile"]),
            roundtrip_cost_bps=study.base_roundtrip_cost_bps,
            minimum_cross_section=study.minimum_cross_section,
        )
        stress, _ = _portfolio_path(
            aligned,
            asset_returns,
            cash_daily,
            horizon=trial.horizon_sessions,
            quantile=float(trial.parameters["selection_quantile"]),
            roundtrip_cost_bps=study.stress_roundtrip_cost_bps,
            minimum_cross_section=study.minimum_cross_section,
        )
        risk_spy = _risk_matched_spy(strategy, spy, cash_daily)
        return_paths[trial.trial_id] = strategy
        risk_benchmarks[trial.trial_id] = risk_spy
        metrics[trial.trial_id] = _candidate_metrics(
            strategy,
            stress,
            spy,
            risk_spy,
            weights,
            folds=study.validation_folds,
        )

    bootstrap_index = stationary_bootstrap_indices(
        len(open_returns),
        n_boot=study.bootstrap_samples,
        mean_block_length=study.bootstrap_block_sessions,
        seed=study.seeds[0],
    )
    pvalues: list[float] = []
    for trial in trials:
        strategy = return_paths[trial.trial_id]
        excess = (strategy - risk_benchmarks[trial.trial_id]).to_numpy(dtype=float)
        observed = float(excess.mean())
        centered = excess - observed
        null_means = centered[bootstrap_index].mean(axis=1)
        pvalue = (float((null_means >= observed).sum()) + 1.0) / (study.bootstrap_samples + 1.0)
        sampled_excess = excess[bootstrap_index].mean(axis=1) * 252.0
        sampled_log_growth = (
            np.log1p(return_paths[trial.trial_id].to_numpy(dtype=float))[bootstrap_index].mean(
                axis=1
            )
            * 252.0
        )
        metrics[trial.trial_id].update(
            {
                "raw_pvalue": pvalue,
                "annual_excess_lower_95": float(np.quantile(sampled_excess, 0.05)),
                "annual_log_growth_lower_95": float(np.quantile(sampled_log_growth, 0.05)),
            }
        )
        pvalues.append(pvalue)
    qvalues = benjamini_hochberg(np.asarray(pvalues, dtype=float))
    for trial, qvalue in zip(trials, qvalues, strict=True):
        metrics[trial.trial_id]["fdr_qvalue"] = float(qvalue)

    candidate_frame = pd.DataFrame(return_paths)
    try:
        spa = spa_test(
            spy,
            candidate_frame,
            reps=study.bootstrap_samples,
            block_size=study.bootstrap_block_sessions,
            seed=study.seeds[0],
        )
        superior = set(
            stepm_superior(
                spy,
                candidate_frame,
                reps=study.bootstrap_samples,
                block_size=study.bootstrap_block_sessions,
                size=study.fdr_alpha,
                seed=study.seeds[0],
            )
        )
        family_error: str | None = None
    except (ImportError, ValidationError, ValueError, np.linalg.LinAlgError) as exc:
        spa = {"p_consistent": 1.0, "best_model": None, "n_models": len(trials)}
        superior = set()
        family_error = f"selection-aware family test failed safely: {type(exc).__name__}"

    preliminary = [
        trial
        for trial in trials
        if metrics[trial.trial_id]["fdr_qvalue"] <= study.fdr_alpha
        and metrics[trial.trial_id]["annual_log_growth_lower_95"] > 0
        and metrics[trial.trial_id]["fold_consistency"] >= 0.60
        and metrics[trial.trial_id]["stress_annual_log_growth"]
        > metrics[trial.trial_id]["risk_matched_spy_annual_log_growth"]
        and float(spa["p_consistent"]) <= study.fdr_alpha
        and trial.trial_id in superior
    ]
    qualified: list[str] = []
    for trial in preliminary:
        strategy = return_paths[trial.trial_id]
        target_volatility = float(strategy.std(ddof=1) * np.sqrt(252.0))
        annual_cash = float(cash_yield.mean())
        financed = financed_risk_matched_spy(
            spy,
            target_volatility=target_volatility,
            financing_rate=annual_cash,
            cash_yield=annual_cash,
        )
        growth_bounds = log_growth_superiority(
            strategy,
            {
                "buy_now_spy": spy,
                "risk_matched_spy": risk_benchmarks[trial.trial_id],
                "diversified_baseline": diversified,
                "financed_spy_same_risk": financed,
            },
            n_boot=study.bootstrap_samples,
            seed=study.seeds[0],
        )
        cache_key = (
            str(trial.parameters["feature_method"]),
            int(trial.parameters["availability_lag_days"]),
            str(trial.parameters["neutralization"]),
        )
        placebo = _placebo_diagnostics(
            trial,
            aligned_cache[cache_key],
            asset_returns,
            cash_daily,
            study=study,
        )
        metrics[trial.trial_id]["growth_lower_bounds"] = growth_bounds
        metrics[trial.trial_id]["placebos"] = placebo
        beats_growth = all(value > 0 for value in growth_bounds.values())
        beats_placebos = (
            metrics[trial.trial_id]["annual_log_growth"] > placebo["maximum_placebo_log_growth"]
        )
        metrics[trial.trial_id]["beats_all_growth_comparators"] = beats_growth
        metrics[trial.trial_id]["beats_placebos"] = beats_placebos
        if beats_growth and beats_placebos:
            qualified.append(trial.trial_id)

    ranked = sorted(
        trials,
        key=lambda item: (
            metrics[item.trial_id]["fdr_qvalue"],
            -metrics[item.trial_id]["annual_excess_mean"],
            item.trial_id,
        ),
    )
    trial_rows = [
        {
            "trial_id": trial.trial_id,
            "parameters": trial.parameters,
            "metrics": metrics[trial.trial_id],
            "historically_qualified": trial.trial_id in qualified,
        }
        for trial in ranked
    ]
    trials_payload_hash = hashlib.sha256(canonical_json(trial_rows)).hexdigest()
    failure_reasons = list(manifest.promotion_blockers)
    if not qualified:
        failure_reasons.insert(
            0,
            "No pre-registered candidate passed every cost, growth, multiplicity, "
            "fold, stress, and placebo gate.",
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": manifest.campaign_id,
        "run_id": run_id,
        "manifest_hash": manifest.manifest_hash,
        "code_hash": code_hash,
        "input_hashes": input_hashes,
        "final_holdout_start": manifest.final_holdout_start.isoformat(),
        "final_holdout_accessed": False,
        "promotion_eligible": False,
        "registered_trials": len(trials),
        "completed_trials": len(trials),
        "trial_results_hash": trials_payload_hash,
        "family_tests": {
            "spa_p_consistent": float(spa["p_consistent"]),
            "spa_best_model": spa.get("best_model"),
            "stepm_superior_trial_ids": sorted(superior),
            "family_test_error": family_error,
        },
        "historically_qualified_trial_ids": qualified,
        "shadow_strategies_started": len(qualified),
        "promoted_sleeves": 0,
        "canonical_alpha_expectation": 0.0,
        "campaign_failure_reasons": failure_reasons,
        "data": {"government_contracts": event_metadata, "cash_yield": cash_metadata},
        "hypotheses": [item.model_dump(mode="json") for item in manifest.hypotheses],
        "combinations": [item.model_dump(mode="json") for item in manifest.combinations],
        "top_trials": trial_rows[:10],
    }
    atomic_write_bytes(run_dir / "trials.json", canonical_json(trial_rows))
    report_bytes = canonical_json(report)
    atomic_write_bytes(report_path, report_bytes)
    artifact_hash = hashlib.sha256(report_bytes).hexdigest()
    _persist_campaign_result(cfg, manifest, trials, report, artifact_hash)
    return report
