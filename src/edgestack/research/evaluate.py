"""Deterministic, calendar-time evaluation for registered research campaigns."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from edgestack.data.catalog import DataCatalog, atomic_write_bytes
from edgestack.discovery.multiple_testing import benjamini_hochberg
from edgestack.execution.costs import CostModel
from edgestack.execution.fills import Bar
from edgestack.execution.orders import Order, OrderType
from edgestack.paper.broker import SimulatedBroker
from edgestack.paper.shadow import ShadowBook
from edgestack.recommendation.growth import (
    REQUIRED_GROWTH_COMPARATORS,
    annualized_log_growth,
    financed_risk_matched_spy,
    log_growth_superiority,
)
from edgestack.recommendation.hashing import stable_hash
from edgestack.recommendation.manifests import ProspectiveEvidenceV2, TrialRecordV2, TrialStatus
from edgestack.recommendation.promotion import (
    PromotionInputs,
    evaluate_promotion,
    risk_match_benchmark,
)
from edgestack.recommendation.registry import RecommendationRegistry
from edgestack.recommendation.schemas import (
    AssetKind,
    EvidenceGrade,
    SleeveContributionV2,
    WeightV2,
)
from edgestack.research.opening_fade import (
    aggregate_session_returns,
    load_campaign_config,
    run_opening_fade_campaign,
)
from edgestack.research.schemas import CampaignLifecycle, CampaignSummaryV1, ShadowStrategyV1
from edgestack.research.store import ResearchStore
from edgestack.research.templates import CampaignTemplate
from edgestack.research.universe import ALL_RESEARCH_ETFS, INVERSE_ETFS
from edgestack.validation.advanced_tests import model_confidence_set, spa_test, stepm_superior
from edgestack.validation.bootstrap import suggested_block_length
from edgestack.validation.clustered import session_effective_sample_size, stationary_mean_test
from edgestack.validation.metrics import deflated_sharpe_ratio, max_drawdown, sharpe_ratio
from edgestack.validation.overfitting import pbo_cscv
from edgestack.validation.selection import persist_trial_returns


def _family_overfitting_diagnostics(
    store: ResearchStore,
    benchmark: pd.Series,
    frame: pd.DataFrame,
    *,
    batch_id: str,
) -> dict:
    """Report-only overfitting diagnostics for a complete candidate family.

    MCS + PBO + Politis-White block-length check, plus persistence of the
    T x k return matrix so the diagnostics can be replayed on stored
    batches. Diagnostics never abort a promotion review: failures degrade
    to UNAVAILABLE envelopes.
    """
    from edgestack.exceptions import DataError, ValidationError

    cfg = store.catalog.cfg.validation
    seed = store.catalog.cfg.project.random_seed
    try:
        mcs: dict = model_confidence_set(
            frame,
            size=cfg.mcs_size,
            reps=max(100, min(cfg.bootstrap_samples, 1_000)),
            block_size=cfg.block_length_sessions,
            seed=seed,
        )
    except ValidationError as exc:
        mcs = {"status": "UNAVAILABLE", "reason": str(exc)}
    try:
        pbo: dict = pbo_cscv(frame, n_partitions=cfg.pbo_partitions)
    except ValidationError as exc:
        pbo = {"status": "UNAVAILABLE", "reason": str(exc)}
    diagnostic = suggested_block_length(
        frame.mean(axis=1).dropna().to_numpy(), fallback=cfg.block_length_sessions
    )
    path: str | None = None
    persist_error: str | None = None
    try:
        artifact = persist_trial_returns(
            benchmark,
            frame,
            persist_dir=store.catalog.artifacts_dir / "trial_returns",
            batch_id=batch_id,
        )
        store.record_trial_returns_artifact(artifact)
        path = artifact["path"]
    except (DataError, OSError) as exc:
        persist_error = str(exc)
    return {
        "mcs": mcs,
        "pbo": pbo,
        "block_length_diagnostic": diagnostic,
        "trial_returns_path": path,
        "persist_error": persist_error,
    }


def _overfitting_promotion_fields(diagnostics: dict, candidate_id: str, cfg: Any) -> dict:
    """Translate family diagnostics into PromotionInputs keyword arguments."""
    mcs = diagnostics["mcs"]
    member = candidate_id in mcs["included"] if "included" in mcs else None
    pvalue = mcs.get("pvalues", {}).get(candidate_id)
    return {
        "mcs_member": member,
        "mcs_pvalue": pvalue,
        "pbo_cscv": diagnostics["pbo"].get("pbo"),
        "pbo_max": cfg.pbo_max,
        "overfitting_gates_binding": cfg.overfitting_gates_binding,
    }


def _raw_daily_signal(candidate: dict[str, Any], prices: pd.DataFrame) -> pd.Series:
    symbol = str(candidate.get("symbol", "SPY"))
    if symbol not in prices:
        return pd.Series(0.0, index=prices.index)
    price = prices[symbol]
    rule = str(candidate["rule"])
    if rule == "trend_dip":
        trend = price > price.rolling(int(candidate["lookback"])).mean()
        signal = trend & (price.pct_change(5) <= float(candidate["dip"]))
    elif rule == "breakout":
        signal = price > price.rolling(int(candidate["lookback"])).max().shift(1)
    elif rule == "mean_reversion":
        signal = price.pct_change(int(candidate["lookback"])) <= float(candidate["threshold"])
    elif rule == "relative_strength":
        benchmark = prices.get("SPY", price)
        lookback = int(candidate["lookback"])
        relative = price.pct_change(lookback) - benchmark.pct_change(lookback)
        signal = relative >= float(candidate["threshold"])
    elif rule == "volatility_regime":
        returns = price.pct_change()
        lookback = int(candidate["lookback"])
        volatility = returns.rolling(lookback).std()
        threshold = volatility.expanding(252).quantile(float(candidate["quantile"]))
        signal = volatility <= threshold
    elif rule == "calendar":
        weekdays = pd.DatetimeIndex(prices.index).weekday
        signal = pd.Series(weekdays == int(candidate["weekday"]), index=prices.index)
    elif rule == "inverse_trend":
        signal = price > price.rolling(int(candidate["lookback"])).mean()
    else:
        signal = pd.Series(False, index=prices.index)
    return signal.astype(float)


def _position(candidate: dict[str, Any], prices: pd.DataFrame) -> pd.Series:
    signal = _raw_daily_signal(candidate, prices)
    hold = int(candidate.get("hold", 1))
    # Signal at close t may affect the position no earlier than t+1.
    return signal.rolling(hold, min_periods=1).max().shift(1).fillna(0.0)


def daily_position_for_next_open(candidate: dict[str, Any], prices: pd.DataFrame) -> float:
    """Frozen close-t signal translated to the desired position at t+1 open."""
    if prices.empty:
        return 0.0
    hold = int(candidate.get("hold", 1))
    desired = _raw_daily_signal(candidate, prices).rolling(hold, min_periods=1).max()
    return float(desired.iloc[-1])


def _calendar_time_metrics(
    candidate: dict[str, Any],
    prices: pd.DataFrame,
    cash_yield: pd.Series,
    *,
    n_trials: int,
    execution_prices: pd.DataFrame | None = None,
) -> tuple[dict[str, float | int | str], pd.Series, float]:
    symbol = str(candidate.get("symbol", "SPY"))
    strategy, position = _daily_strategy_returns(
        candidate,
        prices,
        cash_yield,
        execution_prices=execution_prices,
    )
    execution = execution_prices if execution_prices is not None else prices
    benchmark_returns = _next_open_returns(execution["SPY"])
    aligned = pd.concat(
        [strategy.rename("strategy"), benchmark_returns.rename("spy")], axis=1
    ).dropna()
    strategy_values = aligned["strategy"].to_numpy()
    benchmark_values = aligned["spy"].to_numpy()
    strategy_vol = float(np.std(strategy_values, ddof=1))
    benchmark_vol = float(np.std(benchmark_values, ddof=1))
    risk_scale = strategy_vol / benchmark_vol if benchmark_vol > 0 else 0.0
    risk_matched = aligned["spy"] * risk_scale
    baseline_symbols = ("SPY", "TLT", "SHY", "GLD")
    baseline = (
        _next_open_frame(execution.loc[:, list(baseline_symbols)])
        .reindex(aligned.index)
        .mean(axis=1)
    )
    financed_spy = financed_risk_matched_spy(
        aligned["spy"],
        target_volatility=strategy_vol * np.sqrt(252),
        financing_rate=float(cash_yield.reindex(aligned.index).ffill().fillna(0.0).mean()),
        cash_yield=float(cash_yield.reindex(aligned.index).ffill().fillna(0.0).mean()),
    )
    growth_bounds = log_growth_superiority(
        aligned["strategy"],
        {
            "buy_now_spy": aligned["spy"],
            "risk_matched_spy": risk_matched,
            "diversified_baseline": baseline,
            "financed_spy_same_risk": financed_spy,
        },
        n_boot=500,
        seed=int(stable_hash(candidate)[:8], 16),
    )
    active = aligned["strategy"] - risk_matched
    if len(active) >= 30 and float(active.std(ddof=1)) > 0:
        pvalue = float(
            scipy_stats.ttest_1samp(active.to_numpy(), 0.0, alternative="greater").pvalue
        )
    else:
        pvalue = 1.0
    log_growth = annualized_log_growth(aligned["strategy"])
    spy_log_growth = annualized_log_growth(aligned["spy"])
    risk_spy_log_growth = annualized_log_growth(risk_matched)
    metrics: dict[str, float | int | str] = {
        "candidate_id": stable_hash(candidate)[:16],
        "symbol": symbol,
        "sessions": len(aligned),
        "invested_sessions": int((position > 0).sum()),
        "calendar_log_growth": log_growth,
        "spy_log_growth": spy_log_growth,
        "risk_matched_spy_log_growth": risk_spy_log_growth,
        "log_growth_improvement": log_growth - risk_spy_log_growth,
        **{f"log_growth_lower_vs_{name}": value for name, value in growth_bounds.items()},
        "sharpe": sharpe_ratio(strategy_values) if len(strategy_values) >= 2 else 0.0,
        "spy_sharpe": sharpe_ratio(benchmark_values) if len(benchmark_values) >= 2 else 0.0,
        "max_drawdown": max_drawdown(strategy_values) if len(strategy_values) >= 2 else 0.0,
        "deflated_sharpe_probability": (
            deflated_sharpe_ratio(strategy_values, n_trials=n_trials)
            if len(strategy_values) >= 3 and strategy_vol > 0
            else 0.0
        ),
        "raw_pvalue": pvalue,
    }
    return metrics, strategy, pvalue


def _daily_strategy_returns(
    candidate: dict[str, Any],
    prices: pd.DataFrame,
    cash_yield: pd.Series,
    *,
    cost_per_exposure_change: float = 0.001,
    extra_execution_delay: int = 0,
    execution_prices: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series]:
    symbol = str(candidate.get("symbol", "SPY"))
    execution = execution_prices if execution_prices is not None else prices
    asset_returns = _next_open_returns(execution[symbol])
    position = _position(candidate, prices).reindex(asset_returns.index).fillna(0.0)
    if extra_execution_delay:
        position = position.shift(extra_execution_delay).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    daily_cash = cash_yield.reindex(prices.index).ffill().fillna(0.0) / 252.0
    strategy = (
        position * asset_returns
        + (1.0 - position) * daily_cash
        - turnover * cost_per_exposure_change
    )
    return strategy, position


def _next_open_returns(prices: pd.Series) -> pd.Series:
    """Return from the indexed session's open to the following session's open."""
    return prices.shift(-1).div(prices).sub(1.0).iloc[:-1]


def _next_open_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Vectorized next-open returns for aligned benchmark matrices."""
    return prices.shift(-1).div(prices).sub(1.0).iloc[:-1]


def _adjusted_open_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    factor = (panel["adj_close"] / panel["close"]).replace([np.inf, -np.inf], np.nan)
    if bool(factor.isna().any()) or bool((factor <= 0).any()):
        raise ValueError("daily campaign has invalid adjusted-open factors")
    prepared = panel.assign(adjusted_open=panel["open"] * factor)
    return prepared.pivot(index="date", columns="symbol", values="adjusted_open").sort_index()


def _write_factory_artifact(
    catalog: DataCatalog,
    template: CampaignTemplate,
    payload: dict[str, Any],
) -> tuple[str, Path]:
    stable = {
        "manifest_hash": template.manifest_hash,
        "data_version": catalog.data_manifest_hash(),
        "payload": payload,
    }
    artifact_hash = stable_hash(stable)
    root = catalog.artifacts_dir / "research" / "factory"
    destination = root / "runs" / artifact_hash
    encoded = json.dumps(stable, indent=2, sort_keys=True, default=str).encode()
    if destination.exists():
        existing = (destination / "result.json").read_bytes()
        if existing != encoded:
            raise ValueError("content-addressed research artifact mismatch")
    else:
        atomic_write_bytes(destination / "result.json", encoded)
    pointer = {
        "campaign_id": template.template_id,
        "artifact_hash": artifact_hash,
        "relative_path": f"runs/{artifact_hash}",
    }
    atomic_write_bytes(
        root / f"current-{template.template_id}.json",
        json.dumps(pointer, indent=2, sort_keys=True).encode(),
    )
    return artifact_hash, destination


def _update_trials(
    store: ResearchStore,
    template: CampaignTemplate,
    results: list[dict[str, Any]],
) -> None:
    now = datetime.now(UTC)
    with store.catalog.connect() as con:
        rows = con.execute(
            """SELECT trial_id, payload FROM trial_ledger_v2
               WHERE experiment_id = ? ORDER BY trial_id""",
            [template.template_id],
        ).fetchall()
        by_candidate = {str(item["candidate_id"]): item for item in results}
        for trial_id, payload in rows:
            trial = TrialRecordV2.model_validate_json(payload)
            candidate_id = stable_hash(trial.parameters)[:16]
            result = by_candidate.get(candidate_id)
            status = (
                TrialStatus.SUCCEEDED
                if result is not None and float(result.get("q_value", 1.0)) <= 0.05
                else TrialStatus.REJECTED
            )
            updated = trial.model_copy(update={"status": status})
            con.execute(
                """UPDATE trial_ledger_v2
                   SET created_at = ?, status = ?, payload = ? WHERE trial_id = ?""",
                [now, status.value, updated.model_dump_json(), trial_id],
            )


def evaluate_daily_campaign(
    store: ResearchStore,
    template: CampaignTemplate,
) -> CampaignSummaryV1:
    candidate_symbols = tuple(
        dict.fromkeys(str(item.get("symbol", "SPY")) for item in template.candidate_family)
    )
    symbols = tuple(dict.fromkeys((*candidate_symbols, "TLT", "SHY", "GLD")))
    panel = store.catalog.load_panel(symbols=symbols)
    prices = panel.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    execution_prices = _adjusted_open_matrix(panel)
    common = prices.index.intersection(execution_prices.index)
    prices = prices.loc[common].dropna(subset=list(symbols))
    execution_prices = execution_prices.reindex(prices.index).dropna(subset=list(symbols))
    prices = prices.reindex(execution_prices.index)
    cash_path = store.catalog.data_dir / "curated" / "macro" / "DGS3MO.parquet"
    cash_frame = pd.read_parquet(cash_path)
    cash_frame["observation_date"] = pd.to_datetime(cash_frame["observation_date"])
    cash_frame["realtime_start"] = pd.to_datetime(cash_frame["realtime_start"])
    cash_yield = (
        cash_frame.sort_values("realtime_start")
        .drop_duplicates("observation_date", keep="first")
        .set_index("observation_date")["value"]
        .astype(float)
        .div(100.0)
        .sort_index()
    )
    results: list[dict[str, Any]] = []
    pvalues = []
    global_trials = max(len(template.candidate_family), store.family_trial_count(template.family))
    for candidate in template.candidate_family:
        if str(candidate.get("symbol", "SPY")) not in prices:
            continue
        metrics, _returns, pvalue = _calendar_time_metrics(
            candidate,
            prices,
            cash_yield,
            n_trials=global_trials,
            execution_prices=execution_prices,
        )
        results.append({**candidate, **metrics})
        pvalues.append(pvalue)
    if pvalues:
        qvalues = benjamini_hochberg(np.asarray(pvalues, dtype=float))
        for result, qvalue in zip(results, qvalues, strict=True):
            result["q_value"] = float(qvalue)
    winners = [
        item
        for item in results
        if int(item["sessions"]) >= 756
        and float(item["q_value"]) <= 0.05
        and float(item["log_growth_improvement"]) > 0
        and all(
            float(item[f"log_growth_lower_vs_{name}"]) > 0 for name in REQUIRED_GROWTH_COMPARATORS
        )
        and float(item["deflated_sharpe_probability"]) >= 0.95
    ]
    payload = {
        "campaign_id": template.template_id,
        "manifest_hash": template.manifest_hash,
        "trial_count": len(template.candidate_family),
        "global_family_trial_count": global_trials,
        "calendar_time_includes_cash_days": True,
        "transaction_cost_bps_per_exposure_change": 10.0,
        "results": results,
        "winner_ids": [item["candidate_id"] for item in winners],
        "previously_accessed": template.previously_accessed,
        "promotion_decisions": [],
    }
    artifact_hash, _ = _write_factory_artifact(store.catalog, template, payload)
    _update_trials(store, template, results)
    current = store.campaign(template.template_id)
    if current is None:
        raise ValueError(f"campaign {template.template_id} was not registered")
    lifecycle = CampaignLifecycle.PAPER_SHADOW if winners else CampaignLifecycle.REJECTED
    next_action = (
        "Open independent prospective paper books; promotion still requires every frozen gate."
        if winners
        else "Keep the rejection and advance to the next bounded campaign."
    )
    updated = current.model_copy(
        update={
            "lifecycle": lifecycle,
            "updated_at": datetime.now(UTC),
            "completed_trials": len(results),
            "artifact_hash": artifact_hash,
            "failure_reasons": (
                ()
                if winners
                else (
                    "No candidate passed complete-family multiplicity, deflated Sharpe, "
                    "and lower-bound log-growth gates against every comparator.",
                )
            ),
            "next_action": next_action,
            "metrics": {
                "evaluated": len(results),
                "shadow_eligible": len(winners),
                "best_log_growth_improvement": max(
                    (float(item["log_growth_improvement"]) for item in results), default=0.0
                ),
            },
        }
    )
    store.upsert_campaign(updated)
    for winner in winners:
        strategy_id = f"{template.template_id}:{winner['candidate_id']}"
        store.upsert_shadow(
            ShadowStrategyV1(
                strategy_id=strategy_id,
                campaign_id=template.template_id,
                artifact_hash=artifact_hash,
                status=CampaignLifecycle.PAPER_SHADOW,
                started_at=datetime.now(UTC),
                equity=100_000.0,
                benchmark_returns={
                    "SPY": 0.0,
                    "RISK_MATCHED_SPY": 0.0,
                    "DIVERSIFIED_BASELINE": 0.0,
                },
                warnings=("Historical qualification only; awaiting prospective paper outcomes.",),
            )
        )
    return updated


def evaluate_opening_campaign(
    store: ResearchStore,
    template: CampaignTemplate,
    *,
    repo_root: Path,
) -> CampaignSummaryV1:
    config_path = repo_root / "configs" / "opening_fade.yaml"
    config = load_campaign_config(config_path)
    result, _trades, destination = run_opening_fade_campaign(
        config,
        repo_root=repo_root,
        symbols=tuple(item.symbol for item in store.coverage() if item.frequency == "1m"),
        interval_minutes=1,
        publish=True,
    )
    current = store.campaign(template.template_id)
    if current is None:
        raise ValueError(f"campaign {template.template_id} was not registered")
    winners = result["validation"]["paper_observation_candidates"]
    lifecycle = CampaignLifecycle.PAPER_SHADOW if winners else CampaignLifecycle.REJECTED
    artifact_hash = destination.name if destination is not None else stable_hash(result)
    updated = current.model_copy(
        update={
            "lifecycle": lifecycle,
            "updated_at": datetime.now(UTC),
            "completed_trials": len(result["trial_ledger"]),
            "artifact_hash": artifact_hash,
            "next_action": (
                "Start prospective paper observations under the unchanged manifest."
                if winners
                else "Continue to the next bounded campaign; do not force an opening trade."
            ),
            "metrics": {
                "evaluated": len(result["trial_ledger"]),
                "shadow_eligible": len(winners),
                "eligible_sessions": int(result["data_audit"].get("eligible_sessions", 0)),
            },
        }
    )
    store.upsert_campaign(updated)
    for winner in winners:
        candidate_id = str(winner.get("candidate_id")) if isinstance(winner, dict) else str(winner)
        store.upsert_shadow(
            ShadowStrategyV1(
                strategy_id=f"{template.template_id}:{candidate_id}",
                campaign_id=template.template_id,
                artifact_hash=artifact_hash,
                status=CampaignLifecycle.PAPER_SHADOW,
                started_at=datetime.now(UTC),
                equity=100_000.0,
                benchmark_returns={
                    "SPY": 0.0,
                    "RISK_MATCHED_SPY": 0.0,
                    "DIVERSIFIED_BASELINE": 0.0,
                    "FINANCED_SPY_SAME_RISK": 0.0,
                },
                warnings=("Prospective intraday paper observation; canonical weight is zero.",),
            )
        )
    return updated


def _initial_macro_series(catalog: DataCatalog, series_id: str) -> pd.Series:
    path = catalog.data_dir / "curated" / "macro" / f"{series_id}.parquet"
    frame = pd.read_parquet(path)
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    frame["realtime_start"] = pd.to_datetime(frame["realtime_start"])
    return (
        frame.sort_values("realtime_start")
        .drop_duplicates("observation_date", keep="first")
        .set_index("observation_date")["value"]
        .astype(float)
        .sort_index()
    )


def _event_strategy_metrics(
    candidate: dict[str, Any],
    strategy: pd.Series,
    prices: pd.DataFrame,
    *,
    invested_sessions: int,
    n_trials: int,
    cash_yield: pd.Series,
    execution_prices: pd.DataFrame,
) -> tuple[dict[str, Any], float]:
    spy = _next_open_returns(execution_prices["SPY"])
    aligned = pd.concat([strategy.rename("strategy"), spy.rename("spy")], axis=1).dropna()
    strategy_vol = float(aligned["strategy"].std(ddof=1))
    spy_vol = float(aligned["spy"].std(ddof=1))
    scale = strategy_vol / spy_vol if spy_vol > 0 else 0.0
    risk_matched = aligned["spy"] * scale
    baseline = (
        _next_open_frame(execution_prices.loc[:, ["SPY", "TLT", "SHY", "GLD"]])
        .reindex(aligned.index)
        .mean(axis=1)
    )
    annual_cash = float(cash_yield.reindex(aligned.index).ffill().fillna(0.0).mean())
    financed = financed_risk_matched_spy(
        aligned["spy"],
        target_volatility=strategy_vol * np.sqrt(252),
        financing_rate=annual_cash,
        cash_yield=annual_cash,
    )
    growth_bounds = log_growth_superiority(
        aligned["strategy"],
        {
            "buy_now_spy": aligned["spy"],
            "risk_matched_spy": risk_matched,
            "diversified_baseline": baseline,
            "financed_spy_same_risk": financed,
        },
        n_boot=500,
        seed=int(stable_hash(candidate)[:8], 16),
    )
    active = aligned["strategy"] - risk_matched
    pvalue = (
        float(scipy_stats.ttest_1samp(active, 0.0, alternative="greater").pvalue)
        if len(active) >= 30 and float(active.std(ddof=1)) > 0
        else 1.0
    )
    values = aligned["strategy"].to_numpy()
    metrics: dict[str, Any] = {
        "candidate_id": stable_hash(candidate)[:16],
        "sessions": len(aligned),
        "invested_sessions": invested_sessions,
        "calendar_log_growth": annualized_log_growth(aligned["strategy"]),
        "spy_log_growth": annualized_log_growth(aligned["spy"]),
        "risk_matched_spy_log_growth": annualized_log_growth(risk_matched),
        "sharpe": sharpe_ratio(values) if len(values) >= 2 else 0.0,
        "max_drawdown": max_drawdown(values) if len(values) >= 2 else 0.0,
        "deflated_sharpe_probability": (
            deflated_sharpe_ratio(values, n_trials=n_trials)
            if len(values) >= 3 and strategy_vol > 0
            else 0.0
        ),
        "raw_pvalue": pvalue,
        **{f"log_growth_lower_vs_{name}": value for name, value in growth_bounds.items()},
    }
    return metrics, pvalue


def _event_context(
    catalog: DataCatalog,
    prices: pd.DataFrame,
    stock_symbols: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    cash_yield = _initial_macro_series(catalog, "DGS3MO").div(100.0)
    vix_path = catalog.data_dir / "curated" / "macro" / "VIXCLS.parquet"
    dff_path = catalog.data_dir / "curated" / "macro" / "DFF.parquet"
    vix = (
        _initial_macro_series(catalog, "VIXCLS").reindex(prices.index).ffill()
        if vix_path.exists()
        else pd.Series(np.nan, index=prices.index)
    )
    dff = (
        _initial_macro_series(catalog, "DFF").reindex(prices.index).ffill()
        if dff_path.exists()
        else pd.Series(np.nan, index=prices.index)
    )
    event_signals = pd.DataFrame(False, index=prices.index, columns=stock_symbols)
    short_signals = pd.DataFrame(False, index=prices.index, columns=stock_symbols)
    for symbol in stock_symbols:
        event_path = catalog.data_dir / "curated" / "events" / f"{symbol}.parquet"
        if event_path.exists():
            events = pd.read_parquet(event_path)
            dates = (
                pd.to_datetime(events["accepted_at"], utc=True).dt.tz_convert(None).dt.normalize()
            )
            event_signals.loc[event_signals.index.isin(dates), symbol] = True

        short_path = catalog.data_dir / "curated" / "short_sale_volume" / f"{symbol}.parquet"
        if not short_path.exists():
            continue
        short = pd.read_parquet(short_path)
        short["date"] = pd.to_datetime(short["date"])
        ratio = (
            short.set_index("date")["short_sale_volume"]
            / short.set_index("date")["total_reported_volume"].replace(0, np.nan)
        ).reindex(prices.index)
        threshold = (
            ratio.rolling(60, min_periods=40).mean() + ratio.rolling(60, min_periods=40).std()
        )
        short_signals[symbol] = (ratio > threshold) & (returns[symbol].rolling(3).sum() < 0)
    return event_signals, short_signals, vix, dff, cash_yield


def _event_candidate_returns(
    candidate: dict[str, Any],
    prices: pd.DataFrame,
    stock_symbols: tuple[str, ...],
    event_signals: pd.DataFrame,
    short_signals: pd.DataFrame,
    vix: pd.Series,
    dff: pd.Series,
    cash_yield: pd.Series,
    *,
    cost_per_exposure_change: float = 0.001,
    execution_delay: int = 1,
    extra_execution_delay: int = 0,
    execution_prices: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    execution = execution_prices if execution_prices is not None else prices
    returns = _next_open_frame(execution)
    horizon = int(candidate["horizon"])
    rule = str(candidate["rule"])
    shift = execution_delay + extra_execution_delay
    if rule in {"earnings_drift", "short_sale_volume_reversal"}:
        signals = event_signals if rule == "earnings_drift" else short_signals
        positions = (
            signals.astype(float)
            .rolling(horizon, min_periods=1)
            .max()
            .shift(shift)
            .reindex(returns.index)
        )
        count = positions.sum(axis=1)
        weights = positions.div(count.replace(0, np.nan), axis=0).fillna(0.0)
        exposure = count.gt(0).astype(float)
        turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
        strategy = (weights * returns[list(stock_symbols)]).sum(axis=1)
    else:
        macro_signal = (vix.pct_change(fill_method=None) > 0.10) | (dff.diff().abs() >= 0.25)
        exposure = (
            macro_signal.astype(float)
            .rolling(horizon, min_periods=1)
            .max()
            .shift(shift)
            .fillna(0.0)
        )
        exposure = exposure.reindex(returns.index).fillna(0.0)
        turnover = exposure.diff().abs().fillna(exposure.abs())
        weights = pd.DataFrame({"SPY": exposure}, index=returns.index)
        strategy = exposure * returns["SPY"]
    daily_cash = cash_yield.reindex(returns.index).ffill().fillna(0.0) / 252.0
    strategy = strategy + (1.0 - exposure) * daily_cash - turnover * cost_per_exposure_change
    return strategy, weights


def event_weights_for_next_open(
    catalog: DataCatalog,
    candidate: dict[str, Any],
    prices: pd.DataFrame,
    stock_symbols: tuple[str, ...],
) -> dict[str, float]:
    """Translate point-in-time event inputs through close t into t+1 target weights."""
    event_signals, short_signals, vix, dff, cash_yield = _event_context(
        catalog, prices, stock_symbols
    )
    del cash_yield
    rule = str(candidate["rule"])
    horizon = int(candidate["horizon"])
    if rule in {"earnings_drift", "short_sale_volume_reversal"}:
        signals = event_signals if rule == "earnings_drift" else short_signals
        desired = signals.astype(float).rolling(horizon, min_periods=1).max().iloc[-1]
        count = float(desired.sum())
        weights = desired.div(count) if count > 0 else desired
    else:
        macro_signal = (vix.pct_change(fill_method=None) > 0.10) | (dff.diff().abs() >= 0.25)
        exposure = float(macro_signal.astype(float).rolling(horizon, min_periods=1).max().iloc[-1])
        weights = pd.Series({"SPY": exposure})
    return {str(symbol): float(weight) for symbol, weight in weights.items() if float(weight) > 0}


def evaluate_event_campaign(
    store: ResearchStore,
    template: CampaignTemplate,
) -> CampaignSummaryV1:
    """Evaluate event families with next-session signals and calendar-time cash carry."""
    stock_symbols = next(
        requirement.symbols
        for requirement in template.requirements
        if requirement.dataset == "earnings"
    )
    symbols = ("SPY", "QQQ", "IWM", "DIA", *stock_symbols, "TLT", "SHY", "GLD")
    panel = store.catalog.load_panel(symbols=symbols)
    prices = panel.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    execution_prices = _adjusted_open_matrix(panel)
    common = prices.index.intersection(execution_prices.index)
    prices = prices.reindex(common).dropna(subset=["SPY"])
    execution_prices = execution_prices.reindex(prices.index).dropna(subset=["SPY"])
    prices = prices.reindex(execution_prices.index)
    event_signals, short_signals, vix, dff, cash_yield = _event_context(
        store.catalog, prices, stock_symbols
    )

    results: list[dict[str, Any]] = []
    pvalues: list[float] = []
    global_trials = max(len(template.candidate_family), store.family_trial_count(template.family))
    for candidate in template.candidate_family:
        strategy, weights = _event_candidate_returns(
            candidate,
            prices,
            stock_symbols,
            event_signals,
            short_signals,
            vix,
            dff,
            cash_yield,
            execution_prices=execution_prices,
        )
        exposure = weights.abs().sum(axis=1).clip(upper=1.0)
        metrics, pvalue = _event_strategy_metrics(
            candidate,
            strategy,
            prices,
            invested_sessions=int(exposure.sum()),
            n_trials=global_trials,
            cash_yield=cash_yield,
            execution_prices=execution_prices,
        )
        results.append({**candidate, **metrics})
        pvalues.append(pvalue)
    qvalues = benjamini_hochberg(np.asarray(pvalues, dtype=float))
    for result, qvalue in zip(results, qvalues, strict=True):
        result["q_value"] = float(qvalue)
    winners = [
        item
        for item in results
        if int(item["sessions"]) >= 756
        and float(item["q_value"]) <= 0.05
        and float(item["deflated_sharpe_probability"]) >= 0.95
        and all(
            float(item[f"log_growth_lower_vs_{name}"]) > 0 for name in REQUIRED_GROWTH_COMPARATORS
        )
    ]
    payload = {
        "campaign_id": template.template_id,
        "manifest_hash": template.manifest_hash,
        "trial_count": len(template.candidate_family),
        "global_family_trial_count": global_trials,
        "calendar_time_includes_cash_days": True,
        "signals_shifted_before_execution": True,
        "finra_label": "off-exchange short-sale volume; not short interest",
        "results": results,
        "winner_ids": [item["candidate_id"] for item in winners],
        "promotion_decisions": [],
    }
    artifact_hash, _ = _write_factory_artifact(store.catalog, template, payload)
    _update_trials(store, template, results)
    current = store.campaign(template.template_id)
    if current is None:
        raise ValueError(f"campaign {template.template_id} was not registered")
    updated = current.model_copy(
        update={
            "lifecycle": (
                CampaignLifecycle.PAPER_SHADOW if winners else CampaignLifecycle.REJECTED
            ),
            "updated_at": datetime.now(UTC),
            "completed_trials": len(results),
            "artifact_hash": artifact_hash,
            "next_action": (
                "Start independent prospective shadow books."
                if winners
                else "Retain the rejection and advance to the next registered campaign."
            ),
            "metrics": {"evaluated": len(results), "shadow_eligible": len(winners)},
        }
    )
    store.upsert_campaign(updated)
    for winner in winners:
        store.upsert_shadow(
            ShadowStrategyV1(
                strategy_id=f"{template.template_id}:{winner['candidate_id']}",
                campaign_id=template.template_id,
                artifact_hash=artifact_hash,
                status=CampaignLifecycle.PAPER_SHADOW,
                started_at=datetime.now(UTC),
                equity=100_000.0,
                benchmark_returns={
                    "SPY": 0.0,
                    "RISK_MATCHED_SPY": 0.0,
                    "DIVERSIFIED_BASELINE": 0.0,
                    "FINANCED_SPY_SAME_RISK": 0.0,
                },
            )
        )
    return updated


def update_daily_shadow(store: ResearchStore, strategy_id: str) -> ShadowStrategyV1:
    """Advance one daily shadow from frozen parameters without touching canonical paper."""
    shadow = next((item for item in store.shadows() if item.strategy_id == strategy_id), None)
    if shadow is None:
        raise ValueError(f"unknown shadow strategy {strategy_id}")
    result_path = (
        store.catalog.artifacts_dir
        / "research"
        / "factory"
        / "runs"
        / shadow.artifact_hash
        / "result.json"
    )
    wrapper = json.loads(result_path.read_text(encoding="utf-8"))
    candidate_id = strategy_id.rsplit(":", 1)[-1]
    candidate = next(
        item for item in wrapper["payload"]["results"] if item["candidate_id"] == candidate_id
    )
    symbols = tuple(
        dict.fromkeys((str(candidate.get("symbol", "SPY")), "SPY", "TLT", "SHY", "GLD"))
    )
    with store.catalog.guard.unlock(
        reason="prospective paper-shadow update",
        experiment_id=shadow.campaign_id,
    ) as unlock_key:
        panel = store.catalog.load_panel(symbols=symbols, unlock_key=unlock_key)
    prices = panel.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    execution_prices = _adjusted_open_matrix(panel)
    common = prices.index.intersection(execution_prices.index)
    prices = prices.loc[common].dropna(subset=list(symbols))
    execution_prices = execution_prices.reindex(prices.index).dropna(subset=list(symbols))
    prices = prices.reindex(execution_prices.index)
    cash_yield = _initial_macro_series(store.catalog, "DGS3MO").div(100.0)
    _metrics, strategy, _pvalue = _calendar_time_metrics(
        candidate,
        prices,
        cash_yield,
        n_trials=int(wrapper["payload"]["trial_count"]),
        execution_prices=execution_prices,
    )
    start = pd.Timestamp(shadow.started_at.date())
    prospective_dates = strategy.index[strategy.index > start]
    if len(prospective_dates) == 0:
        return shadow
    spy_full = _next_open_returns(execution_prices["SPY"])
    baseline_full = _next_open_frame(execution_prices[["SPY", "TLT", "SHY", "GLD"]]).mean(axis=1)
    strategy_vol = float(strategy.std(ddof=1) * np.sqrt(252))
    risk_scale = strategy_vol / float(spy_full.std(ddof=1) * np.sqrt(252))
    cash_rates = cash_yield.reindex(prices.index).ffill().fillna(0.0)
    average_rate = float(cash_rates.reindex(prospective_dates).mean())
    financed_full = financed_risk_matched_spy(
        spy_full,
        target_volatility=strategy_vol,
        financing_rate=average_rate,
        cash_yield=average_rate,
    )
    symbol = str(candidate.get("symbol", "SPY"))
    bars = panel.loc[panel["symbol"] == symbol].copy()
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars.set_index("date").sort_index()
    position = _position(candidate, prices)
    book = ShadowBook(
        strategy_id,
        SimulatedBroker(CostModel.from_config(store.catalog.cfg)),
        initial_cash=100_000.0,
    )
    resolved_outcomes = 0
    for session in prospective_dates:
        if session not in bars.index:
            continue
        row = bars.loc[session]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        opening_equity = book.cash + book.positions.get(symbol, 0.0) * float(row["open"])
        prior_quantity = book.positions.get(symbol, 0.0)
        desired_quantity = opening_equity * float(position.loc[session]) / float(row["open"])
        order_quantity = desired_quantity - prior_quantity
        bar = Bar(
            session=pd.Timestamp(session),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        if abs(order_quantity) > 1e-9:
            book.execute(
                Order(
                    symbol,
                    order_quantity,
                    OrderType.MARKET_ON_OPEN,
                    pd.Timestamp(session),
                    tag="shadow_rebalance",
                ),
                bar,
                at_open_phase=True,
            )
        if abs(prior_quantity) > 1e-9 and abs(book.positions.get(symbol, 0.0)) <= 1e-9:
            resolved_outcomes += 1
        rate = float(cash_rates.loc[session])
        spy_return = float(spy_full.loc[session])
        financed_return = float(financed_full.loc[session])
        book.mark_session(
            session=pd.Timestamp(session).date(),
            close_prices={symbol: float(row["close"])},
            benchmark_returns={
                "SPY": spy_return,
                "RISK_MATCHED_SPY": spy_return * risk_scale,
                "DIVERSIFIED_BASELINE": float(baseline_full.loc[session]),
                "FINANCED_SPY_SAME_RISK": financed_return,
            },
            cash_yield=rate,
            financing_rate=rate + 0.02,
        )
    if not book.history:
        return shadow
    final = book.history[-1]
    notional = sum(abs(fill.quantity * fill.price) for fill in book.fills)
    execution_bps = (
        10_000.0 * sum(fill.cost for fill in book.fills) / notional if notional > 0 else None
    )
    return shadow.model_copy(
        update={
            "resolved_through": final.session,
            "sessions": len(book.history),
            "trades": len(book.fills),
            "effective_resolved_outcomes": float(resolved_outcomes),
            "equity": final.equity,
            "net_return": final.equity / 100_000.0 - 1.0,
            "benchmark_returns": {
                name: equity / 100_000.0 - 1.0 for name, equity in final.benchmark_equity.items()
            },
            "realized_slippage_bps": execution_bps,
            "warnings": (
                "Independent simulated book; partial fills, participation, spread, "
                "slippage, cash yield, and funding are applied.",
            ),
        }
    )


def update_event_shadow(
    store: ResearchStore,
    template: CampaignTemplate,
    strategy_id: str,
) -> ShadowStrategyV1:
    """Replay a frozen event rule in its own multi-asset simulated paper book."""
    shadow = next((item for item in store.shadows() if item.strategy_id == strategy_id), None)
    if shadow is None:
        raise ValueError(f"unknown shadow strategy {strategy_id}")
    result_path = (
        store.catalog.artifacts_dir
        / "research"
        / "factory"
        / "runs"
        / shadow.artifact_hash
        / "result.json"
    )
    wrapper = json.loads(result_path.read_text(encoding="utf-8"))
    candidate_id = strategy_id.rsplit(":", 1)[-1]
    candidate = next(
        item for item in wrapper["payload"]["results"] if item["candidate_id"] == candidate_id
    )
    stock_symbols = next(
        requirement.symbols
        for requirement in template.requirements
        if requirement.dataset == "earnings"
    )
    symbols = tuple(
        dict.fromkeys(("SPY", "QQQ", "IWM", "DIA", *stock_symbols, "TLT", "SHY", "GLD"))
    )
    with store.catalog.guard.unlock(
        reason="prospective event paper-shadow update",
        experiment_id=shadow.campaign_id,
    ) as unlock_key:
        panel = store.catalog.load_panel(symbols=symbols, unlock_key=unlock_key)
    panel["date"] = pd.to_datetime(panel["date"])
    prices = panel.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    prices = prices.dropna(subset=["SPY"])
    event_signals, short_signals, vix, dff, cash_yield = _event_context(
        store.catalog, prices, stock_symbols
    )
    strategy, target_weights = _event_candidate_returns(
        candidate,
        prices,
        stock_symbols,
        event_signals,
        short_signals,
        vix,
        dff,
        cash_yield,
    )
    prospective_dates = strategy.index[strategy.index > pd.Timestamp(shadow.started_at.date())]
    if len(prospective_dates) == 0:
        return shadow
    spy_full = prices["SPY"].pct_change(fill_method=None).fillna(0.0)
    baseline_full = (
        prices[["SPY", "TLT", "SHY", "GLD"]].pct_change(fill_method=None).fillna(0.0).mean(axis=1)
    )
    strategy_volatility = float(strategy.std(ddof=1) * np.sqrt(252))
    spy_volatility = float(spy_full.std(ddof=1) * np.sqrt(252))
    risk_scale = strategy_volatility / spy_volatility if spy_volatility > 0 else 0.0
    cash_rates = cash_yield.reindex(prices.index).ffill().fillna(0.0)
    average_rate = float(cash_rates.reindex(prospective_dates).mean())
    financed_full = financed_risk_matched_spy(
        spy_full,
        target_volatility=strategy_volatility,
        financing_rate=average_rate + 0.02,
        cash_yield=average_rate,
    )
    day_rows = {
        pd.Timestamp(cast(Any, session)): rows.set_index("symbol")
        for session, rows in panel.groupby("date", observed=True)
    }
    book = ShadowBook(
        strategy_id,
        SimulatedBroker(CostModel.from_config(store.catalog.cfg)),
        initial_cash=100_000.0,
    )
    resolved_outcomes = 0
    for session in prospective_dates:
        rows = day_rows.get(pd.Timestamp(session))
        if rows is None:
            continue
        opening_values = [
            quantity * float(cast(Any, rows.loc[symbol, "open"]))
            for symbol, quantity in book.positions.items()
            if symbol in rows.index
        ]
        opening_equity = book.cash + sum(opening_values)
        desired = target_weights.reindex(index=[session], fill_value=0.0).iloc[0]
        rebalance_symbols = sorted(set(book.positions) | set(desired.index))
        for symbol in rebalance_symbols:
            if symbol not in rows.index:
                continue
            row = rows.loc[symbol]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            prior_quantity = book.positions.get(symbol, 0.0)
            target_weight = float(desired.get(symbol, 0.0))
            desired_quantity = opening_equity * target_weight / float(row["open"])
            order_quantity = desired_quantity - prior_quantity
            bar = Bar(
                session=pd.Timestamp(session),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            if abs(order_quantity) > 1e-9:
                book.execute(
                    Order(
                        symbol,
                        order_quantity,
                        OrderType.MARKET_ON_OPEN,
                        pd.Timestamp(session),
                        tag="event_shadow_rebalance",
                    ),
                    bar,
                    at_open_phase=True,
                )
            if abs(prior_quantity) > 1e-9 and abs(book.positions.get(symbol, 0.0)) <= 1e-9:
                resolved_outcomes += 1
        rate = float(cash_rates.loc[session])
        book.mark_session(
            session=pd.Timestamp(session).date(),
            close_prices={symbol: float(rows.loc[symbol, "close"]) for symbol in rows.index},
            benchmark_returns={
                "SPY": float(spy_full.loc[session]),
                "RISK_MATCHED_SPY": float(spy_full.loc[session]) * risk_scale,
                "DIVERSIFIED_BASELINE": float(baseline_full.loc[session]),
                "FINANCED_SPY_SAME_RISK": float(financed_full.loc[session]),
            },
            cash_yield=rate,
            financing_rate=rate + 0.02,
        )
    if not book.history:
        return shadow
    final = book.history[-1]
    notional = sum(abs(fill.quantity * fill.price) for fill in book.fills)
    execution_bps = (
        10_000.0 * sum(fill.cost for fill in book.fills) / notional if notional > 0 else None
    )
    return shadow.model_copy(
        update={
            "resolved_through": final.session,
            "sessions": len(book.history),
            "trades": len(book.fills),
            "effective_resolved_outcomes": float(resolved_outcomes),
            "equity": final.equity,
            "net_return": final.equity / 100_000.0 - 1.0,
            "benchmark_returns": {
                name: equity / 100_000.0 - 1.0 for name, equity in final.benchmark_equity.items()
            },
            "realized_slippage_bps": execution_bps,
            "warnings": (
                "Independent multi-asset event book; partial fills, participation, spread, "
                "slippage, cash yield, and funding are applied.",
            ),
        }
    )


def update_opening_shadow(
    store: ResearchStore,
    template: CampaignTemplate,
    strategy_id: str,
    *,
    repo_root: Path,
) -> ShadowStrategyV1:
    """Refresh a capital-ineligible opening shadow on newly arrived sessions."""
    shadow = next((item for item in store.shadows() if item.strategy_id == strategy_id), None)
    if shadow is None:
        raise ValueError(f"unknown shadow strategy {strategy_id}")
    config = load_campaign_config(repo_root / "configs" / "opening_fade.yaml")
    symbols = tuple(
        sorted(
            {
                item.symbol
                for item in store.coverage()
                if item.dataset == "intraday" and item.frequency in {"1m", "5m"}
            }
        )
    )
    _, conservative, _ = run_opening_fade_campaign(
        config,
        repo_root=repo_root,
        symbols=symbols,
        interval_minutes=1,
        publish=False,
    )
    candidate_id = strategy_id.rsplit(":", 1)[-1]
    candidate_trades = conservative.loc[
        conservative.get("candidate_id", pd.Series(dtype=str)) == candidate_id
    ].copy()
    if not candidate_trades.empty:
        candidate_trades["session"] = pd.to_datetime(candidate_trades["session"])
        candidate_trades = candidate_trades.loc[
            candidate_trades["session"] > pd.Timestamp(shadow.started_at.date())
        ]
    with store.catalog.guard.unlock(
        reason="capital-ineligible opening paper-shadow update",
        experiment_id=template.template_id,
    ) as unlock_key:
        panel = store.catalog.load_panel(
            symbols=("SPY", "TLT", "SHY", "GLD"), unlock_key=unlock_key
        )
    prices = panel.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    prospective_dates = prices.index[prices.index > pd.Timestamp(shadow.started_at.date())]
    if len(prospective_dates) == 0:
        return shadow
    session_returns = aggregate_session_returns(candidate_trades).reindex(
        prospective_dates, fill_value=0.0
    )
    cash_path = store.catalog.data_dir / "curated" / "macro" / "DGS3MO.parquet"
    if cash_path.exists():
        cash_yield = _initial_macro_series(store.catalog, "DGS3MO").div(100.0)
        cash_daily = cash_yield.reindex(prospective_dates).ffill().fillna(0.0) / 252.0
    else:
        cash_daily = pd.Series(0.0, index=prospective_dates)
    traded_sessions = set(pd.to_datetime(candidate_trades.get("session", [])))
    strategy = session_returns.copy()
    for session in prospective_dates:
        if session not in traded_sessions:
            strategy.loc[session] = float(cash_daily.loc[session])
    spy = prices["SPY"].pct_change(fill_method=None).fillna(0.0).reindex(prospective_dates)
    baseline = (
        prices[["SPY", "TLT", "SHY", "GLD"]]
        .pct_change(fill_method=None)
        .fillna(0.0)
        .mean(axis=1)
        .reindex(prospective_dates)
    )
    strategy_volatility = float(strategy.std(ddof=1) * np.sqrt(252))
    spy_volatility = float(spy.std(ddof=1) * np.sqrt(252))
    risk_scale = strategy_volatility / spy_volatility if spy_volatility > 0 else 0.0
    average_rate = float(cash_daily.mean() * 252.0)
    financed = financed_risk_matched_spy(
        spy,
        target_volatility=strategy_volatility,
        financing_rate=average_rate + 0.02,
        cash_yield=average_rate,
    )

    def compounded(values: pd.Series) -> float:
        return float(np.prod(1.0 + values.to_numpy(dtype=float)) - 1.0)

    net_equity = 100_000.0 * (1.0 + compounded(strategy))
    benchmark_returns = {
        "SPY": compounded(spy),
        "RISK_MATCHED_SPY": compounded(spy * risk_scale),
        "DIVERSIFIED_BASELINE": compounded(baseline),
        "FINANCED_SPY_SAME_RISK": compounded(financed),
    }
    cost_bps = (
        float(candidate_trades["cost_return"].abs().mean() * 10_000.0)
        if not candidate_trades.empty
        else None
    )
    return shadow.model_copy(
        update={
            "resolved_through": pd.Timestamp(prospective_dates[-1]).date(),
            "sessions": len(prospective_dates),
            "trades": len(candidate_trades),
            "effective_resolved_outcomes": float(len(candidate_trades)),
            "equity": net_equity,
            "net_return": net_equity / 100_000.0 - 1.0,
            "benchmark_returns": benchmark_returns,
            "realized_slippage_bps": cost_bps,
            "warnings": (
                "Previously accessed opening family: server-owned prospective observation "
                "only; it is permanently capital-ineligible.",
                "The opening engine enforces participation and stressed costs, but its "
                "quantity-free historical trade rows are not canonical broker fills.",
            ),
        }
    )


def monitor_promoted_shadow(
    store: ResearchStore,
    shadow: ShadowStrategyV1,
) -> ShadowStrategyV1:
    """Fail closed when a promoted independent book materially deteriorates."""
    if shadow.status not in {
        CampaignLifecycle.PROMOTED,
        CampaignLifecycle.DEGRADED,
        CampaignLifecycle.SUSPENDED,
    }:
        return shadow
    risk_benchmark = float(shadow.benchmark_returns.get("RISK_MATCHED_SPY", 0.0))
    relative_return = shadow.net_return - risk_benchmark
    if shadow.net_return <= -0.20:
        state = CampaignLifecycle.SUSPENDED
        message = "Shadow drawdown breached 20%; canonical sleeve is suspended."
    elif relative_return <= -0.10:
        state = (
            CampaignLifecycle.SUSPENDED
            if shadow.status is CampaignLifecycle.SUSPENDED
            else CampaignLifecycle.DEGRADED
        )
        message = "Shadow lagged risk-matched SPY by at least 10%; canonical sleeve is removed."
    elif shadow.status in {CampaignLifecycle.DEGRADED, CampaignLifecycle.SUSPENDED}:
        state = CampaignLifecycle.PROMOTED
        message = "Shadow recovered its absolute and benchmark-relative health gates."
    else:
        return shadow
    campaign = store.campaign(shadow.campaign_id)
    if campaign is not None and campaign.lifecycle is not state:
        store.upsert_campaign(
            campaign.model_copy(
                update={
                    "lifecycle": state,
                    "updated_at": datetime.now(UTC),
                    "promotion_eligible": state is CampaignLifecycle.PROMOTED,
                    "next_action": message,
                }
            )
        )
    return shadow.model_copy(update={"status": state, "warnings": (*shadow.warnings, message)})


def review_daily_shadow_promotion(
    store: ResearchStore,
    template: CampaignTemplate,
    shadow: ShadowStrategyV1,
) -> ShadowStrategyV1:
    """Run the immutable V2 promotion family once the prospective clock is complete."""
    if shadow.sessions < 252 or shadow.status is not CampaignLifecycle.PAPER_SHADOW:
        return shadow
    with store.catalog.connect() as connection:
        existing = connection.execute(
            """SELECT payload FROM promotion_decisions_v2
               WHERE sleeve_id = ? AND artifact_hash = ?""",
            [shadow.strategy_id, shadow.artifact_hash],
        ).fetchone()
    if existing is not None:
        return shadow
    trials = RecommendationRegistry(store.catalog).trials(template.template_id)
    candidate_id = shadow.strategy_id.rsplit(":", 1)[-1]
    parameters = next(
        dict(trial.parameters)
        for trial in trials
        if stable_hash(trial.parameters)[:16] == candidate_id
    )
    symbol = str(parameters.get("symbol", "SPY"))
    asset_kind = AssetKind.ETF if symbol in ALL_RESEARCH_ETFS else AssetKind.STOCK
    resolved_outcomes = shadow.effective_resolved_outcomes
    if asset_kind is AssetKind.STOCK and resolved_outcomes < 100:
        return shadow
    symbols = tuple(
        dict.fromkeys(
            (
                *(str(item.get("symbol", "SPY")) for item in template.candidate_family),
                "SPY",
                "TLT",
                "SHY",
                "GLD",
            )
        )
    )
    with store.catalog.guard.unlock(
        reason="single frozen V2 promotion review",
        experiment_id=template.template_id,
    ) as unlock_key:
        panel = store.catalog.load_panel(symbols=symbols, unlock_key=unlock_key)
    prices = panel.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    execution_prices = _adjusted_open_matrix(panel)
    common = prices.index.intersection(execution_prices.index)
    prices = prices.loc[common].dropna(subset=list(symbols))
    execution_prices = execution_prices.reindex(prices.index).dropna(subset=list(symbols))
    prices = prices.reindex(execution_prices.index)
    cash_yield = _initial_macro_series(store.catalog, "DGS3MO").div(100.0)
    family_returns = {
        stable_hash(candidate)[:16]: _daily_strategy_returns(
            candidate,
            prices,
            cash_yield,
            execution_prices=execution_prices,
        )[0]
        for candidate in template.candidate_family
        if str(candidate.get("symbol", "SPY")) in prices
    }
    strategy = family_returns[candidate_id]
    spy = _next_open_returns(execution_prices["SPY"])
    risk_spy = risk_match_benchmark(strategy, spy)
    baseline = _next_open_frame(execution_prices[["SPY", "TLT", "SHY", "GLD"]]).mean(axis=1)
    annual_rate = float(cash_yield.reindex(prices.index).ffill().fillna(0.0).mean())
    target_volatility = float(strategy.std(ddof=1) * np.sqrt(252))
    financed_spy = financed_risk_matched_spy(
        spy,
        target_volatility=target_volatility,
        financing_rate=annual_rate + 0.02,
        cash_yield=annual_rate,
    )
    frame = pd.DataFrame(family_returns).reindex(prices.index).fillna(0.0)
    spa = spa_test(
        baseline,
        frame,
        reps=max(100, store.catalog.cfg.validation.bootstrap_samples),
        block_size=store.catalog.cfg.validation.block_length_sessions,
        seed=store.catalog.cfg.project.random_seed,
    )
    superior = stepm_superior(
        baseline,
        frame,
        reps=max(100, store.catalog.cfg.validation.bootstrap_samples),
        block_size=store.catalog.cfg.validation.block_length_sessions,
        seed=store.catalog.cfg.project.random_seed,
    )
    overfitting = _family_overfitting_diagnostics(
        store, baseline, frame, batch_id=f"{template.template_id}--{candidate_id}"
    )
    fold_size = len(strategy) // 5
    folds = {
        f"outer_{index + 1}": strategy.iloc[index * fold_size : (index + 1) * fold_size]
        for index in range(5)
    }
    conservative, _position_series = _daily_strategy_returns(
        parameters,
        prices,
        cash_yield,
        cost_per_exposure_change=0.0015,
        execution_prices=execution_prices,
    )
    stressed, _ = _daily_strategy_returns(
        parameters,
        prices,
        cash_yield,
        cost_per_exposure_change=0.003,
        execution_prices=execution_prices,
    )
    delayed, _ = _daily_strategy_returns(
        parameters,
        prices,
        cash_yield,
        cost_per_exposure_change=0.0015,
        extra_execution_delay=1,
        execution_prices=execution_prices,
    )
    risk_growth = annualized_log_growth(risk_spy)
    stressed_growth = annualized_log_growth(stressed)
    conservative_growth = annualized_log_growth(conservative)
    delayed_growth = annualized_log_growth(delayed)
    symbol_rows = panel.loc[panel["symbol"] == symbol].sort_values("date").tail(60)
    median_adv = float((symbol_rows["close"] * symbol_rows["volume"]).median())
    capacity_pass = bool(median_adv > 0 and 0.10 * median_adv >= 100_000.0)
    funding_passes = {}
    for spread in (200, 400, 800):
        comparison = financed_risk_matched_spy(
            spy,
            target_volatility=target_volatility,
            financing_rate=annual_rate + spread / 10_000.0,
            cash_yield=annual_rate,
        )
        funding_passes[f"financing_{spread}bps"] = annualized_log_growth(
            strategy
        ) > annualized_log_growth(comparison)
    stress_scenarios = {
        "conservative": conservative_growth > risk_growth,
        "stress": stressed_growth > risk_growth,
        "delayed_fill": delayed_growth > risk_growth,
        "liquidity": capacity_pass,
        "participation": capacity_pass,
        "adverse_execution": stressed_growth > risk_growth,
        **funding_passes,
    }
    prospective = ProspectiveEvidenceV2(
        sleeve_id=shadow.strategy_id,
        frozen_artifact_hash=shadow.artifact_hash,
        prospective_start=shadow.started_at.date(),
        resolved_through=shadow.resolved_through,
        prospective_sessions=shadow.sessions,
        effective_resolved_outcomes=resolved_outcomes,
    )
    registry = RecommendationRegistry(store.catalog)
    registry.append_prospective_evidence(prospective)
    decision = evaluate_promotion(
        PromotionInputs(
            sleeve_id=shadow.strategy_id,
            artifact_hash=shadow.artifact_hash,
            asset_kind=asset_kind,
            fold_returns=folds,
            strategy_returns=strategy,
            risk_matched_spy=risk_spy,
            diversified_baseline=baseline,
            buy_now_spy=spy,
            financed_spy_same_risk=financed_spy,
            spa_consistent_pvalue=float(spa["p_consistent"]),
            stepm_superior_ids=(shadow.strategy_id,) if candidate_id in superior else (),
            stress_scenarios=stress_scenarios,
            prospective_evidence=prospective,
            **_overfitting_promotion_fields(
                overfitting, candidate_id, store.catalog.cfg.validation
            ),
        ),
        seed=store.catalog.cfg.project.random_seed,
    )
    registry.save_promotion(decision)
    campaign = store.campaign(template.template_id)
    if campaign is None:
        raise ValueError(f"campaign {template.template_id} was not registered")
    if not decision.promoted:
        store.upsert_campaign(
            campaign.model_copy(
                update={
                    "lifecycle": CampaignLifecycle.REJECTED,
                    "updated_at": datetime.now(UTC),
                    "failure_reasons": decision.failure_reasons,
                    "next_action": "Promotion review failed; retain zero canonical weight.",
                }
            )
        )
        return shadow.model_copy(
            update={
                "status": CampaignLifecycle.REJECTED,
                "warnings": decision.failure_reasons,
            }
        )
    estimate = stationary_mean_test(
        strategy,
        n_boot=max(100, store.catalog.cfg.validation.bootstrap_samples),
        mean_block_length=store.catalog.cfg.validation.block_length_sessions,
        seed=store.catalog.cfg.project.random_seed,
    )
    sleeve = SleeveContributionV2(
        sleeve_id=shadow.strategy_id,
        artifact_hash=shadow.artifact_hash,
        horizon_sessions=max(1, int(parameters.get("hold", 1))),
        family=template.family,
        symbol_weights=(
            WeightV2(
                symbol=symbol,
                weight=1.0,
                asset_kind=asset_kind,
                sector=(
                    "inverse_bearish"
                    if symbol in INVERSE_ETFS
                    else "broad_equity"
                    if asset_kind is AssetKind.ETF
                    else "unknown"
                ),
            ),
        ),
        expected_net_return=float(estimate["mean"]) * 252.0,
        expected_return_lower_95=float(estimate["ci_low"]) * 252.0,
        effective_sample_size=session_effective_sample_size(strategy),
        evidence_grade=EvidenceGrade.PROMOTED,
        signal_parameters=parameters,
    )
    store.save_promoted_sleeve(sleeve)
    store.upsert_campaign(
        campaign.model_copy(
            update={
                "lifecycle": CampaignLifecycle.PROMOTED,
                "updated_at": datetime.now(UTC),
                "promotion_eligible": True,
                "next_action": "Include the immutable sleeve in the next canonical publication.",
            }
        )
    )
    return shadow.model_copy(
        update={
            "status": CampaignLifecycle.PROMOTED,
            "warnings": ("Passed immutable V2 promotion; awaiting canonical publication.",),
        }
    )


def review_event_shadow_promotion(
    store: ResearchStore,
    template: CampaignTemplate,
    shadow: ShadowStrategyV1,
) -> ShadowStrategyV1:
    """Apply the same frozen V2 gates to a prospectively observed event rule."""
    if shadow.sessions < 252 or shadow.status is not CampaignLifecycle.PAPER_SHADOW:
        return shadow
    with store.catalog.connect() as connection:
        existing = connection.execute(
            """SELECT payload FROM promotion_decisions_v2
               WHERE sleeve_id = ? AND artifact_hash = ?""",
            [shadow.strategy_id, shadow.artifact_hash],
        ).fetchone()
    if existing is not None:
        return shadow
    candidate_id = shadow.strategy_id.rsplit(":", 1)[-1]
    trials = RecommendationRegistry(store.catalog).trials(template.template_id)
    parameters = next(
        dict(trial.parameters)
        for trial in trials
        if stable_hash(trial.parameters)[:16] == candidate_id
    )
    stock_symbols = next(
        requirement.symbols
        for requirement in template.requirements
        if requirement.dataset == "earnings"
    )
    rule = str(parameters["rule"])
    is_stock_strategy = rule in {"earnings_drift", "short_sale_volume_reversal"}
    asset_kind = AssetKind.STOCK if is_stock_strategy else AssetKind.ETF
    if is_stock_strategy and shadow.effective_resolved_outcomes < 100:
        return shadow
    symbols = tuple(
        dict.fromkeys(("SPY", "QQQ", "IWM", "DIA", *stock_symbols, "TLT", "SHY", "GLD"))
    )
    with store.catalog.guard.unlock(
        reason="single frozen event V2 promotion review",
        experiment_id=template.template_id,
    ) as unlock_key:
        panel = store.catalog.load_panel(symbols=symbols, unlock_key=unlock_key)
    prices = panel.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    prices = prices.dropna(subset=["SPY"])
    event_signals, short_signals, vix, dff, cash_yield = _event_context(
        store.catalog, prices, stock_symbols
    )
    family_returns = {
        stable_hash(candidate)[:16]: _event_candidate_returns(
            candidate,
            prices,
            stock_symbols,
            event_signals,
            short_signals,
            vix,
            dff,
            cash_yield,
        )[0]
        for candidate in template.candidate_family
    }
    strategy = family_returns[candidate_id]
    spy = prices["SPY"].pct_change(fill_method=None).fillna(0.0)
    risk_spy = risk_match_benchmark(strategy, spy)
    baseline = (
        prices[["SPY", "TLT", "SHY", "GLD"]].pct_change(fill_method=None).fillna(0.0).mean(axis=1)
    )
    annual_rate = float(cash_yield.reindex(prices.index).ffill().fillna(0.0).mean())
    target_volatility = float(strategy.std(ddof=1) * np.sqrt(252))
    financed_spy = financed_risk_matched_spy(
        spy,
        target_volatility=target_volatility,
        financing_rate=annual_rate + 0.02,
        cash_yield=annual_rate,
    )
    frame = pd.DataFrame(family_returns).reindex(prices.index).fillna(0.0)
    spa = spa_test(
        baseline,
        frame,
        reps=max(100, store.catalog.cfg.validation.bootstrap_samples),
        block_size=store.catalog.cfg.validation.block_length_sessions,
        seed=store.catalog.cfg.project.random_seed,
    )
    superior = stepm_superior(
        baseline,
        frame,
        reps=max(100, store.catalog.cfg.validation.bootstrap_samples),
        block_size=store.catalog.cfg.validation.block_length_sessions,
        seed=store.catalog.cfg.project.random_seed,
    )
    overfitting = _family_overfitting_diagnostics(
        store, baseline, frame, batch_id=f"{template.template_id}--{candidate_id}"
    )
    fold_size = len(strategy) // 5
    folds = {
        f"outer_{index + 1}": strategy.iloc[index * fold_size : (index + 1) * fold_size]
        for index in range(5)
    }
    conservative = _event_candidate_returns(
        parameters,
        prices,
        stock_symbols,
        event_signals,
        short_signals,
        vix,
        dff,
        cash_yield,
        cost_per_exposure_change=0.0015,
    )[0]
    stressed = _event_candidate_returns(
        parameters,
        prices,
        stock_symbols,
        event_signals,
        short_signals,
        vix,
        dff,
        cash_yield,
        cost_per_exposure_change=0.003,
    )[0]
    delayed = _event_candidate_returns(
        parameters,
        prices,
        stock_symbols,
        event_signals,
        short_signals,
        vix,
        dff,
        cash_yield,
        cost_per_exposure_change=0.0015,
        extra_execution_delay=1,
    )[0]
    risk_growth = annualized_log_growth(risk_spy)
    strategy_symbols = stock_symbols if is_stock_strategy else ("SPY",)
    median_advs = []
    for symbol in strategy_symbols:
        rows = panel.loc[panel["symbol"] == symbol].sort_values("date").tail(60)
        median_advs.append(float((rows["close"] * rows["volume"]).median()))
    capacity_pass = bool(median_advs and min(median_advs) * 0.10 >= 100_000.0)
    funding_passes = {}
    for spread in (200, 400, 800):
        comparison = financed_risk_matched_spy(
            spy,
            target_volatility=target_volatility,
            financing_rate=annual_rate + spread / 10_000.0,
            cash_yield=annual_rate,
        )
        funding_passes[f"financing_{spread}bps"] = annualized_log_growth(
            strategy
        ) > annualized_log_growth(comparison)
    stress_scenarios = {
        "conservative": annualized_log_growth(conservative) > risk_growth,
        "stress": annualized_log_growth(stressed) > risk_growth,
        "delayed_fill": annualized_log_growth(delayed) > risk_growth,
        "liquidity": capacity_pass,
        "participation": capacity_pass,
        "adverse_execution": annualized_log_growth(stressed) > risk_growth,
        **funding_passes,
    }
    prospective = ProspectiveEvidenceV2(
        sleeve_id=shadow.strategy_id,
        frozen_artifact_hash=shadow.artifact_hash,
        prospective_start=shadow.started_at.date(),
        resolved_through=shadow.resolved_through,
        prospective_sessions=shadow.sessions,
        effective_resolved_outcomes=shadow.effective_resolved_outcomes,
    )
    registry = RecommendationRegistry(store.catalog)
    registry.append_prospective_evidence(prospective)
    decision = evaluate_promotion(
        PromotionInputs(
            sleeve_id=shadow.strategy_id,
            artifact_hash=shadow.artifact_hash,
            asset_kind=asset_kind,
            fold_returns=folds,
            strategy_returns=strategy,
            risk_matched_spy=risk_spy,
            diversified_baseline=baseline,
            buy_now_spy=spy,
            financed_spy_same_risk=financed_spy,
            spa_consistent_pvalue=float(spa["p_consistent"]),
            stepm_superior_ids=(shadow.strategy_id,) if candidate_id in superior else (),
            stress_scenarios=stress_scenarios,
            prospective_evidence=prospective,
            **_overfitting_promotion_fields(
                overfitting, candidate_id, store.catalog.cfg.validation
            ),
        ),
        seed=store.catalog.cfg.project.random_seed,
    )
    registry.save_promotion(decision)
    campaign = store.campaign(template.template_id)
    if campaign is None:
        raise ValueError(f"campaign {template.template_id} was not registered")
    if not decision.promoted:
        store.upsert_campaign(
            campaign.model_copy(
                update={
                    "lifecycle": CampaignLifecycle.REJECTED,
                    "updated_at": datetime.now(UTC),
                    "failure_reasons": decision.failure_reasons,
                    "next_action": "Event promotion review failed; retain zero canonical weight.",
                }
            )
        )
        return shadow.model_copy(
            update={"status": CampaignLifecycle.REJECTED, "warnings": decision.failure_reasons}
        )
    estimate = stationary_mean_test(
        strategy,
        n_boot=max(100, store.catalog.cfg.validation.bootstrap_samples),
        mean_block_length=store.catalog.cfg.validation.block_length_sessions,
        seed=store.catalog.cfg.project.random_seed,
    )
    weight = 1.0 / len(strategy_symbols)
    sleeve = SleeveContributionV2(
        sleeve_id=shadow.strategy_id,
        artifact_hash=shadow.artifact_hash,
        horizon_sessions=max(1, int(parameters.get("horizon", 1))),
        family=template.family,
        symbol_weights=tuple(
            WeightV2(
                symbol=symbol,
                weight=weight,
                asset_kind=asset_kind,
                sector="broad_equity" if asset_kind is AssetKind.ETF else "event_stock",
            )
            for symbol in strategy_symbols
        ),
        expected_net_return=float(estimate["mean"]) * 252.0,
        expected_return_lower_95=float(estimate["ci_low"]) * 252.0,
        effective_sample_size=session_effective_sample_size(strategy),
        evidence_grade=EvidenceGrade.PROMOTED,
        signal_parameters={
            **parameters,
            "evaluator": "events",
            "stock_symbols": list(stock_symbols if is_stock_strategy else ()),
        },
    )
    store.save_promoted_sleeve(sleeve)
    store.upsert_campaign(
        campaign.model_copy(
            update={
                "lifecycle": CampaignLifecycle.PROMOTED,
                "updated_at": datetime.now(UTC),
                "promotion_eligible": True,
                "next_action": "Include the active event sleeve in canonical publication.",
            }
        )
    )
    return shadow.model_copy(
        update={
            "status": CampaignLifecycle.PROMOTED,
            "warnings": ("Passed immutable event V2 promotion.",),
        }
    )
