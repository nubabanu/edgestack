"""Opening-spike/fade research campaign.

This module is deliberately isolated from recommendation publication, portfolio
weights, broker tickets, and the V2 promotion pipeline.  It turns timestamped
intraday OHLCV into auditable occurrence statistics and paper-only strategy
diagnostics.  Unsupported candidates stay in the trial ledger so missing data
cannot quietly shrink the multiple-testing family.

Bar timestamps are interpreted as bar *start* times.  Signals use completed
bars and entries occur at the following bar's open.  When a stop and target are
both touched inside one OHLC bar, the stop wins.  These conventions are more
important at 15-minute resolution, where the true intrabar path is unknowable.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import math
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edgestack.config import load_config
from edgestack.data.catalog import DataCatalog, atomic_write_bytes
from edgestack.data.schemas import INTRADAY_BAR_COLUMNS, validate_intraday_bars
from edgestack.discovery.multiple_testing import benjamini_hochberg
from edgestack.exceptions import DataError, ValidationError
from edgestack.execution.costs import CostModel
from edgestack.types import CostScenario, Side
from edgestack.validation.advanced_tests import spa_test, stepm_superior
from edgestack.validation.bootstrap import block_bootstrap_ci, mean_pvalue_bootstrap
from edgestack.validation.metrics import (
    deflated_sharpe_ratio,
    effective_sample_size,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    var_es,
)

EXCHANGE_TZ = "America/New_York"
DISCLAIMER = (
    "Research and paper-trading output only. Previously accessed history; not investment "
    "advice and never an instruction to trade."
)


class DataSufficiency(StrEnum):
    INSUFFICIENT = "INSUFFICIENT"
    EXPLORATORY = "EXPLORATORY"
    RESEARCH_USABLE = "RESEARCH_USABLE"
    PROMOTION_ELIGIBLE = "PROMOTION_ELIGIBLE"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DefinitionConfig(_FrozenModel):
    version: str = "opening-fade-occurrence-v1"
    positive_gap_min: float = Field(default=0.002, ge=0)
    spike_atr_min: float = Field(default=0.15, ge=0)
    opening_window_minutes: int = Field(default=15, ge=5, le=30)
    evaluation_time: str = "10:30"
    premarket_start: str = "04:00"
    regular_open: str = "09:30"
    regular_close: str = "16:00"
    gap_buckets: tuple[float, ...] = (0.0, 0.002, 0.004, 0.0075, 0.0125)
    spike_bps_thresholds: tuple[float, ...] = (10.0, 20.0, 35.0, 50.0)
    spike_atr_thresholds: tuple[float, ...] = (0.10, 0.20, 0.30)

    @field_validator("evaluation_time", "premarket_start", "regular_open", "regular_close")
    @classmethod
    def _valid_clock(cls, value: str) -> str:
        time.fromisoformat(value)
        return value


class CostConfig(_FrozenModel):
    commission_bps: float = Field(default=0.0, ge=0)
    half_spread_bps: float = Field(default=0.5, ge=0)
    base_slippage_bps: float = Field(default=0.75, ge=0)
    impact_coeff_bps: float = Field(default=5.0, ge=0)
    short_borrow_annualized: float = Field(default=0.03, ge=0)
    sec_fee_bps: float = Field(default=0.03, ge=0)
    assumed_participation: float = Field(default=0.001, ge=0, le=0.10)
    maximum_bar_participation: float = Field(default=0.05, gt=0, le=0.10)


class ValidationConfig(_FrozenModel):
    train_sessions: int = Field(default=252, ge=60)
    test_sessions: int = Field(default=63, ge=20)
    n_folds: int = Field(default=4, ge=2)
    embargo_sessions: int = Field(default=1, ge=1)
    bootstrap_samples: int = Field(default=2000, ge=100)
    block_length: int = Field(default=20, ge=1)
    minimum_effective_sample_size: int = Field(default=100, ge=20)
    minimum_positive_fold_fraction: float = Field(default=0.70, ge=0.5, le=1.0)
    minimum_deflated_sharpe_probability: float = Field(default=0.95, ge=0.5, le=1.0)
    maximum_drawdown_floor: float = Field(default=-0.20, ge=-1.0, lt=0)
    minimum_instruments: int = Field(default=2, ge=2)
    minimum_years: int = Field(default=2, ge=2)
    maximum_single_instrument_trade_fraction: float = Field(default=0.75, gt=0.5, le=1.0)
    minimum_neighborhood_positive_fraction: float = Field(default=0.50, ge=0.5, le=1.0)
    fdr_alpha: float = Field(default=0.05, gt=0, lt=1)
    random_seed: int = 42


class CandidateConfig(_FrozenModel):
    candidate_id: str
    family: str
    direction: str
    rule: str
    params: dict[str, Any] = Field(default_factory=dict)
    requires: tuple[str, ...] = ()

    @field_validator("family")
    @classmethod
    def _family(cls, value: str) -> str:
        if value not in set("ABCDEFGH"):
            raise ValueError("family must be A through H")
        return value

    @field_validator("direction")
    @classmethod
    def _direction(cls, value: str) -> str:
        if value not in {"SHORT", "LONG", "NONE"}:
            raise ValueError("direction must be SHORT, LONG, or NONE")
        return value


class OpeningFadeConfig(_FrozenModel):
    campaign_version: str = "opening-fade-v1"
    repository_config: Path = Path("configs/live.yaml")
    provider: str = "yahoo"
    primary_interval_minutes: int = 5
    fallback_interval_minutes: int = 15
    primary_symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM", "DIA")
    secondary_symbols: tuple[str, ...] = (
        "XLK",
        "SMH",
        "XLF",
        "XLE",
        "XLY",
        "XLP",
        "TLT",
        "GLD",
    )
    stock_symbols: tuple[str, ...] = (
        "AAPL",
        "MSFT",
        "NVDA",
        "AMD",
        "AMZN",
        "META",
        "GOOGL",
        "TSLA",
    )
    artifacts_dir: Path = Path("artifacts/research/opening_fade")
    event_calendar_csv: Path | None = None
    previously_accessed_start: date = date(2024, 1, 1)
    minimum_stock_price: float = Field(default=5.0, gt=0)
    minimum_stock_opening_dollar_volume: float = Field(default=1_000_000.0, gt=0)
    definition: DefinitionConfig = Field(default_factory=DefinitionConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    candidates: tuple[CandidateConfig, ...]

    @model_validator(mode="after")
    def _bounded_family(self) -> OpeningFadeConfig:
        ids = [item.candidate_id for item in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_id values must be unique")
        missing = set("ABCDEFGH") - {item.family for item in self.candidates}
        if missing:
            raise ValueError(f"bounded family must represent A-H; missing {sorted(missing)}")
        if len(ids) > 200:
            raise ValueError("candidate family is not bounded (maximum 200)")
        return self

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(self.primary_symbols + self.secondary_symbols + self.stock_symbols)
        )

    def stable_hash(self) -> str:
        payload = self.model_dump(mode="json")
        return _hash_json(payload)


class PointInTimeEventCalendar(Protocol):
    def flags_known_at(self, symbol: str, at: pd.Timestamp) -> tuple[str, ...]: ...


class CsvEventCalendar:
    """Optional point-in-time event input.

    Required columns are ``event_time_utc``, ``known_at_utc``, ``event_type``
    and optional ``symbol``.  A flag is returned only when ``known_at_utc`` is
    no later than the decision timestamp.
    """

    def __init__(self, path: Path) -> None:
        frame = pd.read_csv(path)
        required = {"event_time_utc", "known_at_utc", "event_type"}
        missing = required - set(frame)
        if missing:
            raise DataError(f"event calendar missing columns: {sorted(missing)}")
        self.frame = frame.copy()
        self.frame["event_time_utc"] = pd.to_datetime(self.frame["event_time_utc"], utc=True)
        self.frame["known_at_utc"] = pd.to_datetime(self.frame["known_at_utc"], utc=True)
        if "symbol" not in self.frame:
            self.frame["symbol"] = ""
        self.frame["symbol"] = self.frame["symbol"].fillna("")

    def flags_known_at(self, symbol: str, at: pd.Timestamp) -> tuple[str, ...]:
        utc = _as_utc(at)
        local_day = utc.tz_convert(EXCHANGE_TZ).date()
        frame = self.frame.loc[
            (self.frame["known_at_utc"] <= utc)
            & (self.frame["event_time_utc"].dt.tz_convert(EXCHANGE_TZ).dt.date == local_day)
            & (self.frame["symbol"].astype(str).str.upper().isin({"", symbol.upper()}))
        ]
        return tuple(sorted(set(frame["event_type"].astype(str))))


@dataclass(frozen=True)
class SimulatedTrade:
    candidate_id: str
    family: str
    symbol: str
    session: str
    direction: str
    signal_time_et: str
    signal_time_utc: str
    entry_time_et: str
    entry_time_utc: str
    exit_time_et: str
    exit_time_utc: str
    entry: float
    stop: float
    target: float
    exit: float
    exit_reason: str
    gross_return: float
    net_return: float
    cost_return: float
    mfe: float
    mae: float
    holding_minutes: int
    regime: str
    event_flags: tuple[str, ...]
    warnings: tuple[str, ...]


def load_campaign_config(path: str | Path) -> OpeningFadeConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return OpeningFadeConfig.model_validate(raw)


def _hash_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_utc(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _to_float(value: Any) -> float:
    return float(value)


def _clock(value: str) -> time:
    return time.fromisoformat(value)


def classify_data_sufficiency(
    sessions: int, *, interval_minutes: int, prospectively_collected: bool = False
) -> DataSufficiency:
    """Conservative evidence label; this campaign never self-promotes."""
    if interval_minutes > 15 or sessions < 60:
        return DataSufficiency.INSUFFICIENT
    if sessions < 252:
        return DataSufficiency.EXPLORATORY
    if interval_minutes <= 5 and sessions >= 252:
        # Even a long/high-resolution archive still needs independent V2 gates;
        # Prospective collection is reported but cannot auto-award promotion.
        _ = prospectively_collected
        return DataSufficiency.RESEARCH_USABLE
    return DataSufficiency.EXPLORATORY


def _provider_capabilities(provider: str) -> dict[str, Any]:
    if provider == "yahoo":
        return {
            "intervals": {
                "1m": "7 calendar days",
                "5m": "59 calendar days",
                "15m": "59 calendar days",
                "60m": "729 calendar days",
            },
            "extended_hours_optional": True,
            "timestamp_storage": "UTC",
            "fields": ["open", "high", "low", "close", "volume"],
            "bid_ask": False,
            "historical_auction_imbalance": False,
            "limitations": [
                "unofficial and throttle-prone endpoint",
                "survivorship-biased symbol availability",
                "OHLCV trade bars, not executable bid/ask quotes",
                "history may be restated after corporate actions",
            ],
        }
    return {
        "intervals": {},
        "extended_hours_optional": "provider-dependent",
        "timestamp_storage": "canonical input must be timezone-aware; catalog stores UTC",
        "fields": list(INTRADAY_BAR_COLUMNS),
        "bid_ask": False,
        "limitations": ["capabilities were not declared by this campaign"],
    }


def audit_local_data(repo_root: Path, cfg: OpeningFadeConfig) -> dict[str, Any]:
    """Inspect local evidence without mistaking archive freshness for sufficiency."""
    root = repo_root / "data" / "curated" / "intraday"
    rows: list[dict[str, Any]] = []
    for interval in (1, 5, 15, 60):
        for symbol in cfg.symbols:
            path = root / f"{interval}m" / f"{symbol}.parquet"
            if not path.exists():
                rows.append(
                    {
                        "symbol": symbol,
                        "interval_minutes": interval,
                        "exists": False,
                        "rows": 0,
                        "sessions": 0,
                        "classification": DataSufficiency.INSUFFICIENT,
                    }
                )
                continue
            frame = validate_intraday_bars(pd.read_parquet(path), context=str(path))
            utc = pd.to_datetime(frame["timestamp"], utc=True)
            et = utc.dt.tz_convert(EXCHANGE_TZ)
            session_count = int(et.dt.date.nunique())
            premarket = (et.dt.time >= time(4)) & (et.dt.time < time(9, 30))
            afterhours = et.dt.time >= time(16)
            rows.append(
                {
                    "symbol": symbol,
                    "interval_minutes": interval,
                    "exists": True,
                    "path": str(path.relative_to(repo_root)),
                    "sha256": _hash_file(path),
                    "rows": len(frame),
                    "sessions": session_count,
                    "start_utc": utc.min().isoformat(),
                    "end_utc": utc.max().isoformat(),
                    "premarket_rows": int(premarket.sum()),
                    "afterhours_rows": int(afterhours.sum()),
                    "positive_volume_fraction": float((frame["volume"] > 0).mean()),
                    "duplicates": int(
                        frame.duplicated(["symbol", "timestamp", "interval_minutes"]).sum()
                    ),
                    "classification": classify_data_sufficiency(
                        session_count, interval_minutes=interval
                    ),
                }
            )
    available = [row for row in rows if row["exists"]]
    max_class = DataSufficiency.INSUFFICIENT
    if any(row["classification"] == DataSufficiency.RESEARCH_USABLE for row in available):
        max_class = DataSufficiency.RESEARCH_USABLE
    elif any(row["classification"] == DataSufficiency.EXPLORATORY for row in available):
        max_class = DataSufficiency.EXPLORATORY
    daily_dir = repo_root / "data" / "curated" / "prices"
    daily = {symbol: (daily_dir / f"{symbol}.parquet").exists() for symbol in cfg.symbols}
    futures = [
        symbol
        for symbol in ("ES", "MES", "NQ", "MNQ", "RTY", "M2K")
        if (root / "5m" / f"{symbol}.parquet").exists()
    ]
    event_path = repo_root / cfg.event_calendar_csv if cfg.event_calendar_csv else None
    return {
        "campaign": cfg.campaign_version,
        "checked_at": datetime.now(UTC).isoformat(),
        "overall_classification": max_class,
        "provider": cfg.provider,
        "provider_capabilities": _provider_capabilities(cfg.provider),
        "files": rows,
        "daily_context_available": daily,
        "premarket_available": any(row.get("premarket_rows", 0) > 0 for row in available),
        "afterhours_available": any(row.get("afterhours_rows", 0) > 0 for row in available),
        "timestamps": "timezone-aware UTC; converted to America/New_York for session logic",
        "volume": "bar volume is available; auction/venue decomposition is not",
        "vwap": "OHLCV typical-price approximation only; no consolidated trade tape",
        "bid_ask": "not available historically",
        "index_futures": futures,
        "tradable_series": "ETF proxies are the only clean cross-index series currently targeted",
        "economic_calendar": {
            "available": bool(event_path and event_path.exists()),
            "path": str(event_path) if event_path else None,
            "requirement": "event_time_utc and known_at_utc are both required",
        },
        "breadth": {
            "research_usable": False,
            "reason": (
                "no timestamp-aligned point-in-time component intraday panel; "
                "current membership would leak survivors"
            ),
        },
        "promotion_eligible": False,
        "warnings": [
            "A short forward archive cannot establish historical significance.",
            "15-minute bars cannot reveal whether their high occurred before their low.",
            "60-minute bars are not used to identify the opening-spike sequence.",
            "Daily bars are context only and never reconstruct intraday ordering.",
            DISCLAIMER,
        ],
    }


def normalize_intraday(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Deduplicate overlapping collector pulls and add exchange-local timestamps."""
    raw = frame.copy()
    duplicate_count = int(raw.duplicated(["symbol", "timestamp", "interval_minutes"]).sum())
    raw = raw.drop_duplicates(["symbol", "timestamp", "interval_minutes"], keep="last")
    output = validate_intraday_bars(raw, context="opening_fade")
    output["timestamp_utc"] = pd.to_datetime(output["timestamp"], utc=True)
    output["timestamp_et"] = output["timestamp_utc"].dt.tz_convert(EXCHANGE_TZ)
    output["session"] = pd.to_datetime(output["timestamp_et"].dt.date)
    return output, {"duplicate_bars_dropped": duplicate_count}


def calculate_gap(
    *, previous_close: float, official_open: float, premarket_last: float | None
) -> dict[str, float | str | None]:
    if previous_close <= 0 or official_open <= 0:
        raise ValidationError("previous_close and official_open must be positive")
    official = official_open / previous_close - 1.0
    premarket = None if premarket_last is None else premarket_last / previous_close - 1.0
    return {
        "official_open_gap": official,
        "premarket_gap": premarket,
        "primary_gap": premarket if premarket is not None else official,
        "gap_source": "PREMARKET_LAST" if premarket is not None else "OFFICIAL_OPEN",
    }


def cumulative_vwap(bars: pd.DataFrame) -> pd.Series:
    """OHLCV typical-price approximation; not a quote- or tape-exact VWAP."""
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    volume = bars["volume"].clip(lower=0).astype(float)
    cumulative = volume.cumsum()
    fallback = bars["close"].expanding().mean()
    return (typical.mul(volume).cumsum() / cumulative.replace(0, np.nan)).fillna(fallback)


def opening_range(
    regular_bars: pd.DataFrame, *, minutes: int, interval_minutes: int
) -> dict[str, float | int | bool]:
    needed = math.ceil(minutes / interval_minutes)
    if len(regular_bars) < needed:
        return {"complete": False, "bars": len(regular_bars)}
    window = regular_bars.iloc[:needed]
    return {
        "complete": True,
        "bars": needed,
        "open": float(window.iloc[0]["open"]),
        "high": float(window["high"].max()),
        "low": float(window["low"].min()),
        "close": float(window.iloc[-1]["close"]),
        "volume": float(window["volume"].sum()),
    }


def retracement_flags(
    *, regular_open: float, opening_high: float, evaluation_price: float
) -> dict[str, float | bool | None]:
    denominator = opening_high - regular_open
    fraction = None if denominator <= 0 else (opening_high - evaluation_price) / denominator
    return {
        "retracement_fraction": fraction,
        "any_pullback": evaluation_price < opening_high,
        "retraced_25": fraction is not None and fraction >= 0.25,
        "retraced_50": fraction is not None and fraction >= 0.50,
        "retraced_75": fraction is not None and fraction >= 0.75,
        "retraced_100": fraction is not None and fraction >= 1.00,
        "at_or_below_open": evaluation_price <= regular_open,
        "below_open": evaluation_price < regular_open,
    }


def gap_fill_flags(
    *, previous_close: float, regular_open: float, evaluation_price: float
) -> dict[str, bool]:
    if regular_open >= previous_close:
        midpoint = previous_close + (regular_open - previous_close) * 0.5
        return {
            "half_gap_fill": evaluation_price <= midpoint,
            "full_gap_fill": evaluation_price <= previous_close,
        }
    midpoint = regular_open + (previous_close - regular_open) * 0.5
    return {
        "half_gap_fill": evaluation_price >= midpoint,
        "full_gap_fill": evaluation_price >= previous_close,
    }


def detect_false_breakout(
    bars: pd.DataFrame, *, opening_range_bars: int, buffer_bps: float = 0.0
) -> int | None:
    if len(bars) <= opening_range_bars:
        return None
    level = float(bars.iloc[:opening_range_bars]["high"].max())
    threshold = level * (1.0 + buffer_bps / 10_000.0)
    for position in range(opening_range_bars, len(bars)):
        bar = bars.iloc[position]
        if float(bar["high"]) >= threshold and float(bar["close"]) < level:
            return position
    return None


def detect_overnight_rejection(
    bars: pd.DataFrame, *, overnight_high: float | None, overshoot_bps: float = 0.0
) -> int | None:
    if overnight_high is None or not np.isfinite(overnight_high):
        return None
    trigger = overnight_high * (1.0 + overshoot_bps / 10_000.0)
    for position, raw_bar in enumerate(bars.itertuples(index=False)):
        bar = cast(Any, raw_bar)
        if float(bar.high) >= trigger and float(bar.close) < overnight_high:
            return position
    return None


def next_bar_entry(bars: pd.DataFrame, signal_position: int) -> tuple[int, float]:
    position = signal_position + 1
    if position >= len(bars):
        raise ValidationError("signal has no following bar; same-bar entry is forbidden")
    return position, float(bars.iloc[position]["open"])


def align_cross_index(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner-align completed bars; never forward-fill one market into another."""
    parts = []
    for symbol, frame in frames.items():
        indexed = frame.set_index("timestamp_utc")[["close", "vwap"]].add_prefix(f"{symbol}_")
        parts.append(indexed)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1, join="inner").dropna().sort_index()


def aggregate_session_returns(trades: pd.DataFrame) -> pd.Series:
    """One equal-weight contribution per session, preventing correlation double-counting."""
    if trades.empty:
        return pd.Series(dtype=float, name="net_return")
    return trades.groupby(pd.to_datetime(trades["session"]))["net_return"].mean().sort_index()


def build_daily_context(daily: pd.DataFrame) -> pd.DataFrame:
    """Lagged daily features; today's close is never available at the open."""
    if daily.empty:
        return pd.DataFrame()
    rows = []
    for symbol, group in daily.sort_values("date").groupby("symbol", sort=True):
        item = group.copy().reset_index(drop=True)
        previous_close = item["close"].shift(1)
        true_range = pd.concat(
            [
                item["high"] - item["low"],
                (item["high"] - previous_close).abs(),
                (item["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        returns = item["close"].pct_change()
        item["previous_close"] = previous_close
        item["daily_atr20"] = true_range.rolling(20, min_periods=10).mean().shift(1)
        item["ma20"] = item["close"].rolling(20, min_periods=20).mean().shift(1)
        item["ma50"] = item["close"].rolling(50, min_periods=50).mean().shift(1)
        item["realized_vol20"] = returns.rolling(20, min_periods=10).std().shift(1) * np.sqrt(252)
        item["previous_day_return"] = returns.shift(1)
        item["previous_day_range"] = ((item["high"] - item["low"]) / item["open"]).shift(1)
        prior_high60 = item["close"].rolling(60, min_periods=20).max().shift(1)
        item["recent_drawdown60"] = previous_close / prior_high60 - 1.0
        item["symbol"] = symbol
        rows.append(item)
    output = pd.concat(rows, ignore_index=True)
    return output[
        [
            "symbol",
            "date",
            "previous_close",
            "daily_atr20",
            "ma20",
            "ma50",
            "realized_vol20",
            "previous_day_return",
            "previous_day_range",
            "recent_drawdown60",
        ]
    ].rename(columns={"date": "session"})


def _intraday_previous_close(normalized: pd.DataFrame) -> pd.DataFrame:
    regular = normalized.loc[
        (normalized["timestamp_et"].dt.time >= time(9, 30))
        & (normalized["timestamp_et"].dt.time < time(16))
    ].copy()
    closes = (
        regular.sort_values("timestamp_utc")
        .groupby(["symbol", "session"], as_index=False)
        .tail(1)[["symbol", "session", "close"]]
        .sort_values(["symbol", "session"])
    )
    closes["previous_close"] = closes.groupby("symbol")["close"].shift(1)
    return closes[["symbol", "session", "previous_close"]]


def build_session_features(
    intraday: pd.DataFrame,
    *,
    daily_context: pd.DataFrame,
    definition: DefinitionConfig,
    corporate_actions: pd.DataFrame | None = None,
    event_calendar: PointInTimeEventCalendar | None = None,
    individual_stock_symbols: tuple[str, ...] = (),
    minimum_stock_price: float = 5.0,
    minimum_stock_opening_dollar_volume: float = 1_000_000.0,
) -> tuple[pd.DataFrame, dict[tuple[str, pd.Timestamp], pd.DataFrame]]:
    normalized, quality = normalize_intraday(intraday)
    interval_values = sorted(int(value) for value in normalized["interval_minutes"].unique())
    if len(interval_values) != 1:
        raise ValidationError(f"one interval per study run required, got {interval_values}")
    interval = interval_values[0]
    regular_open = _clock(definition.regular_open)
    regular_close = _clock(definition.regular_close)
    premarket_start = _clock(definition.premarket_start)
    evaluation_clock = _clock(definition.evaluation_time)
    fallback_close = _intraday_previous_close(normalized)
    context = daily_context.copy()
    if context.empty:
        context = fallback_close
    else:
        context["session"] = pd.to_datetime(context["session"])
        context = context.merge(
            fallback_close.rename(columns={"previous_close": "intraday_previous_close"}),
            on=["symbol", "session"],
            how="outer",
        )
        context["previous_close"] = context["previous_close"].fillna(
            context["intraday_previous_close"]
        )

    split_sessions: set[tuple[str, date]] = set()
    if corporate_actions is not None and not corporate_actions.empty:
        splits = corporate_actions.loc[corporate_actions["action_type"] == "split"]
        split_sessions = set()
        for raw_row in splits.itertuples(index=False):
            row = cast(Any, raw_row)
            split_sessions.add((str(row.symbol).upper(), pd.Timestamp(row.date).date()))

    feature_rows: list[dict[str, Any]] = []
    session_bars: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
    opening_volume_history: dict[str, list[float]] = defaultdict(list)
    opening_width_history: dict[str, list[float]] = defaultdict(list)
    stock_symbols = set(individual_stock_symbols)
    for raw_key, group in normalized.groupby(["symbol", "session"], sort=True):
        symbol, session = cast(tuple[str, Any], raw_key)
        session_timestamp = pd.Timestamp(cast(Any, session))
        ordered = group.sort_values("timestamp_utc").reset_index(drop=True)
        premarket = ordered.loc[
            (ordered["timestamp_et"].dt.time >= premarket_start)
            & (ordered["timestamp_et"].dt.time < regular_open)
        ]
        regular = ordered.loc[
            (ordered["timestamp_et"].dt.time >= regular_open)
            & (ordered["timestamp_et"].dt.time < regular_close)
        ].copy()
        if regular.empty:
            feature_rows.append(
                {
                    "symbol": symbol,
                    "session": session,
                    "eligible": False,
                    "exclusion_reason": "NO_REGULAR_SESSION_BARS",
                    "interval_minutes": interval,
                }
            )
            continue
        regular = regular.reset_index(drop=True)
        regular["vwap"] = cumulative_vwap(regular)
        session_bars[(str(symbol), session_timestamp)] = regular
        first_time = cast(pd.Timestamp, regular.iloc[0]["timestamp_et"]).time()
        opening = opening_range(
            regular, minutes=definition.opening_window_minutes, interval_minutes=interval
        )
        context_row = context.loc[
            (context["symbol"] == symbol) & (pd.to_datetime(context["session"]) == session)
        ]
        ctx = context_row.iloc[0].to_dict() if not context_row.empty else {}
        previous_close = ctx.get("previous_close")
        excluded = []
        if first_time != regular_open:
            excluded.append("MISSING_OPENING_BAR")
        if not bool(opening.get("complete")):
            excluded.append("INCOMPLETE_OPENING_RANGE")
        if (
            previous_close is None
            or not np.isfinite(cast(Any, previous_close))
            or _to_float(previous_close) <= 0
        ):
            excluded.append("MISSING_PREVIOUS_CLOSE")
        if (str(symbol), session_timestamp.date()) in split_sessions:
            excluded.append("CORPORATE_ACTION_BOUNDARY")
        # Half sessions remain observable but are classified separately and are
        # excluded from strategy evaluation because requested time exits may not exist.
        last_start = cast(pd.Timestamp, regular.iloc[-1]["timestamp_et"]).time()
        early_close = last_start < time(15, 30)
        if early_close:
            excluded.append("EARLY_CLOSE")
        if excluded:
            feature_rows.append(
                {
                    "symbol": symbol,
                    "session": session,
                    "eligible": False,
                    "exclusion_reason": ",".join(excluded),
                    "interval_minutes": interval,
                    "early_close": early_close,
                }
            )
            continue
        official_open = _to_float(opening["open"])
        premarket_last = float(premarket.iloc[-1]["close"]) if not premarket.empty else None
        gap = calculate_gap(
            previous_close=_to_float(previous_close),
            official_open=official_open,
            premarket_last=premarket_last,
        )
        evaluation = regular.loc[regular["timestamp_et"].dt.time < evaluation_clock]
        if evaluation.empty:
            evaluation = regular.iloc[: int(opening["bars"])]
        evaluation_price = float(evaluation.iloc[-1]["close"])
        opening_high = _to_float(opening["high"])
        opening_low = _to_float(opening["low"])
        opening_width = opening_high / opening_low - 1.0
        prior_opening_volumes = opening_volume_history[str(symbol)]
        prior_opening_widths = opening_width_history[str(symbol)]
        opening_volume = _to_float(opening["volume"])
        relative_opening_volume = (
            opening_volume / float(np.median(prior_opening_volumes))
            if prior_opening_volumes and np.median(prior_opening_volumes) > 0
            else None
        )
        median_prior_opening_width = (
            float(np.median(prior_opening_widths)) if prior_opening_widths else None
        )
        atr = ctx.get("daily_atr20")
        spike = opening_high / official_open - 1.0
        spike_atr = (
            (opening_high - official_open) / _to_float(atr)
            if atr is not None and np.isfinite(atr) and atr > 0
            else None
        )
        post_opening = evaluation.iloc[int(opening["bars"]) :]
        minimum_after_opening = (
            float(post_opening["low"].min()) if not post_opening.empty else evaluation_price
        )
        retracement = retracement_flags(
            regular_open=official_open,
            opening_high=opening_high,
            evaluation_price=minimum_after_opening,
        )
        gap_fill = gap_fill_flags(
            previous_close=_to_float(previous_close),
            regular_open=official_open,
            evaluation_price=minimum_after_opening,
        )
        retracement_times: dict[str, float | None] = {}
        spike_width = opening_high - official_open
        for fraction in (0.25, 0.50, 0.75, 1.0):
            target = opening_high - spike_width * fraction
            touches = post_opening.loc[post_opening["low"] <= target]
            key = f"time_to_retrace_{int(fraction * 100)}_minutes"
            if touches.empty or spike_width <= 0:
                retracement_times[key] = None
            else:
                completion = cast(pd.Timestamp, touches.iloc[0]["timestamp_et"]) + pd.Timedelta(
                    minutes=interval
                )
                session_open = cast(pd.Timestamp, regular.iloc[0]["timestamp_et"])
                retracement_times[key] = float((completion - session_open).total_seconds() / 60.0)
        morning = regular.loc[regular["timestamp_et"].dt.time < time(11)]
        morning_high_pos = int(morning["high"].to_numpy().argmax())
        morning_high_time = morning.iloc[morning_high_pos]["timestamp_et"].isoformat()
        later = regular.iloc[morning_high_pos + 1 :]
        subsequent_new_high = bool(
            (later["high"] > float(morning.iloc[morning_high_pos]["high"])).any()
        )
        afternoon = regular.loc[regular["timestamp_et"].dt.time >= time(12)]
        afternoon_continuation = bool(
            not afternoon.empty
            and float(afternoon["high"].max()) > float(morning.iloc[morning_high_pos]["high"])
            and float(regular.iloc[-1]["close"]) > official_open
        )
        current_vwap = float(evaluation.iloc[-1]["vwap"])
        reached_vwap = bool((evaluation.iloc[int(opening["bars"]) :]["low"] <= current_vwap).any())
        overnight_high = float(premarket["high"].max()) if not premarket.empty else None
        overnight_low = float(premarket["low"].min()) if not premarket.empty else None
        # Scheduled-day flags must already be known at the regular-session open;
        # later publications cannot leak into an earlier candidate signal.
        signal_at = cast(pd.Timestamp, regular.iloc[0]["timestamp_utc"])
        event_flags = (
            event_calendar.flags_known_at(str(symbol), signal_at) if event_calendar else ()
        )
        realized_vol = ctx.get("realized_vol20")
        regime = (
            "UNKNOWN"
            if realized_vol is None or not np.isfinite(realized_vol)
            else "STRESSED"
            if _to_float(realized_vol) >= 0.25
            else "CALM"
        )
        row = {
            "symbol": symbol,
            "session": session,
            "eligible": True,
            "exclusion_reason": "",
            "interval_minutes": interval,
            "early_close": False,
            "previous_close": _to_float(previous_close),
            "premarket_last": premarket_last,
            "premarket_available": not premarket.empty,
            "overnight_high": overnight_high,
            "overnight_low": overnight_low,
            "overnight_range": (
                overnight_high / overnight_low - 1.0
                if overnight_high is not None and overnight_low is not None
                else None
            ),
            "official_open": official_open,
            **gap,
            "opening_range_high": opening_high,
            "opening_range_low": opening_low,
            "opening_range_width": opening_width,
            "opening_range_atr": (
                (opening_high - opening_low) / _to_float(atr)
                if atr is not None and np.isfinite(cast(Any, atr)) and _to_float(atr) > 0
                else None
            ),
            "opening_volume": opening_volume,
            "opening_volume_relative_to_prior_median": relative_opening_volume,
            "opening_spike_return": spike,
            "opening_spike_bps": spike * 10_000.0,
            "opening_spike_atr": spike_atr,
            "opening_spike_relative_to_median_opening_range": (
                spike / median_prior_opening_width
                if median_prior_opening_width is not None and median_prior_opening_width > 0
                else None
            ),
            "opening_spike_relative_to_overnight_range": (
                spike / (overnight_high / overnight_low - 1.0)
                if overnight_high is not None
                and overnight_low is not None
                and overnight_high > overnight_low
                else None
            ),
            "evaluation_price": evaluation_price,
            "minimum_price_by_evaluation": minimum_after_opening,
            "evaluation_vwap": current_vwap,
            "distance_from_vwap": evaluation_price / current_vwap - 1.0,
            "reached_vwap": reached_vwap,
            "daily_atr20": atr,
            "ma20": ctx.get("ma20"),
            "ma50": ctx.get("ma50"),
            "realized_vol20": realized_vol,
            "intraday_realized_range": (
                float(evaluation["high"].max()) / float(evaluation["low"].min()) - 1.0
            ),
            "previous_day_return": ctx.get("previous_day_return"),
            "previous_day_range": ctx.get("previous_day_range"),
            "recent_drawdown60": ctx.get("recent_drawdown60"),
            "gap_atr": (
                (official_open - _to_float(previous_close)) / _to_float(atr)
                if atr is not None and np.isfinite(cast(Any, atr)) and _to_float(atr) > 0
                else None
            ),
            "short_trend_state": (
                "UNKNOWN"
                if ctx.get("ma20") is None or not np.isfinite(cast(Any, ctx.get("ma20")))
                else "UP"
                if official_open >= _to_float(ctx["ma20"])
                else "DOWN"
            ),
            "medium_trend_state": (
                "UNKNOWN"
                if ctx.get("ma50") is None or not np.isfinite(cast(Any, ctx.get("ma50")))
                else "UP"
                if official_open >= _to_float(ctx["ma50"])
                else "DOWN"
            ),
            "regime": regime,
            "event_flags": event_flags,
            "earnings_flag": any("EARNING" in flag.upper() for flag in event_flags),
            "morning_high_time_et": morning_high_time,
            "subsequent_new_high": subsequent_new_high,
            "afternoon_continuation": afternoon_continuation,
            "full_opening_fade": bool(
                _to_float(gap["primary_gap"]) >= definition.positive_gap_min
                and spike_atr is not None
                and spike_atr >= definition.spike_atr_min
                and bool(retracement["retraced_50"])
            ),
            **retracement,
            **retracement_times,
            **gap_fill,
        }
        if str(symbol) in stock_symbols:
            stock_failures = []
            if official_open < minimum_stock_price:
                stock_failures.append("BELOW_MINIMUM_STOCK_PRICE")
            if opening_volume * official_open < minimum_stock_opening_dollar_volume:
                stock_failures.append("INSUFFICIENT_STOCK_OPENING_DOLLAR_VOLUME")
            if stock_failures:
                row["eligible"] = False
                row["exclusion_reason"] = ",".join(stock_failures)
        feature_rows.append(row)
        opening_volume_history[str(symbol)].append(opening_volume)
        opening_width_history[str(symbol)].append(opening_width)
    features = pd.DataFrame(feature_rows)
    features.attrs["quality"] = quality
    return features, session_bars


def _gap_bucket(value: float, boundaries: tuple[float, ...]) -> str:
    if value < boundaries[0]:
        return "negative"
    for lower, upper in pairwise(boundaries):
        if lower <= value < upper:
            return f"{lower:.4f}_{upper:.4f}"
    return f"ge_{boundaries[-1]:.4f}"


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _frequency_record(name: str, values: pd.Series) -> dict[str, Any]:
    clean = values.dropna().astype(bool)
    total = len(clean)
    hits = int(clean.sum())
    lo, hi = wilson_interval(hits, total)
    return {
        "definition": name,
        "sessions": total,
        "count": hits,
        "frequency": hits / total if total else None,
        "ci95": [lo, hi],
        "effective_sample_size": (
            effective_sample_size(clean.astype(float).to_numpy()) if total >= 2 else float(total)
        ),
    }


def occurrence_analysis(features: pd.DataFrame, definition: DefinitionConfig) -> dict[str, Any]:
    eligible = features.loc[features.get("eligible", False) == True].copy()  # noqa: E712
    if eligible.empty:
        return {
            "summary": [],
            "conditional": [],
            "exclusions": features.get("exclusion_reason", pd.Series(dtype=str))
            .value_counts()
            .to_dict(),
        }
    eligible["gap_direction"] = np.select(
        [eligible["primary_gap"] > 0.0005, eligible["primary_gap"] < -0.0005],
        ["positive", "negative"],
        default="flat",
    )
    eligible["gap_bucket"] = eligible["primary_gap"].map(
        lambda value: _gap_bucket(float(value), definition.gap_buckets)
    )
    eligible["weekday"] = pd.to_datetime(eligible["session"]).dt.day_name()
    eligible["year"] = pd.to_datetime(eligible["session"]).dt.year
    eligible["prior_direction"] = np.where(
        eligible["previous_day_return"].fillna(0) >= 0, "up_or_flat", "down"
    )
    eligible["previous_range_bucket"] = pd.cut(
        eligible["previous_day_range"],
        bins=[-np.inf, 0.01, 0.02, 0.04, np.inf],
        labels=["lt_1pct", "1_to_2pct", "2_to_4pct", "ge_4pct"],
    )
    eligible["above_ma20"] = eligible["official_open"] >= eligible["ma20"]
    eligible["above_ma50"] = eligible["official_open"] >= eligible["ma50"]
    eligible["event_day"] = eligible["event_flags"].map(bool)
    outcomes = (
        "retraced_25",
        "retraced_50",
        "retraced_75",
        "retraced_100",
        "at_or_below_open",
        "half_gap_fill",
        "full_gap_fill",
        "subsequent_new_high",
        "afternoon_continuation",
        "full_opening_fade",
    )
    summary = [_frequency_record(outcome, eligible[outcome]) for outcome in outcomes]
    conditional: list[dict[str, Any]] = []
    group_columns = (
        "gap_direction",
        "gap_bucket",
        "symbol",
        "year",
        "regime",
        "weekday",
        "prior_direction",
        "previous_range_bucket",
        "above_ma20",
        "above_ma50",
        "event_day",
        "earnings_flag",
    )
    for group_column in group_columns:
        for group_value, group in eligible.groupby(group_column, dropna=False):
            for outcome in outcomes:
                record = _frequency_record(outcome, group[outcome])
                record.update({"condition": group_column, "value": str(group_value)})
                conditional.append(record)
    high_windows: Counter[str] = Counter()
    for value in eligible["morning_high_time_et"]:
        if int(eligible["interval_minutes"].iloc[0]) == 15:
            high_windows["09:30-09:45_AMBIGUOUS_15M"] += int(
                pd.Timestamp(value).time() < time(9, 45)
            )
            high_windows["09:45-10:00_AMBIGUOUS_15M"] += int(
                time(9, 45) <= pd.Timestamp(value).time() < time(10)
            )
            high_windows["after_10:00"] += int(pd.Timestamp(value).time() >= time(10))
            continue
        clock = pd.Timestamp(value).time()
        if clock < time(9, 35):
            high_windows["09:30-09:35"] += 1
        elif clock < time(9, 45):
            high_windows["09:35-09:45"] += 1
        elif clock < time(10):
            high_windows["09:45-10:00"] += 1
        else:
            high_windows["after_10:00"] += 1
    extreme_exclusions = {}
    for percent in (0.01, 0.02, 0.05):
        if len(eligible) < 20:
            extreme_exclusions[str(percent)] = {"sessions": len(eligible), "status": "INSUFFICIENT"}
            continue
        cutoff = eligible["opening_spike_return"].abs().quantile(1.0 - percent)
        subset = eligible.loc[eligible["opening_spike_return"].abs() <= cutoff]
        extreme_exclusions[str(percent)] = _frequency_record("retraced_50", subset["retraced_50"])
    time_to_fade: dict[str, dict[str, Any]] = {}
    for fraction in (25, 50, 75, 100):
        column = f"time_to_retrace_{fraction}_minutes"
        observed = eligible[column].dropna().astype(float)
        time_to_fade[str(fraction)] = {
            "count": len(observed),
            "median_minutes_from_open": float(observed.median()) if len(observed) else None,
            "p25_minutes_from_open": float(observed.quantile(0.25)) if len(observed) else None,
            "p75_minutes_from_open": float(observed.quantile(0.75)) if len(observed) else None,
        }
    per_session_columns = [
        "symbol",
        "session",
        "gap_source",
        "primary_gap",
        "opening_spike_return",
        "opening_spike_atr",
        "retracement_fraction",
        *outcomes,
        *(f"time_to_retrace_{value}_minutes" for value in (25, 50, 75, 100)),
    ]
    return {
        "total_eligible_sessions": len(eligible),
        "positive_gap_sessions": int((eligible["primary_gap"] > 0).sum()),
        "qualifying_positive_gap_sessions": int(
            (eligible["primary_gap"] >= definition.positive_gap_min).sum()
        ),
        "opening_spike_sessions": int(
            (eligible["opening_spike_atr"] >= definition.spike_atr_min).fillna(False).sum()
        ),
        "primary_full_fade_sessions": int(eligible["full_opening_fade"].sum()),
        "summary": summary,
        "conditional": conditional,
        "morning_high_windows": dict(high_windows),
        "time_to_fade": time_to_fade,
        "per_session": cast(pd.DataFrame, eligible[per_session_columns]).to_dict(orient="records"),
        "extreme_session_exclusions": extreme_exclusions,
        "exclusions": features.loc[~features["eligible"].astype(bool), "exclusion_reason"]
        .value_counts()
        .to_dict(),
    }


def _cost_model(config: CostConfig, scenario: CostScenario) -> CostModel:
    return CostModel(
        scenario=scenario,
        commission_bps=config.commission_bps,
        half_spread_bps=config.half_spread_bps,
        base_slippage_bps=config.base_slippage_bps,
        impact_coeff_bps=config.impact_coeff_bps,
        short_borrow_annualized=config.short_borrow_annualized,
        sec_fee_bps=config.sec_fee_bps,
    )


def simulate_intraday_trade(
    bars: pd.DataFrame,
    *,
    signal_position: int,
    direction: str,
    stop: float,
    target: float,
    exit_time: time,
    cost_model: CostModel,
    participation: float,
) -> dict[str, Any]:
    """Next-bar entry with stop-first ambiguity and gap-through-stop fills."""
    entry_position, entry = next_bar_entry(bars, signal_position)
    side = Side.SHORT if direction == "SHORT" else Side.LONG
    entry_bar = bars.iloc[entry_position]
    entry_time = cast(pd.Timestamp, entry_bar["timestamp_et"])
    exit_price = float(entry_bar["close"])
    exit_timestamp = cast(pd.Timestamp, entry_bar["timestamp_et"])
    exit_reason = "time_exit"
    extrema: list[tuple[float, float]] = []
    considered = bars.iloc[entry_position:]
    for raw_bar in considered.itertuples(index=False):
        bar = cast(Any, raw_bar)
        bar_clock = bar.timestamp_et.time()
        if bar_clock >= exit_time:
            exit_price = float(bar.open)
            exit_timestamp = bar.timestamp_et
            exit_reason = "time_exit"
            break
        extrema.append((float(bar.low), float(bar.high)))
        if direction == "SHORT":
            # Stop first when both levels occur inside the same OHLC bar.
            if float(bar.open) >= stop:
                exit_price = float(bar.open)
                exit_timestamp = bar.timestamp_et
                exit_reason = "gap_through_stop"
                break
            if float(bar.high) >= stop:
                exit_price = stop
                exit_timestamp = bar.timestamp_et
                exit_reason = "stop"
                break
            if float(bar.low) <= target:
                exit_price = min(float(bar.open), target) if float(bar.open) <= target else target
                exit_timestamp = bar.timestamp_et
                exit_reason = "target"
                break
        else:
            if float(bar.open) <= stop:
                exit_price = float(bar.open)
                exit_timestamp = bar.timestamp_et
                exit_reason = "gap_through_stop"
                break
            if float(bar.low) <= stop:
                exit_price = stop
                exit_timestamp = bar.timestamp_et
                exit_reason = "stop"
                break
            if float(bar.high) >= target:
                exit_price = max(float(bar.open), target) if float(bar.open) >= target else target
                exit_timestamp = bar.timestamp_et
                exit_reason = "target"
                break
        exit_price = float(bar.close)
        exit_timestamp = bar.timestamp_et
    if direction == "SHORT":
        gross = entry / exit_price - 1.0
        mfe = max((entry - low) / entry for low, _ in extrema) if extrema else 0.0
        mae = min((entry - high) / entry for _, high in extrema) if extrema else 0.0
    else:
        gross = exit_price / entry - 1.0
        mfe = max((high - entry) / entry for _, high in extrema) if extrema else 0.0
        mae = min((low - entry) / entry for low, _ in extrema) if extrema else 0.0
    holding_minutes = max(0, int((exit_timestamp - entry_time).total_seconds() // 60))
    # Intraday borrow accrual is conservatively charged as one session.
    cost = cost_model.roundtrip_cost(side, holding_sessions=1, participation=participation)
    return {
        "entry_position": entry_position,
        "entry": entry,
        "entry_time_et": entry_time,
        "exit": exit_price,
        "exit_time_et": exit_timestamp,
        "exit_reason": exit_reason,
        "gross_return": gross,
        "cost_return": cost,
        "net_return": gross - cost,
        "mfe": mfe,
        "mae": mae,
        "holding_minutes": holding_minutes,
    }


def _first_position_at_or_after(bars: pd.DataFrame, value: time) -> int | None:
    matches = np.flatnonzero(bars["timestamp_et"].dt.time.to_numpy() >= value)
    return int(matches[0]) if len(matches) else None


def _candidate_signal(
    candidate: CandidateConfig,
    feature: pd.Series,
    bars: pd.DataFrame,
    *,
    cross_frames: dict[str, pd.DataFrame],
    seed: int,
) -> tuple[int | None, float | None, float | None, str | None]:
    """Return completed signal bar, stop, target, or a rejection reason."""
    params = candidate.params
    interval = int(feature["interval_minutes"])
    opening_minutes = int(params.get("opening_minutes", 15))
    opening_bars = math.ceil(opening_minutes / interval)
    if len(bars) <= opening_bars:
        return None, None, None, "INCOMPLETE_OPENING_RANGE"
    instrument = params.get("instrument")
    if instrument and str(feature["symbol"]) != str(instrument):
        return None, None, None, "INSTRUMENT_FILTER"
    gap = float(feature["primary_gap"])
    gap_min = float(params.get("gap_min", 0.002))
    spike_atr_min = float(params.get("spike_atr_min", 0.15))
    spike_atr = feature.get("opening_spike_atr")
    spike_ok = (
        spike_atr is not None and np.isfinite(spike_atr) and float(spike_atr) >= spike_atr_min
    )
    candidate_opening = opening_range(bars, minutes=opening_minutes, interval_minutes=interval)
    if not candidate_opening["complete"]:
        return None, None, None, "INCOMPLETE_CANDIDATE_OPENING_RANGE"
    opening_high = float(candidate_opening["high"])
    opening_low = float(candidate_opening["low"])
    official_open = float(feature["official_open"])
    previous_close = float(feature["previous_close"])
    rule = candidate.rule
    signal: int | None = None

    if rule in {"confirmed_vwap_failure", "confirmed_opening_low", "gap_fill_confirmed"}:
        if gap < gap_min or not spike_ok:
            return None, None, None, "GAP_OR_SPIKE_FILTER"
        for pos in range(opening_bars, len(bars)):
            bar = bars.iloc[pos]
            if rule in {"confirmed_vwap_failure", "gap_fill_confirmed"} and float(
                bar["close"]
            ) < float(bar["vwap"]):
                signal = pos
                break
            if rule == "confirmed_opening_low" and float(bar["close"]) < opening_low:
                signal = pos
                break
    elif rule == "lower_high_break":
        if gap < gap_min or not spike_ok:
            return None, None, None, "GAP_OR_SPIKE_FILTER"
        for pos in range(opening_bars + 1, len(bars)):
            prior = bars.iloc[pos - 1]
            current = bars.iloc[pos]
            if float(prior["high"]) < opening_high and float(current["close"]) < float(
                prior["low"]
            ):
                signal = pos
                break
    elif rule == "false_breakout":
        population = str(params.get("gap_population", "ALL"))
        if population == "POSITIVE" and gap < gap_min:
            return None, None, None, "POSITIVE_GAP_POPULATION_FILTER"
        if population == "NONPOSITIVE" and gap >= gap_min:
            return None, None, None, "NONPOSITIVE_GAP_POPULATION_FILTER"
        signal = detect_false_breakout(
            bars,
            opening_range_bars=opening_bars,
            buffer_bps=float(params.get("buffer_bps", 0.0)),
        )
    elif rule == "overnight_rejection":
        premarket_gap = feature.get("premarket_gap")
        if (
            premarket_gap is None
            or not np.isfinite(premarket_gap)
            or float(premarket_gap) < gap_min
        ):
            return None, None, None, "POSITIVE_PREMARKET_GAP_FILTER"
        overshoot_bps = float(params.get("overshoot_bps", 0.0))
        if "overshoot_atr_fraction" in params:
            daily_atr = feature.get("daily_atr20")
            overnight_high = feature.get("overnight_high")
            if (
                daily_atr is None
                or overnight_high is None
                or not np.isfinite(daily_atr)
                or float(overnight_high) <= 0
            ):
                return None, None, None, "MISSING_ATR_NORMALIZED_OVERSHOOT_INPUT"
            overshoot_bps = (
                float(daily_atr)
                * float(params["overshoot_atr_fraction"])
                / float(overnight_high)
                * 10_000.0
            )
        signal = detect_overnight_rejection(
            bars,
            overnight_high=feature.get("overnight_high"),
            overshoot_bps=overshoot_bps,
        )
    elif rule == "naive_positive_gap":
        if gap < gap_min:
            return None, None, None, "GAP_FILTER"
        at = _clock(str(params.get("signal_time", "09:45")))
        signal = _first_position_at_or_after(bars, at)
    elif rule == "unconditional_short":
        signal = _first_position_at_or_after(bars, _clock(str(params.get("signal_time", "09:45"))))
    elif rule == "random_time_matched":
        if gap < gap_min:
            return None, None, None, "GAP_FILTER"
        choices = list(range(opening_bars, max(opening_bars, len(bars) - 2)))
        if choices:
            token = f"{seed}:{feature['symbol']}:{feature['session']}:{candidate.candidate_id}"
            signal = choices[int(hashlib.sha256(token.encode()).hexdigest(), 16) % len(choices)]
    elif rule == "long_mirror":
        if gap > -gap_min:
            return None, None, None, "NEGATIVE_GAP_FILTER"
        opening_selloff = (official_open - opening_low) / max(
            float(feature.get("daily_atr20") or 0), 1e-12
        )
        if opening_selloff < spike_atr_min:
            return None, None, None, "SELLOFF_FILTER"
        for pos in range(opening_bars, len(bars)):
            if float(bars.iloc[pos]["close"]) > float(bars.iloc[pos]["vwap"]):
                signal = pos
                break
    elif rule == "continuation":
        if gap < gap_min or not spike_ok:
            return None, None, None, "GAP_OR_SPIKE_FILTER"
        for pos in range(opening_bars, len(bars)):
            bar = bars.iloc[pos]
            if float(bar["close"]) > opening_high and float(bar["close"]) > float(bar["vwap"]):
                signal = pos
                break
    elif rule == "opening_range_breakout":
        buffer = 1.0 + float(params.get("buffer_bps", 0.0)) / 10_000.0
        for pos in range(opening_bars, len(bars)):
            bar = bars.iloc[pos]
            if float(bar["close"]) > opening_high * buffer and float(bar["close"]) > float(
                bar["vwap"]
            ):
                signal = pos
                break
    elif rule == "vwap_reclaim":
        traded_below = False
        for pos in range(opening_bars, len(bars)):
            bar = bars.iloc[pos]
            close = float(bar["close"])
            vwap = float(bar["vwap"])
            traded_below = traded_below or close < vwap
            if traded_below and close > vwap:
                signal = pos
                break
    elif rule == "cross_index_vwap":
        if gap < gap_min or not spike_ok:
            return None, None, None, "GAP_OR_SPIKE_FILTER"
        peer = str(params.get("peer", "SPY"))
        peer_frame = cross_frames.get(peer)
        if peer_frame is None:
            return None, None, None, "MISSING_CROSS_SERIES"
        aligned = align_cross_index({str(feature["symbol"]): bars, peer: peer_frame})
        for timestamp, row in aligned.iterrows():
            own = str(feature["symbol"])
            if (
                timestamp >= bars.iloc[opening_bars]["timestamp_utc"]
                and float(row[f"{own}_close"]) < float(row[f"{own}_vwap"])
                and float(row[f"{peer}_close"]) < float(row[f"{peer}_vwap"])
            ):
                positions = np.flatnonzero(bars["timestamp_utc"].to_numpy() == timestamp)
                signal = int(positions[0]) if len(positions) else None
                break
    elif rule == "cross_index_divergence":
        if gap < gap_min or not spike_ok:
            return None, None, None, "GAP_OR_SPIKE_FILTER"
        peer = str(params.get("peer", "SPY"))
        peer_frame = cross_frames.get(peer)
        if peer_frame is None:
            return None, None, None, "MISSING_CROSS_SERIES"
        aligned = align_cross_index({str(feature["symbol"]): bars, peer: peer_frame})
        for timestamp, row in aligned.iterrows():
            own = str(feature["symbol"])
            if (
                timestamp >= bars.iloc[opening_bars]["timestamp_utc"]
                and float(row[f"{own}_close"]) < float(row[f"{own}_vwap"])
                and float(row[f"{peer}_close"]) >= float(row[f"{peer}_vwap"])
            ):
                positions = np.flatnonzero(bars["timestamp_utc"].to_numpy() == timestamp)
                signal = int(positions[0]) if len(positions) else None
                break
    elif rule in {"regime_calm", "regime_stressed"}:
        wanted = "CALM" if rule == "regime_calm" else "STRESSED"
        if feature.get("regime") != wanted:
            return None, None, None, "REGIME_FILTER"
        base = candidate.model_copy(update={"rule": "confirmed_vwap_failure"})
        return _candidate_signal(base, feature, bars, cross_frames=cross_frames, seed=seed)
    elif rule == "feature_filter":
        field = str(params["feature"])
        value = feature.get(field)
        if value is None:
            return None, None, None, f"MISSING_FEATURE:{field}"
        operator = str(params.get("operator", "eq"))
        expected = params.get("value")
        if operator != "eq" and not np.isfinite(_to_float(value)):
            return None, None, None, f"MISSING_FEATURE:{field}"
        passed = (
            value == expected
            if operator == "eq"
            else _to_float(value) >= _to_float(expected)
            if operator == "ge"
            else _to_float(value) <= _to_float(expected)
            if operator == "le"
            else float(params["low"]) <= float(value) < float(params["high"])
            if operator == "between"
            else False
        )
        if not passed:
            return None, None, None, f"FEATURE_FILTER:{field}"
        base = candidate.model_copy(update={"rule": "confirmed_vwap_failure"})
        return _candidate_signal(base, feature, bars, cross_frames=cross_frames, seed=seed)
    elif rule in {"event_population", "event_exclude", "event_wait"}:
        has_event = bool(feature.get("event_flags"))
        if rule in {"event_population", "event_wait"} and not has_event:
            return None, None, None, "NO_POINT_IN_TIME_EVENT_FLAG"
        if rule == "event_exclude" and has_event:
            return None, None, None, "SCHEDULED_EVENT_EXCLUDED"
        base = candidate.model_copy(update={"rule": "confirmed_vwap_failure"})
        result = _candidate_signal(base, feature, bars, cross_frames=cross_frames, seed=seed)
        if rule != "event_wait" or result[0] is None:
            return result
        wait_until = _clock(str(params.get("wait_until", "10:30")))
        waited = _first_position_at_or_after(bars, wait_until)
        if waited is None:
            return None, None, None, "EVENT_WAIT_WINDOW_UNAVAILABLE"
        signal = max(result[0], waited)
        if signal + 1 >= len(bars):
            return None, None, None, "NO_NEXT_BAR_AFTER_EVENT_WAIT"
    elif rule == "no_trade":
        return None, None, None, "CONTROL_NO_TRADE"
    else:
        return None, None, None, "UNSUPPORTED_RULE"

    if signal is None:
        return None, None, None, "NO_SIGNAL"
    if signal + 1 >= len(bars):
        return None, None, None, "NO_NEXT_BAR"
    direction = candidate.direction
    buffer_bps = float(params.get("stop_buffer_bps", 5.0))
    if direction == "SHORT":
        stop = max(opening_high, float(bars.iloc[: signal + 1]["high"].max())) * (
            1.0 + buffer_bps / 10_000.0
        )
        target_kind = str(params.get("target", "OPEN"))
        target = {
            "VWAP": float(bars.iloc[signal]["vwap"]),
            "OPEN": official_open,
            "HALF_SPIKE": official_open + (opening_high - official_open) * 0.5,
            "PREVIOUS_CLOSE": previous_close,
            "OR_MID": (opening_high + opening_low) * 0.5,
            "OR_LOW": opening_low,
            "HALF_GAP": previous_close + (official_open - previous_close) * 0.5,
            "FULL_GAP": previous_close,
        }.get(target_kind, official_open)
        entry = float(bars.iloc[signal + 1]["open"])
        if target >= entry:
            target = entry - max(
                (stop - entry) * float(params.get("r_multiple", 1.0)), entry * 0.0001
            )
    else:
        stop = min(opening_low, float(bars.iloc[: signal + 1]["low"].min())) * (
            1.0 - buffer_bps / 10_000.0
        )
        target_kind = str(params.get("target", "OPEN"))
        target = {
            "VWAP": float(bars.iloc[signal]["vwap"]),
            "OPEN": official_open,
            "OR_MID": (opening_high + opening_low) * 0.5,
            "OR_HIGH": opening_high,
        }.get(target_kind, official_open)
        entry = float(bars.iloc[signal + 1]["open"])
        if target <= entry:
            target = entry + max(
                (entry - stop) * float(params.get("r_multiple", 1.0)), entry * 0.0001
            )
    return signal, stop, target, None


def _requirements_available(
    candidate: CandidateConfig, audit: dict[str, Any]
) -> tuple[bool, str | None]:
    checks = {
        "premarket": bool(audit["premarket_available"]),
        "event_calendar": bool(audit["economic_calendar"]["available"]),
        "breadth": bool(audit["breadth"]["research_usable"]),
        "futures": bool(audit["index_futures"]),
        "bid_ask": audit["bid_ask"] != "not available historically",
    }
    missing = [
        requirement for requirement in candidate.requires if not checks.get(requirement, False)
    ]
    return (not missing, f"MISSING_REQUIRED_DATA:{','.join(missing)}" if missing else None)


def _signal_diagnostics(
    bars: pd.DataFrame,
    cross_frames: dict[str, pd.DataFrame],
    signal_position: int,
) -> dict[str, Any]:
    """Causal diagnostics using bars available when the signal completes."""
    observed = bars.iloc[: signal_position + 1]
    signal = observed.iloc[-1]
    previous = observed["close"].shift(1)
    true_range = pd.concat(
        [
            observed["high"] - observed["low"],
            (observed["high"] - previous).abs(),
            (observed["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    signal_utc = cast(pd.Timestamp, signal["timestamp_utc"])
    states: dict[str, str] = {
        str(signal["symbol"]): (
            "BELOW_VWAP" if float(signal["close"]) < float(signal["vwap"]) else "ABOVE_VWAP"
        )
    }
    for symbol, frame in cross_frames.items():
        aligned = frame.loc[frame["timestamp_utc"] <= signal_utc]
        if aligned.empty:
            states[symbol] = "UNAVAILABLE"
            continue
        peer = aligned.iloc[-1]
        states[symbol] = (
            "BELOW_VWAP" if float(peer["close"]) < float(peer["vwap"]) else "ABOVE_VWAP"
        )
    return {
        "signal_vwap": float(signal["vwap"]),
        "signal_distance_from_vwap": float(signal["close"] / signal["vwap"] - 1.0),
        "intraday_atr_at_signal": float(true_range.tail(14).mean()),
        "intraday_realized_range_at_signal": float(
            observed["high"].max() / observed["low"].min() - 1.0
        ),
        "cross_index_confirmation_state": json.dumps(states, sort_keys=True),
    }


def generate_candidate_trades(
    cfg: OpeningFadeConfig,
    features: pd.DataFrame,
    session_bars: dict[tuple[str, pd.Timestamp], pd.DataFrame],
    audit: dict[str, Any],
    *,
    scenario: CostScenario,
    entry_delay_bars: int = 0,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    eligible = features.loc[features.get("eligible", False) == True].copy()  # noqa: E712
    model = _cost_model(cfg.costs, scenario)
    requested_participation = cfg.costs.assumed_participation
    effective_participation = min(requested_participation, cfg.costs.maximum_bar_participation)
    fill_fraction = (
        min(1.0, cfg.costs.maximum_bar_participation / requested_participation)
        if requested_participation > 0
        else 1.0
    )
    records: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for candidate in cfg.candidates:
        requirements_ok, unsupported_reason = _requirements_available(candidate, audit)
        candidate_records = 0
        rejection_counts: Counter[str] = Counter()
        if not requirements_ok:
            ledger.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "status": "UNSUPPORTED",
                    "reason": unsupported_reason,
                    "trades": 0,
                    "params": candidate.params,
                }
            )
            continue
        for _, feature in eligible.iterrows():
            key = (str(feature["symbol"]), pd.Timestamp(feature["session"]))
            bars = session_bars[key]
            cross_frames = {
                symbol: value
                for (symbol, session), value in session_bars.items()
                if session == key[1] and symbol != key[0]
            }
            signal, stop, target, rejection = _candidate_signal(
                candidate,
                feature,
                bars,
                cross_frames=cross_frames,
                seed=cfg.validation.random_seed,
            )
            if rejection:
                rejection_counts[rejection] += 1
                continue
            assert signal is not None and stop is not None and target is not None
            execution_signal = signal + entry_delay_bars
            if execution_signal >= len(bars) - 1:
                rejection_counts["NO_BAR_AFTER_DELAYED_SIGNAL"] += 1
                continue
            simulated = simulate_intraday_trade(
                bars,
                signal_position=execution_signal,
                direction=candidate.direction,
                stop=stop,
                target=target,
                exit_time=_clock(str(candidate.params.get("exit_time", "11:00"))),
                cost_model=model,
                participation=effective_participation,
            )
            signal_et = bars.iloc[signal]["timestamp_et"]
            signal_utc = bars.iloc[signal]["timestamp_utc"]
            diagnostics = _signal_diagnostics(bars, cross_frames, signal)
            signal_available_et = signal_et + pd.Timedelta(minutes=int(feature["interval_minutes"]))
            entry_et = simulated["entry_time_et"]
            exit_et = simulated["exit_time_et"]
            records.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "symbol": feature["symbol"],
                    "session": pd.Timestamp(feature["session"]),
                    "direction": candidate.direction,
                    "signal_time_et": signal_et,
                    "signal_time_utc": signal_utc,
                    "signal_available_time_et": signal_available_et,
                    "signal_available_time_utc": signal_available_et.tz_convert("UTC"),
                    "entry_time_et": entry_et,
                    "entry_time_utc": entry_et.tz_convert("UTC"),
                    "exit_time_et": exit_et,
                    "exit_time_utc": exit_et.tz_convert("UTC"),
                    "entry": simulated["entry"],
                    "stop": stop,
                    "target": target,
                    "exit": simulated["exit"],
                    "exit_reason": simulated["exit_reason"],
                    "gross_return": simulated["gross_return"],
                    "net_return": simulated["net_return"],
                    "cost_return": simulated["cost_return"],
                    "mfe": simulated["mfe"],
                    "mae": simulated["mae"],
                    "holding_minutes": simulated["holding_minutes"],
                    "previous_close": feature["previous_close"],
                    "premarket_last": feature["premarket_last"],
                    "official_open": feature["official_open"],
                    "gap_source": feature["gap_source"],
                    "gap_pct": feature["primary_gap"],
                    "gap_atr": feature["gap_atr"],
                    "overnight_high": feature["overnight_high"],
                    "overnight_low": feature["overnight_low"],
                    "overnight_range": feature["overnight_range"],
                    "opening_range_high": feature["opening_range_high"],
                    "opening_range_low": feature["opening_range_low"],
                    "opening_range_width": feature["opening_range_width"],
                    "opening_spike_return": feature["opening_spike_return"],
                    "opening_spike_atr": feature["opening_spike_atr"],
                    "opening_volume_relative_to_prior_median": feature[
                        "opening_volume_relative_to_prior_median"
                    ],
                    "daily_atr20": feature["daily_atr20"],
                    "previous_day_return": feature["previous_day_return"],
                    "previous_day_range": feature["previous_day_range"],
                    "short_trend_state": feature["short_trend_state"],
                    "medium_trend_state": feature["medium_trend_state"],
                    "recent_drawdown60": feature["recent_drawdown60"],
                    "regime": feature["regime"],
                    "event_flags": json.dumps(feature["event_flags"]),
                    "earnings_flag": feature["earnings_flag"],
                    **diagnostics,
                    "cost_scenario": scenario.value,
                    "modeled_commission_bps": model.commission_bps,
                    "modeled_half_spread_bps_per_leg": (model.half_spread_bps * model.multiplier),
                    "modeled_slippage_bps_per_leg": (model.base_slippage_bps * model.multiplier),
                    "modeled_impact_bps_per_leg": (
                        model.impact_coeff_bps * effective_participation * model.multiplier
                    ),
                    "requested_participation": requested_participation,
                    "effective_participation": effective_participation,
                    "maximum_bar_participation": cfg.costs.maximum_bar_participation,
                    "fill_fraction": fill_fraction,
                    "fill_assumption": (
                        "FULL_AT_CONFIGURED_PARTICIPATION"
                        if fill_fraction >= 1.0
                        else "PARTIAL_AT_MAXIMUM_BAR_PARTICIPATION"
                    ),
                    "additional_entry_delay_bars": entry_delay_bars,
                    "warnings": json.dumps(
                        [
                            "bar-sequence ambiguity handled stop-first",
                            "OHLCV spread/slippage is modeled, not observed",
                        ]
                    ),
                }
            )
            candidate_records += 1
        ledger.append(
            {
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "status": "EVALUATED" if candidate_records else "NO_TRADES",
                "reason": None if candidate_records else "NO_QUALIFYING_SIGNALS",
                "trades": candidate_records,
                "rejections": dict(rejection_counts),
                "params": candidate.params,
            }
        )
    columns = list(SimulatedTrade.__dataclass_fields__)
    # Timestamps are included in records but the stable output schema is wider
    # than SimulatedTrade for convenient parquet analysis.
    trades = pd.DataFrame(records)
    if trades.empty:
        trades = pd.DataFrame(columns=columns)
    return trades, ledger


def _longest_losing_streak(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def trade_metrics(
    trades: pd.DataFrame, *, seed: int, n_boot: int, block_length: int
) -> dict[str, Any]:
    if trades.empty:
        return {
            "trade_count": 0,
            "session_count": 0,
            "status": "NO_TRADES",
            "cagr": None,
            "cagr_status": "NOT_MEANINGFUL",
            "mean_net_return_ci95": [None, None],
            "sharpe_ci95": [None, None],
        }
    returns = trades["net_return"].astype(float).to_numpy()
    gross = trades["gross_return"].astype(float).to_numpy()
    sessions = aggregate_session_returns(trades)
    daily = sessions.to_numpy()
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    result: dict[str, Any] = {
        "status": "CALCULATED" if len(daily) >= 2 else "INSUFFICIENT",
        "trade_count": len(trades),
        "session_count": len(daily),
        "effective_sample_size": effective_sample_size(daily)
        if len(daily) >= 2
        else float(len(daily)),
        "hit_rate": float((returns > 0).mean()),
        "average_winner": float(winners.mean()) if len(winners) else None,
        "average_loser": float(losers.mean()) if len(losers) else None,
        "payoff_ratio": (
            float(winners.mean() / abs(losers.mean())) if len(winners) and len(losers) else None
        ),
        "expectancy": float(returns.mean()),
        "gross_expectancy": float(gross.mean()),
        "profit_factor": profit_factor(returns) if len(returns) >= 2 else None,
        "cumulative_return": float(np.prod(1.0 + daily) - 1.0),
        "cagr": (
            float(np.prod(1.0 + daily) ** (252.0 / len(daily)) - 1.0)
            if len(daily) >= 252 and np.prod(1.0 + daily) > 0
            else None
        ),
        "cagr_status": "CALCULATED" if len(daily) >= 252 else "NOT_MEANINGFUL",
        "average_mfe": float(trades["mfe"].mean()),
        "average_mae": float(trades["mae"].mean()),
        "average_holding_minutes": float(trades["holding_minutes"].mean()),
        "longest_losing_streak": _longest_losing_streak(returns),
        "turnover": float(len(trades) * 2),
        "total_cost_return": float(trades["cost_return"].sum()),
        "gross_net_degradation": float(gross.mean() - returns.mean()),
        "cost_fraction_of_absolute_gross_expectancy": (
            float((gross.mean() - returns.mean()) / abs(gross.mean()))
            if abs(gross.mean()) > 1e-12
            else None
        ),
        "fill_statuses": trades.get("fill_assumption", pd.Series(dtype=str))
        .value_counts()
        .to_dict(),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
        "by_instrument": trades.groupby("symbol")["net_return"]
        .agg(["count", "mean"])
        .to_dict(orient="index"),
        "by_year": trades.assign(year=pd.to_datetime(trades["session"]).dt.year)
        .groupby("year")["net_return"]
        .agg(["count", "mean"])
        .to_dict(orient="index"),
        "by_regime": trades.groupby("regime")["net_return"]
        .agg(["count", "mean"])
        .to_dict(orient="index"),
    }
    if len(daily) >= 2:
        result.update(
            {
                "annualized_volatility": float(np.std(daily, ddof=1) * np.sqrt(252)),
                "sharpe": sharpe_ratio(daily),
                "sortino": sortino_ratio(daily),
                "maximum_drawdown": max_drawdown(daily),
            }
        )
        result["calmar"] = (
            float((np.mean(daily) * 252) / abs(result["maximum_drawdown"]))
            if result["maximum_drawdown"] < 0
            else None
        )
        value_at_risk, expected_shortfall = var_es(daily)
        result["var_95"] = value_at_risk
        result["expected_shortfall_95"] = expected_shortfall
    if len(daily) >= 3:
        rng = np.random.default_rng(seed)
        result["mean_net_return_ci95"] = list(
            block_bootstrap_ci(
                daily,
                rng=rng,
                n_boot=n_boot,
                block_length=block_length,
            )
        )
        result["pvalue_mean_positive"] = mean_pvalue_bootstrap(
            daily,
            rng=np.random.default_rng(seed + 1),
            n_boot=n_boot,
        )
        result["sharpe_ci95"] = list(
            block_bootstrap_ci(
                daily,
                rng=np.random.default_rng(seed + 2),
                n_boot=n_boot,
                block_length=block_length,
                stat=lambda values: sharpe_ratio(values),
            )
        )
    else:
        result["mean_net_return_ci95"] = [None, None]
        result["pvalue_mean_positive"] = 1.0
        result["sharpe_ci95"] = [None, None]
    return result


def _walk_forward_folds(
    sessions: pd.DatetimeIndex, config: ValidationConfig
) -> list[dict[str, Any]]:
    unique = pd.DatetimeIndex(sorted(pd.to_datetime(sessions).unique()))
    needed = config.train_sessions + config.n_folds * config.test_sessions
    if len(unique) < needed:
        return []
    first_test = len(unique) - config.n_folds * config.test_sessions
    folds = []
    for fold_id in range(config.n_folds):
        test_start_pos = first_test + fold_id * config.test_sessions
        test_end_pos = test_start_pos + config.test_sessions
        train_end_pos = max(0, test_start_pos - config.embargo_sessions)
        train = unique[:train_end_pos]
        test = unique[test_start_pos:test_end_pos]
        if len(train) < config.train_sessions or len(test) < config.test_sessions:
            continue
        folds.append(
            {
                "fold_id": fold_id,
                "train_start": train[0],
                "train_end": train[-1],
                "test_start": test[0],
                "test_end": test[-1],
            }
        )
    return folds


def _extreme_return_robustness(trades: pd.DataFrame) -> dict[str, Any]:
    daily = aggregate_session_returns(trades)
    rows = []
    for fraction in (0.01, 0.02, 0.05):
        remove = max(1, math.ceil(len(daily) * fraction)) if len(daily) else 0
        if len(daily) - remove < 20:
            rows.append(
                {
                    "excluded_fraction": fraction,
                    "status": "INSUFFICIENT",
                    "remaining_sessions": max(0, len(daily) - remove),
                    "mean_net_return": None,
                }
            )
            continue
        kept = daily.loc[daily.abs().sort_values().index[:-remove]]
        rows.append(
            {
                "excluded_fraction": fraction,
                "status": "CALCULATED",
                "remaining_sessions": len(kept),
                "mean_net_return": float(kept.mean()),
            }
        )
    return {
        "rows": rows,
        "survives": bool(rows)
        and all(
            item["status"] == "CALCULATED" and cast(float, item["mean_net_return"]) > 0
            for item in rows
        ),
    }


def validate_candidates(
    cfg: OpeningFadeConfig,
    scenario_trades: dict[str, pd.DataFrame],
    ledger: list[dict[str, Any]],
    all_sessions: pd.DatetimeIndex,
) -> dict[str, Any]:
    conservative = scenario_trades[CostScenario.CONSERVATIVE.value]
    folds = _walk_forward_folds(all_sessions, cfg.validation)
    candidate_results: list[dict[str, Any]] = []
    pvalues: list[float] = []
    for offset, candidate in enumerate(cfg.candidates):
        subset = conservative.loc[conservative["candidate_id"] == candidate.candidate_id]
        metrics = trade_metrics(
            subset,
            seed=cfg.validation.random_seed + offset * 10,
            n_boot=cfg.validation.bootstrap_samples,
            block_length=cfg.validation.block_length,
        )
        fold_results: list[dict[str, Any]] = []
        for fold in folds:
            test = subset.loc[
                (pd.to_datetime(subset["session"]) >= fold["test_start"])
                & (pd.to_datetime(subset["session"]) <= fold["test_end"])
            ]
            fold_daily = aggregate_session_returns(test)
            fold_results.append(
                {
                    "fold_id": fold["fold_id"],
                    "train_start": fold["train_start"].date().isoformat(),
                    "train_end": fold["train_end"].date().isoformat(),
                    "test_start": fold["test_start"].date().isoformat(),
                    "test_end": fold["test_end"].date().isoformat(),
                    "test_sessions": len(fold_daily),
                    "mean_net_return": float(fold_daily.mean()) if len(fold_daily) else None,
                }
            )
        positive_folds = [item for item in fold_results if item["mean_net_return"] is not None]
        positive_fold_fraction = (
            sum(float(item["mean_net_return"]) > 0 for item in positive_folds) / len(positive_folds)
            if positive_folds
            else 0.0
        )
        pvalue = float(metrics.get("pvalue_mean_positive", 1.0))
        pvalues.append(pvalue)
        daily_returns = aggregate_session_returns(subset).to_numpy(dtype=float)
        dsr = (
            deflated_sharpe_ratio(daily_returns, n_trials=len(cfg.candidates))
            if len(daily_returns) >= 3
            else 0.0
        )
        stress = scenario_trades[CostScenario.STRESS.value]
        stress_subset = stress.loc[stress["candidate_id"] == candidate.candidate_id]
        stress_mean = (
            float(aggregate_session_returns(stress_subset).mean())
            if not stress_subset.empty
            else None
        )
        delayed = scenario_trades.get("delayed_conservative", pd.DataFrame())
        delayed_subset = (
            delayed.loc[delayed["candidate_id"] == candidate.candidate_id]
            if not delayed.empty and "candidate_id" in delayed
            else pd.DataFrame()
        )
        delayed_mean = (
            float(aggregate_session_returns(delayed_subset).mean())
            if not delayed_subset.empty
            else None
        )
        instrument_counts = subset["symbol"].value_counts() if not subset.empty else pd.Series()
        instrument_means = (
            subset.groupby("symbol")["net_return"].mean().to_dict() if not subset.empty else {}
        )
        year_means = (
            subset.assign(year=pd.to_datetime(subset["session"]).dt.year)
            .groupby("year")["net_return"]
            .mean()
            .to_dict()
            if not subset.empty
            else {}
        )
        random_control = (
            conservative.loc[conservative["candidate_id"] == "H_RANDOM_MATCHED"]
            if "candidate_id" in conservative
            else pd.DataFrame()
        )
        continuation_control = (
            conservative.loc[conservative["candidate_id"] == "H_CONTINUATION"]
            if "candidate_id" in conservative
            else pd.DataFrame()
        )
        random_comparison = _paired_control_summary(
            subset,
            random_control,
            seed=cfg.validation.random_seed + offset + 1000,
            n_boot=cfg.validation.bootstrap_samples,
            block=cfg.validation.block_length,
        )
        continuation_comparison = _paired_control_summary(
            subset,
            continuation_control,
            seed=cfg.validation.random_seed + offset + 2000,
            n_boot=cfg.validation.bootstrap_samples,
            block=cfg.validation.block_length,
        )
        result: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "family": candidate.family,
            "direction": candidate.direction,
            "rule": candidate.rule,
            "metrics": metrics,
            "folds": fold_results,
            "walk_forward_status": "COMPLETE"
            if len(folds) == cfg.validation.n_folds
            else "INSUFFICIENT",
            "positive_fold_fraction": positive_fold_fraction,
            "deflated_sharpe_probability": dsr,
            "stress_expectancy": stress_mean,
            "delayed_entry_expectancy": delayed_mean,
            "extreme_return_robustness": _extreme_return_robustness(subset),
            "instrument_consistency": {
                "instrument_count": len(instrument_means),
                "means": instrument_means,
                "largest_trade_fraction": (
                    float(instrument_counts.iloc[0] / instrument_counts.sum())
                    if len(instrument_counts)
                    else None
                ),
            },
            "year_consistency": {"year_count": len(year_means), "means": year_means},
            "random_control_comparison": random_comparison,
            "continuation_control_comparison": continuation_comparison,
            "raw_pvalue": pvalue,
        }
        candidate_results.append(result)
    neighborhood = _parameter_neighborhoods(candidate_results)
    neighborhood_by_family = {item["family"]: item for item in neighborhood}
    qvalues = benjamini_hochberg(np.asarray(pvalues, dtype=float))
    promoted: list[str] = []
    for result, qvalue in zip(candidate_results, qvalues, strict=True):
        metrics = cast(dict[str, Any], result["metrics"])
        ci = metrics.get("mean_net_return_ci95", [None, None])
        failures: list[str] = []
        if result["walk_forward_status"] != "COMPLETE":
            failures.append("INSUFFICIENT_WALK_FORWARD_HISTORY")
        if _to_float(metrics.get("effective_sample_size", 0)) < (
            cfg.validation.minimum_effective_sample_size
        ):
            failures.append("INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE")
        if _to_float(metrics.get("expectancy", -math.inf)) <= 0:
            failures.append("NONPOSITIVE_CONSERVATIVE_EXPECTANCY")
        if result["stress_expectancy"] is None or _to_float(result["stress_expectancy"]) <= 0:
            failures.append("DID_NOT_SURVIVE_STRESS_COSTS")
        if _to_float(result["positive_fold_fraction"]) < (
            cfg.validation.minimum_positive_fold_fraction
        ):
            failures.append("FOLD_SIGN_INCONSISTENCY")
        if qvalue > cfg.validation.fdr_alpha:
            failures.append("MULTIPLE_TESTING_ADJUSTED_EVIDENCE_FAILED")
        if _to_float(result["deflated_sharpe_probability"]) < (
            cfg.validation.minimum_deflated_sharpe_probability
        ):
            failures.append("DEFLATED_SHARPE_FAILED")
        if ci[0] is None or ci[0] <= 0:
            failures.append("MEAN_RETURN_CI_INCLUDES_ZERO")
        maximum_drawdown = metrics.get("maximum_drawdown")
        if maximum_drawdown is None or _to_float(maximum_drawdown) < (
            cfg.validation.maximum_drawdown_floor
        ):
            failures.append("MAXIMUM_DRAWDOWN_GATE_FAILED")
        if (
            result["delayed_entry_expectancy"] is None
            or _to_float(result["delayed_entry_expectancy"]) <= 0
        ):
            failures.append("DELAYED_ENTRY_SURVIVAL_FAILED")
        if not cast(dict[str, Any], result["extreme_return_robustness"])["survives"]:
            failures.append("EXTREME_SESSION_ROBUSTNESS_FAILED")
        instrument_consistency = cast(dict[str, Any], result["instrument_consistency"])
        gated_instrument_means = cast(dict[str, float], instrument_consistency["means"])
        if (
            instrument_consistency["instrument_count"] < cfg.validation.minimum_instruments
            or _to_float(instrument_consistency.get("largest_trade_fraction", 1.0))
            > cfg.validation.maximum_single_instrument_trade_fraction
            or sum(value > 0 for value in gated_instrument_means.values())
            < math.ceil(max(1, len(gated_instrument_means)) / 2)
        ):
            failures.append("INSTRUMENT_CONCENTRATION_GATE_FAILED")
        year_consistency = cast(dict[str, Any], result["year_consistency"])
        gated_year_means = cast(dict[int, float], year_consistency["means"])
        if year_consistency["year_count"] < cfg.validation.minimum_years or sum(
            value > 0 for value in gated_year_means.values()
        ) < math.ceil(max(1, len(gated_year_means)) / 2):
            failures.append("SUBPERIOD_CONSISTENCY_GATE_FAILED")
        neighbor = neighborhood_by_family[result["family"]]
        if (
            neighbor["status"] != "CALCULATED"
            or _to_float(neighbor.get("positive_fraction", 0.0))
            < cfg.validation.minimum_neighborhood_positive_fraction
            or _to_float(neighbor.get("median_expectancy", -math.inf)) <= 0
        ):
            failures.append("PARAMETER_NEIGHBORHOOD_GATE_FAILED")
        random_comparison = cast(dict[str, Any], result["random_control_comparison"])
        random_ci = random_comparison.get("ci95", [None, None])
        if (
            random_comparison.get("status") != "CALCULATED"
            or random_ci[0] is None
            or (random_ci[0] <= 0)
        ):
            failures.append("TIME_MATCHED_CONTROL_GATE_FAILED")
        continuation_comparison = cast(dict[str, Any], result["continuation_control_comparison"])
        continuation_ci = continuation_comparison.get("ci95", [None, None])
        if result["family"] != "H" and (
            continuation_comparison.get("status") != "CALCULATED"
            or continuation_ci[0] is None
            or continuation_ci[0] <= 0
        ):
            failures.append("CONTINUATION_CONTROL_GATE_FAILED")
        if result["family"] == "H":
            failures.append("CONTROL_OR_COMPETING_HYPOTHESIS_NOT_PROMOTABLE")
        # A historical pass is only a paper-observation candidate; this module
        # has no code path that can promote or allocate it.
        result["q_value"] = float(qvalue)
        result["failure_reasons"] = failures
        result["paper_observation_candidate"] = not failures
        if not failures:
            promoted.append(result["candidate_id"])
    ledger_by_id = {item["candidate_id"]: item for item in ledger}
    for result in candidate_results:
        result["trial_status"] = ledger_by_id[result["candidate_id"]]["status"]
    family_tests = _complete_family_tests(cfg, conservative, all_sessions)
    return {
        "trial_count": len(cfg.candidates),
        "complete_family_includes_unsupported": True,
        "walk_forward_folds": len(folds),
        "candidates": candidate_results,
        "paper_observation_candidates": promoted,
        "promotion_decisions": [],
        "canonical_integration": False,
        "spa": family_tests["spa"],
        "stepm": family_tests["stepm"],
        "parameter_neighborhoods": neighborhood,
    }


def _complete_family_tests(
    cfg: OpeningFadeConfig,
    conservative: pd.DataFrame,
    all_sessions: pd.DatetimeIndex,
) -> dict[str, Any]:
    if len(all_sessions) < 60 or conservative.empty:
        reason = f"need at least 60 aligned sessions; have {len(all_sessions)}"
        return {
            "spa": {"status": "INSUFFICIENT", "reason": reason},
            "stepm": {"status": "INSUFFICIENT", "reason": reason},
        }
    series: dict[str, pd.Series] = {}
    for candidate in cfg.candidates:
        subset = conservative.loc[conservative["candidate_id"] == candidate.candidate_id]
        candidate_returns = aggregate_session_returns(subset).reindex(all_sessions, fill_value=0.0)
        series[candidate.candidate_id] = candidate_returns
    control_id = "H_RANDOM_MATCHED"
    benchmark = series[control_id]
    models = pd.DataFrame(
        {key: value for key, value in series.items() if key not in {control_id, "H_NO_TRADE"}},
        index=all_sessions,
    )
    try:
        spa = spa_test(
            benchmark,
            models,
            reps=cfg.validation.bootstrap_samples,
            block_size=cfg.validation.block_length,
            seed=cfg.validation.random_seed,
        )
        superior = stepm_superior(
            benchmark,
            models,
            reps=cfg.validation.bootstrap_samples,
            block_size=cfg.validation.block_length,
            size=cfg.validation.fdr_alpha,
            seed=cfg.validation.random_seed,
        )
        return {
            "spa": {"status": "CALCULATED", **spa},
            "stepm": {"status": "CALCULATED", "superior_models": superior},
        }
    except (ImportError, ValidationError, ValueError) as exc:
        return {
            "spa": {"status": "UNAVAILABLE", "reason": str(exc)},
            "stepm": {"status": "UNAVAILABLE", "reason": str(exc)},
        }


def _parameter_neighborhoods(candidate_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for candidate in candidate_results:
        metrics = cast(dict[str, Any], candidate["metrics"])
        expectancy = metrics.get("expectancy")
        if expectancy is not None and np.isfinite(cast(Any, expectancy)):
            grouped[str(candidate["family"])].append(_to_float(expectancy))
    output = []
    for family in sorted(set("ABCDEFGH")):
        values = grouped.get(family, [])
        output.append(
            {
                "family": family,
                "evaluated_neighbors": len(values),
                "positive_fraction": (
                    sum(value > 0 for value in values) / len(values) if values else None
                ),
                "median_expectancy": float(np.median(values)) if values else None,
                "status": "CALCULATED" if len(values) >= 2 else "INSUFFICIENT",
            }
        )
    return output


def cost_sensitivity(
    cfg: OpeningFadeConfig,
    scenario_trades: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    rows = []
    violations = []
    order = [scenario.value for scenario in CostScenario]
    for candidate in cfg.candidates:
        means = []
        for scenario in order:
            trades = scenario_trades[scenario]
            subset = trades.loc[trades.get("candidate_id", "") == candidate.candidate_id]
            mean = float(subset["net_return"].mean()) if not subset.empty else None
            means.append(mean)
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "scenario": scenario,
                    "trades": len(subset),
                    "mean_net_return": mean,
                    "total_cost_return": float(subset["cost_return"].sum())
                    if not subset.empty
                    else 0.0,
                }
            )
        observed = [value for value in means if value is not None]
        if any(later > earlier + 1e-12 for earlier, later in pairwise(observed)):
            violations.append(candidate.candidate_id)
    return {
        "scenarios": order,
        "rows": rows,
        "monotonic": not violations,
        "violations": violations,
        "cfd_sensitivity": {
            "status": "NOT_RUN",
            "reason": "no executable broker-CFD bid/ask history; ETF results cannot be transferred",
            "warnings": [
                "eToro NSDQ100/SPX500 are broker CFDs, not QQQ/SPY/NQ/ES/cash indices",
                "chart midpoint is not assumed executable",
                "no trade tickets or live recommendations are produced",
            ],
        },
    }


def _paired_control_summary(
    candidate: pd.DataFrame, control: pd.DataFrame, *, seed: int, n_boot: int, block: int
) -> dict[str, Any]:
    a = aggregate_session_returns(candidate).rename("candidate")
    b = aggregate_session_returns(control).rename("control")
    joined = pd.concat([a, b], axis=1).fillna(0.0)
    if len(joined) < 3:
        return {"status": "INSUFFICIENT", "sessions": len(joined), "ci95": [None, None]}
    difference = (joined["candidate"] - joined["control"]).to_numpy()
    ci = block_bootstrap_ci(
        difference,
        rng=np.random.default_rng(seed),
        n_boot=n_boot,
        block_length=block,
    )
    return {
        "status": "CALCULATED",
        "sessions": len(joined),
        "mean_paired_difference": float(difference.mean()),
        "ci95": list(ci),
    }


def control_analysis(cfg: OpeningFadeConfig, conservative: pd.DataFrame) -> dict[str, Any]:
    ids = {item.candidate_id for item in cfg.candidates}
    pairs: list[dict[str, Any]] = []
    for candidate_id, control_id in (
        ("A_VWAP_15", "H_NAIVE_0945"),
        ("D_GAP_VWAP", "D_NAIVE_GAP"),
        ("A_VWAP_15", "H_CONTINUATION"),
        ("A_VWAP_15", "H_RANDOM_MATCHED"),
    ):
        if candidate_id not in ids or control_id not in ids:
            continue
        candidate = conservative.loc[conservative["candidate_id"] == candidate_id]
        control = conservative.loc[conservative["candidate_id"] == control_id]
        result = _paired_control_summary(
            candidate,
            control,
            seed=cfg.validation.random_seed + len(pairs),
            n_boot=cfg.validation.bootstrap_samples,
            block=cfg.validation.block_length,
        )
        result.update({"candidate_id": candidate_id, "control_id": control_id})
        pairs.append(result)
    return {
        "paired_comparisons": pairs,
        "session_aggregation": (
            "simultaneous instrument trades are equal-weighted into one daily return"
        ),
        "negative_control": next(
            (item for item in pairs if item["control_id"] == "H_RANDOM_MATCHED"),
            {"status": "NOT_CONFIGURED"},
        ),
    }


def instrument_rankings(
    cfg: OpeningFadeConfig,
    features: pd.DataFrame,
    conservative: pd.DataFrame,
    validation: dict[str, Any],
    scenario_trades: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    eligible = features.loc[features.get("eligible", False) == True].copy()  # noqa: E712
    movement: list[dict[str, Any]] = []
    conservative_model = _cost_model(cfg.costs, CostScenario.CONSERVATIVE)
    spread_return = 2.0 * conservative_model.half_spread_bps * conservative_model.multiplier / 1e4
    for symbol, group in eligible.groupby("symbol"):
        absolute = group["opening_range_width"].abs()
        movement.append(
            {
                "symbol": symbol,
                "sessions": len(group),
                "median_absolute_opening_range": float(absolute.median()),
                "p90_absolute_opening_range": float(absolute.quantile(0.90)),
                "p95_absolute_opening_range": float(absolute.quantile(0.95)),
                "median_opening_range_atr": (
                    float(group["opening_range_atr"].dropna().median())
                    if group["opening_range_atr"].notna().any()
                    else None
                ),
                "spread_adjusted_movement": float(absolute.median() - spread_return),
                "modeled_roundtrip_spread_return": spread_return,
                "median_mae": (
                    float(conservative.loc[conservative["symbol"] == symbol, "mae"].median())
                    if not conservative.empty and (conservative["symbol"] == symbol).any()
                    else None
                ),
            }
        )
    movement.sort(key=lambda item: _to_float(item["median_absolute_opening_range"]), reverse=True)
    validation_by_id = {item["candidate_id"]: item for item in validation["candidates"]}
    neighborhoods = {item["family"]: item for item in validation.get("parameter_neighborhoods", [])}
    stress = scenario_trades[CostScenario.STRESS.value]
    tradability: list[dict[str, Any]] = []
    for symbol, group in conservative.groupby("symbol") if not conservative.empty else []:
        research_group = group.loc[group["family"] != "H"]
        candidate_groups = list(research_group.groupby("candidate_id"))
        if not candidate_groups:
            continue
        candidate_id, selected = max(
            candidate_groups,
            key=lambda item: float(item[1]["net_return"].mean()),
        )
        daily = aggregate_session_returns(selected)
        selected_metrics = trade_metrics(
            selected,
            seed=cfg.validation.random_seed,
            n_boot=cfg.validation.bootstrap_samples,
            block_length=cfg.validation.block_length,
        )
        candidate_validation = validation_by_id[str(candidate_id)]
        stress_selected = stress.loc[
            (stress["symbol"] == symbol) & (stress["candidate_id"] == candidate_id)
        ]
        stress_expectancy = (
            float(stress_selected["net_return"].mean()) if not stress_selected.empty else None
        )
        conservative_expectancy = float(selected["net_return"].mean())
        ci = selected_metrics.get("mean_net_return_ci95", [None, None])
        tradability.append(
            {
                "symbol": symbol,
                "candidate_id": candidate_id,
                "selection_note": (
                    "highest observed conservative expectancy; descriptive only and still "
                    "subject to the complete-family multiplicity gates"
                ),
                "trades": len(selected),
                "sessions": len(daily),
                "conservative_expectancy": conservative_expectancy,
                "lower_confidence_bound": ci[0],
                "profit_factor": profit_factor(selected["net_return"].to_numpy())
                if len(selected) >= 2
                else None,
                "maximum_drawdown": max_drawdown(daily.to_numpy()) if len(daily) >= 2 else None,
                "effective_sample_size": effective_sample_size(daily.to_numpy())
                if len(daily) >= 2
                else len(daily),
                "q_value": candidate_validation["q_value"],
                "positive_fold_fraction": candidate_validation["positive_fold_fraction"],
                "parameter_robustness": neighborhoods.get(
                    candidate_validation["family"], {"status": "INSUFFICIENT"}
                ),
                "stress_expectancy": stress_expectancy,
                "execution_sensitivity": (
                    conservative_expectancy - stress_expectancy
                    if stress_expectancy is not None
                    else None
                ),
                "validated": candidate_id in validation["paper_observation_candidates"],
                "median_opening_volume": float(
                    eligible.loc[eligible["symbol"] == symbol, "opening_volume"].median()
                ),
                "liquidity_note": "ETF/stock bar volume only; no historical spread observations",
            }
        )
    tradability.sort(key=lambda item: _to_float(item["conservative_expectancy"]), reverse=True)
    return {
        "movement_ranking": movement,
        "tradability_edge_ranking": tradability,
        "warning": (
            "movement and tradability rankings answer different questions; neither is actionable"
        ),
    }


def paper_risk_example(trades: pd.DataFrame) -> dict[str, Any]:
    """Non-actionable risk illustration; performance is always reported unlevered first."""
    stop_fraction = None
    illustrative_units = None
    if not trades.empty:
        distances = (trades["stop"] - trades["entry"]).abs() / trades["entry"]
        usable = distances.loc[distances > 0]
        if not usable.empty:
            stop_fraction = float(usable.median())
            illustrative_units = 0.0025 / stop_fraction
    return {
        "paper_only": True,
        "account_equity_example": 100_000.0,
        "maximum_fractional_risk_per_trade": 0.0025,
        "maximum_loss_budget_example": 250.0,
        "median_observed_stop_fraction": stop_fraction,
        "illustrative_unit_exposure_at_risk_cap": illustrative_units,
        "leverage_applied_to_reported_returns": False,
        "averaging_down": False,
        "maximum_correlated_index_fade_positions": 1,
        "daily_loss_limit_fraction": 0.005,
        "pause_after_consecutive_losses": 3,
        "creates_order_or_ticket": False,
        "warning": "illustrates path risk only; not a deployment recommendation",
    }


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _load_inputs(
    repo_root: Path,
    campaign: OpeningFadeConfig,
    *,
    symbols: tuple[str, ...],
    interval_minutes: int,
    start: date | None,
    end: date | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    repository_cfg = load_config(repo_root / campaign.repository_config)
    catalog = DataCatalog(repository_cfg)
    intraday_frames = []
    data_hashes: dict[str, str] = {}
    for symbol in symbols:
        path = catalog.intraday_dir / f"{interval_minutes}m" / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = catalog.load_intraday_bars(symbol, interval_minutes=interval_minutes)
        local_dates = frame["timestamp"].dt.tz_convert(EXCHANGE_TZ).dt.date
        if start is not None:
            frame = frame.loc[local_dates >= start]
            local_dates = frame["timestamp"].dt.tz_convert(EXCHANGE_TZ).dt.date
        if end is not None:
            frame = frame.loc[local_dates <= end]
        if not frame.empty:
            intraday_frames.append(frame)
            data_hashes[str(path.resolve().relative_to(repo_root.resolve()))] = _hash_file(path)
    if not intraday_frames:
        return (
            pd.DataFrame(columns=INTRADAY_BAR_COLUMNS),
            pd.DataFrame(),
            pd.DataFrame(),
            data_hashes,
        )
    intraday = pd.concat(intraday_frames, ignore_index=True)
    available_daily = tuple(symbol for symbol in symbols if symbol in catalog.list_symbols())
    daily = pd.DataFrame()
    if available_daily:
        daily = catalog.load_panel(available_daily, start=start, end=end)
        for symbol in available_daily:
            path = catalog.prices_dir / f"{symbol}.parquet"
            data_hashes[str(path.resolve().relative_to(repo_root.resolve()))] = _hash_file(path)
    corporate_actions = catalog.load_corporate_actions(available_daily, start=start, end=end)
    return intraday, daily, corporate_actions, data_hashes


def _plain_summary(result: dict[str, Any]) -> str:
    audit = result["data_audit"]
    sessions = result["occurrence"].get("total_eligible_sessions", 0)
    candidates = result["validation"].get("paper_observation_candidates", [])
    if audit["overall_classification"] == DataSufficiency.INSUFFICIENT:
        conclusion = (
            f"The available archive contains only {sessions} eligible symbol-sessions at this "
            "resolution, so it cannot estimate a stable opening-fade frequency or expectancy."
        )
    elif candidates:
        conclusion = (
            "One or more historical variants cleared the configured research gates, but they "
            "remain "
            "paper-only candidates requiring frozen prospective confirmation."
        )
    else:
        conclusion = (
            "No strategy variant cleared every research gate under conservative assumptions."
        )
    return f"{conclusion} {DISCLAIMER}"


def render_markdown_report(result: dict[str, Any]) -> str:
    summary = _plain_summary(result)
    occurrence = {item["definition"]: item for item in result["occurrence"].get("summary", [])}
    movement = result["rankings"].get("movement_ranking", [])
    tradability = result["rankings"].get("tradability_edge_ranking", [])
    best_move = movement[0]["symbol"] if movement else "not measurable"
    best_edge = (
        tradability[0]["symbol"]
        if tradability and result["validation"]["paper_observation_candidates"]
        else "none"
    )
    fade = occurrence.get("retraced_50", {})
    positive = [
        item
        for item in result["occurrence"].get("conditional", [])
        if item.get("definition") == "retraced_50"
        and item.get("condition") == "gap_direction"
        and item.get("value") == "positive"
    ]
    continuation = occurrence.get("subsequent_new_high", {})
    lines = [
        summary,
        "",
        "# Opening spike and fade research report",
        "",
        "## Direct answers",
        "",
        f"- Opening-fade frequency (50% retracement): {_format_frequency(fade)}.",
        f"- Frequency after a positive gap: {_format_frequency(positive[0] if positive else {})}.",
        f"- Sharpest observed movement: {best_move}; this is not a tradability conclusion.",
        f"- Best validated net expectancy: {best_edge}; no instrument is labeled actionable.",
        (
            "- QQQ versus SPY stability: insufficient evidence unless the table below "
            "contains a research-usable sample for both."
        ),
        (
            "- IWM execution/consistency: not testable when IWM intraday history is "
            "absent or insufficient."
        ),
        (
            "- Individual stocks: reported separately only when liquid intraday inputs "
            "pass the data gates."
        ),
        (
            "- Event-day predictability: not testable without a point-in-time event "
            "calendar and a research-usable event sample."
        ),
        (
            "- Improving versus overfit filters: none can be identified without complete "
            "out-of-sample folds."
        ),
        f"- Continuation-day trap rate after an observed fade: {_format_frequency(continuation)}.",
        (
            "- Conservative/stress-cost survival: "
            f"{len(result['validation']['paper_observation_candidates'])} candidate(s) "
            "cleared all gates."
        ),
        f"- Data sufficiency: {result['data_audit']['overall_classification']}.",
        (
            "- Prospective monitoring: permitted only after a candidate clears every "
            "historical gate; none is automatically promoted."
        ),
        "",
        "## Exact definitions",
        "",
        "```json",
        json.dumps(result["definition"], indent=2, sort_keys=True, default=str),
        "```",
        "",
        "## Data audit",
        "",
        (
            f"Provider: `{result['data_audit']['provider']}`. Timestamps: "
            f"{result['data_audit']['timestamps']}."
        ),
        (
            f"Premarket available: {result['data_audit']['premarket_available']}. "
            f"Bid/ask: {result['data_audit']['bid_ask']}."
        ),
        f"Breadth: {result['data_audit']['breadth']['reason']}.",
        "",
        "## Occurrence statistics",
        "",
        "| Definition | Count | Sessions | Frequency | 95% CI | ESS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in result["occurrence"].get("summary", []):
        ci = item.get("ci95", [None, None])
        lines.append(
            f"| {item['definition']} | {item['count']} | {item['sessions']} | "
            f"{_pct(item.get('frequency'))} | {_pct(ci[0])}-{_pct(ci[1])} | "
            f"{item['effective_sample_size']:.1f} |"
        )
    lines.extend(
        [
            "",
            "### Time to retracement",
            "",
            "Times are completed-bar minutes from the 09:30 ET open.",
            "",
            "| Retracement | Observations | Median | 25th percentile | 75th percentile |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for fraction, item in result["occurrence"].get("time_to_fade", {}).items():
        median = item["median_minutes_from_open"]
        p25 = item["p25_minutes_from_open"]
        p75 = item["p75_minutes_from_open"]
        lines.append(
            f"| {fraction}% | {item['count']} | "
            f"{median if median is not None else 'n/a'} | "
            f"{p25 if p25 is not None else 'n/a'} | "
            f"{p75 if p75 is not None else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## Instrument rankings",
            "",
            (
                "Movement and tradability are separate. A larger opening range is not "
                "evidence of a more tradable edge."
            ),
            "",
            "| Instrument | Sessions | Median opening range | P90 | P95 | Range / daily ATR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in movement:
        range_atr = item["median_opening_range_atr"]
        lines.append(
            f"| {item['symbol']} | {item['sessions']} | "
            f"{_pct(item['median_absolute_opening_range'])} | "
            f"{_pct(item['p90_absolute_opening_range'])} | "
            f"{_pct(item['p95_absolute_opening_range'])} | "
            f"{range_atr if range_atr is not None else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            (
                "| Instrument | Descriptive best candidate | Trades | Conservative expectancy | "
                "Lower 95% bound | Stress expectancy | ESS | Validated |"
            ),
            "|---|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for item in tradability:
        lines.append(
            f"| {item['symbol']} | {item['candidate_id']} | {item['trades']} | "
            f"{_pct(item['conservative_expectancy'])} | "
            f"{_pct(item['lower_confidence_bound'])} | {_pct(item['stress_expectancy'])} | "
            f"{item['effective_sample_size']:.1f} | {item['validated']} |"
        )
    lines.extend(
        [
            "",
            "## Complete strategy family and validation",
            "",
            (
                f"All {result['validation']['trial_count']} configured trials - including "
                "unsupported and zero-trade variants - are counted."
            ),
            "",
            (
                "| Candidate | Family | Trades | Gross expectancy | Conservative expectancy | "
                "q-value | DSR | Result |"
            ),
            "|---|:---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in result["validation"]["candidates"]:
        metrics = item["metrics"]
        lines.append(
            f"| {item['candidate_id']} | {item['family']} | {metrics.get('trade_count', 0)} | "
            f"{_pct(metrics.get('gross_expectancy'))} | {_pct(metrics.get('expectancy'))} | "
            f"{item['q_value']:.4f} | "
            f"{item['deflated_sharpe_probability']:.4f} | "
            f"{'; '.join(item['failure_reasons']) or 'paper-only gate pass'} |"
        )
    fold_rows = [
        (candidate["candidate_id"], fold)
        for candidate in result["validation"]["candidates"]
        for fold in candidate.get("folds", [])
    ]
    lines.extend(["", "### Walk-forward fold results", ""])
    if fold_rows:
        lines.extend(
            [
                "| Candidate | Fold | Train end | Test window | Test sessions | Mean net return |",
                "|---|---:|---|---|---:|---:|",
            ]
        )
        for candidate_id, fold in fold_rows:
            lines.append(
                f"| {candidate_id} | {fold['fold_id']} | {fold['train_end']} | "
                f"{fold['test_start']} to {fold['test_end']} | {fold['test_sessions']} | "
                f"{_pct(fold['mean_net_return'])} |"
            )
    else:
        lines.append("No complete out-of-sample fold exists in this archive.")
    lines.extend(
        [
            "",
            "## Cost, controls, and robustness",
            "",
            (
                f"Cost monotonicity: {result['cost_sensitivity']['monotonic']}. CFD "
                f"sensitivity: {result['cost_sensitivity']['cfd_sensitivity']['status']}."
            ),
            (
                f"Walk-forward folds available: {result['validation']['walk_forward_folds']} "
                f"of {result['validation_plan']['n_folds']}."
            ),
            (
                "Extreme-session exclusion results and paired controls are retained in "
                "`metrics.json` even when insufficient."
            ),
            (
                "The paper-only risk illustration caps hypothetical loss at 0.25% of "
                "example equity, permits one correlated index fade, forbids averaging "
                "down, and does not create an order."
            ),
        ]
    )
    cost_rows = result["cost_sensitivity"].get("rows", [])
    lines.extend(
        [
            "",
            "| Cost scenario | Candidates with trades | Median candidate expectancy |",
            "|---|---:|---:|",
        ]
    )
    for scenario in result["cost_sensitivity"].get("scenarios", []):
        values = [
            item["mean_net_return"]
            for item in cost_rows
            if item["scenario"] == scenario and item["mean_net_return"] is not None
        ]
        lines.append(
            f"| {scenario} | {len(values)} | {_pct(float(np.median(values)) if values else None)} |"
        )
    lines.extend(
        [
            "",
            "| Candidate | Paired control | Sessions | Mean difference | 95% CI | Status |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for item in result["controls"].get("paired_comparisons", []):
        ci = item.get("ci95", [None, None])
        lines.append(
            f"| {item['candidate_id']} | {item['control_id']} | {item['sessions']} | "
            f"{_pct(item.get('mean_paired_difference'))} | {_pct(ci[0])} to "
            f"{_pct(ci[1])} | {item['status']} |"
        )
    lines.extend(["", "## Important negative findings and limitations", ""])
    lines.extend(f"- {warning}" for warning in result["warnings"])
    lines.extend(
        [
            "",
            "## Prospective evidence milestones",
            "",
            "- 3 months: archive/clock/data-quality audit and descriptive examples only.",
            (
                "- 6 months: exploratory frequency estimates by primary ETF; still weak "
                "for regimes and multiplicity."
            ),
            (
                "- 12 months: first bounded walk-forward study may become possible if "
                "1m/5m coverage is continuous."
            ),
            (
                "- 24 months: multi-regime research may become `RESEARCH_USABLE`; "
                "promotion still requires separate V2 and prospective gates."
            ),
            "",
            (
                "Exact next step: collect extended-hours 1m and 5m OHLCV every session "
                "for the frozen liquid universe, then rerun this unchanged manifest after "
                "at least 252 complete sessions."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _format_frequency(item: dict[str, Any]) -> str:
    if not item or item.get("frequency") is None:
        return "not measurable"
    return (
        f"{_pct(item['frequency'])} ({item['count']}/{item['sessions']}, Wilson 95% CI "
        f"{_pct(item['ci95'][0])}-{_pct(item['ci95'][1])})"
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{float(value) * 100:.2f}%"


def render_html_report(markdown: str) -> str:
    """Self-contained readable HTML; markdown source remains canonical."""
    lines = markdown.splitlines()
    output = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Opening fade research</title>",
        (
            "<style>body{font:15px system-ui;max-width:1200px;margin:2rem auto;"
            "padding:0 1rem;color:#18202a}table{border-collapse:collapse;width:100%;"
            "font-size:12px}th,td{border:1px solid #ccd4dd;padding:.35rem;"
            "vertical-align:top}th{background:#eef2f6}code,pre{background:#f4f6f8;"
            "padding:.2rem;overflow:auto}.warning{background:#fff4d6;padding:1rem;"
            "border-left:4px solid #cc8a00}</style></head><body>"
        ),
    ]
    in_code = False
    in_list = False
    in_table = False
    for index, raw in enumerate(lines):
        if raw.startswith("```"):
            if in_code:
                output.append("</code></pre>")
            else:
                output.append("<pre><code>")
            in_code = not in_code
            continue
        if in_code:
            output.append(html.escape(raw))
            continue
        if raw.startswith("|---"):
            continue
        if raw.startswith("|"):
            cells = [html.escape(cell.strip()) for cell in raw.strip("|").split("|")]
            first_table_row = not in_table
            if not in_table:
                output.append("<table>")
                in_table = True
            tag = "th" if first_table_row else "td"
            output.append("<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>")
            continue
        if in_table:
            output.append("</table>")
            in_table = False
        if raw.startswith("- "):
            if not in_list:
                output.append("<ul>")
                in_list = True
            output.append(f"<li>{html.escape(raw[2:])}</li>")
            continue
        if in_list:
            output.append("</ul>")
            in_list = False
        if raw.startswith("# "):
            output.append(f"<h1>{html.escape(raw[2:])}</h1>")
        elif raw.startswith("## "):
            output.append(f"<h2>{html.escape(raw[3:])}</h2>")
        elif raw.startswith("### "):
            output.append(f"<h3>{html.escape(raw[4:])}</h3>")
        elif raw:
            css = " class='warning'" if index == 0 else ""
            output.append(f"<p{css}>{html.escape(raw)}</p>")
    if in_table:
        output.append("</table>")
    if in_list:
        output.append("</ul>")
    output.append("</body></html>")
    return "\n".join(output)


def publish_immutable_run(
    result: dict[str, Any],
    trades: pd.DataFrame,
    *,
    artifacts_root: Path,
) -> Path:
    # Wall-clock metadata describes publication, not research content. Excluding
    # it makes an identical rerun resolve to the same immutable directory.
    stable_manifest = {
        key: value
        for key, value in result["manifest"].items()
        if key not in {"created_at", "run_id"}
    }
    stable_audit = {
        key: value for key, value in result["data_audit"].items() if key != "checked_at"
    }
    manifest_seed = {
        "manifest": stable_manifest,
        "data_audit": stable_audit,
        "occurrence": result["occurrence"],
        "validation": result["validation"],
        "cost_sensitivity": result["cost_sensitivity"],
        "controls": result["controls"],
        "rankings": result["rankings"],
        "paper_risk_example": result.get("paper_risk_example", {}),
        "trades_hash": _hash_json(trades.to_dict(orient="records")),
    }
    run_id = _hash_json(manifest_seed)
    result["manifest"]["run_id"] = run_id
    runs_dir = artifacts_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    destination = runs_dir / run_id
    markdown = render_markdown_report(result)
    html_report = render_html_report(markdown)
    if not destination.exists():
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_dir))
        try:
            atomic_write_bytes(
                temp_dir / "manifest.json",
                json.dumps(result["manifest"], indent=2, sort_keys=True, default=str).encode(),
            )
            atomic_write_bytes(
                temp_dir / "data_audit.json",
                json.dumps(result["data_audit"], indent=2, sort_keys=True, default=str).encode(),
            )
            atomic_write_bytes(
                temp_dir / "occurrence.json",
                json.dumps(result["occurrence"], indent=2, sort_keys=True, default=str).encode(),
            )
            parquet = io.BytesIO()
            trades.to_parquet(parquet, index=False)
            atomic_write_bytes(temp_dir / "trades.parquet", parquet.getvalue())
            atomic_write_bytes(
                temp_dir / "metrics.json",
                json.dumps(
                    {
                        "validation": result["validation"],
                        "cost_sensitivity": result["cost_sensitivity"],
                        "controls": result["controls"],
                        "rankings": result["rankings"],
                        "paper_risk_example": result.get("paper_risk_example", {}),
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                ).encode(),
            )
            atomic_write_bytes(temp_dir / "report.md", markdown.encode())
            atomic_write_bytes(temp_dir / "report.html", html_report.encode())
            os.replace(temp_dir, destination)
        except BaseException:
            # Temp content is outside the published namespace. Leave it for
            # forensic inspection rather than risk deleting an ambiguous path.
            raise
    published_at = result["manifest"]["created_at"]
    if destination.exists():
        existing_manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if existing_manifest.get("run_id") != run_id:
            raise DataError(f"immutable run manifest mismatch at {destination}")
        published_at = existing_manifest["created_at"]
    pointer = {
        "run_id": run_id,
        "path": f"runs/{run_id}",
        "classification": result["data_audit"]["overall_classification"],
        "published_at": published_at,
        "paper_only": True,
    }
    atomic_write_bytes(
        artifacts_root / "current.json",
        json.dumps(pointer, indent=2, sort_keys=True, default=str).encode(),
    )
    return destination


def run_opening_fade_campaign(
    cfg: OpeningFadeConfig,
    *,
    repo_root: Path,
    symbols: tuple[str, ...] | None = None,
    interval_minutes: int | None = None,
    start: date | None = None,
    end: date | None = None,
    publish: bool = True,
    run_strategies: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame, Path | None]:
    wanted = symbols or cfg.primary_symbols
    interval = interval_minutes or cfg.primary_interval_minutes
    if interval not in {1, 5, 15}:
        raise ValidationError("opening-fade sequence research requires 1m, 5m, or coarse 15m bars")
    audit = audit_local_data(repo_root, cfg)
    intraday, daily, corporate_actions, data_hashes = _load_inputs(
        repo_root,
        cfg,
        symbols=wanted,
        interval_minutes=interval,
        start=start,
        end=end,
    )
    if intraday.empty:
        raise DataError(f"no local {interval}m intraday data for {wanted}")
    event_calendar = None
    if cfg.event_calendar_csv:
        event_path = repo_root / cfg.event_calendar_csv
        if event_path.exists():
            event_calendar = CsvEventCalendar(event_path)
    features, session_bars = build_session_features(
        intraday,
        daily_context=build_daily_context(daily),
        definition=cfg.definition,
        corporate_actions=corporate_actions,
        event_calendar=event_calendar,
        individual_stock_symbols=cfg.stock_symbols,
        minimum_stock_price=cfg.minimum_stock_price,
        minimum_stock_opening_dollar_volume=cfg.minimum_stock_opening_dollar_volume,
    )
    occurrence = occurrence_analysis(features, cfg.definition)
    scenario_trades: dict[str, pd.DataFrame] = {}
    conservative_ledger: list[dict[str, Any]] = []
    if run_strategies:
        for scenario in CostScenario:
            frame, ledger = generate_candidate_trades(
                cfg, features, session_bars, audit, scenario=scenario
            )
            scenario_trades[scenario.value] = frame
            if scenario is CostScenario.CONSERVATIVE:
                conservative_ledger = ledger
        delayed_conservative, _ = generate_candidate_trades(
            cfg,
            features,
            session_bars,
            audit,
            scenario=CostScenario.CONSERVATIVE,
            entry_delay_bars=1,
        )
    else:
        delayed_conservative = pd.DataFrame(columns=list(SimulatedTrade.__dataclass_fields__))
        for scenario in CostScenario:
            scenario_trades[scenario.value] = delayed_conservative.copy()
        conservative_ledger = [
            {
                "candidate_id": item.candidate_id,
                "family": item.family,
                "status": "NOT_RUN",
                "reason": "DESCRIBE_ONLY_MODE",
                "trades": 0,
                "params": item.params,
            }
            for item in cfg.candidates
        ]
    scenario_trades["delayed_conservative"] = delayed_conservative
    conservative = scenario_trades[CostScenario.CONSERVATIVE.value]
    all_sessions = pd.DatetimeIndex(
        sorted(pd.to_datetime(features.loc[features["eligible"].astype(bool), "session"]).unique())
    )
    validation = validate_candidates(cfg, scenario_trades, conservative_ledger, all_sessions)
    sensitivity = cost_sensitivity(cfg, scenario_trades)
    controls = control_analysis(cfg, conservative)
    rankings = instrument_rankings(cfg, features, conservative, validation, scenario_trades)
    risk_example = paper_risk_example(conservative)
    selected_rows = [
        item
        for item in audit["files"]
        if item["symbol"] in wanted and item["interval_minutes"] == interval
    ]
    selected_sessions = max((int(item.get("sessions", 0)) for item in selected_rows), default=0)
    selected_classification = classify_data_sufficiency(
        selected_sessions, interval_minutes=interval
    )
    audit = dict(audit)
    audit["overall_classification"] = selected_classification
    warnings = list(audit["warnings"])
    if interval == 15:
        warnings.append(
            "This run is a coarse 15-minute secondary study; bar sequencing is ambiguous."
        )
    if selected_sessions < 252:
        warnings.append(
            f"Only {selected_sessions} sessions are locally available at {interval}m; "
            "no robust historical conclusion is supported."
        )
    if start is None or (start and start >= cfg.previously_accessed_start):
        warnings.append(
            "All sampled 2024-2026 history is previously accessed and cannot promote a V2 sleeve."
        )
    warnings.extend(
        [
            "No historical bid/ask, auction imbalance, or venue-level fill data is available.",
            (
                "No point-in-time breadth result is reported without timestamp-aligned "
                "historical members."
            ),
            "No causal claim is made about economic releases merely because clock times coincide.",
            "Individual-stock findings require separate earnings/event and liquidity treatment.",
        ]
    )
    manifest = {
        "campaign": cfg.campaign_version,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(repo_root),
        "dirty_worktree": bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=repo_root, text=True, capture_output=True
            ).stdout.strip()
        ),
        "config_hash": cfg.stable_hash(),
        "definition_hash": _hash_json(cfg.definition.model_dump(mode="json")),
        "code_hash": _hash_file(Path(__file__)),
        "data_hashes": data_hashes,
        "provider": cfg.provider,
        "symbols": list(wanted),
        "interval_minutes": interval,
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "session_filters": {
            "exchange_timezone": EXCHANGE_TZ,
            "missing_opening_bar": "excluded",
            "early_close": "excluded from initial strategy campaign",
            "corporate_action_boundary": "excluded",
        },
        "candidate_family": [item.model_dump(mode="json") for item in cfg.candidates],
        "trial_count": len(cfg.candidates),
        "costs": cfg.costs.model_dump(mode="json"),
        "random_seed": cfg.validation.random_seed,
        "previously_accessed": True,
        "paper_only": True,
        "study_mode": "STRATEGIES" if run_strategies else "DESCRIBE_ONLY",
        "canonical_integration": False,
        "warnings": warnings,
    }
    result = {
        "summary": "filled during rendering",
        "manifest": manifest,
        "definition": cfg.definition.model_dump(mode="json"),
        "validation_plan": cfg.validation.model_dump(mode="json"),
        "data_audit": audit,
        "occurrence": occurrence,
        "trial_ledger": conservative_ledger,
        "validation": validation,
        "cost_sensitivity": sensitivity,
        "controls": controls,
        "rankings": rankings,
        "paper_risk_example": risk_example,
        "warnings": warnings,
        "conclusion": (
            "SUITABLE_FOR_PROSPECTIVE_PAPER_OBSERVATION"
            if validation["paper_observation_candidates"]
            else "BLOCKED_BY_INSUFFICIENT_DATA"
            if selected_classification == DataSufficiency.INSUFFICIENT
            else "REJECTED"
        ),
        "next_research_step": (
            "Continue frozen extended-hours 1m/5m forward collection; rerun unchanged "
            "after at least 252 complete sessions."
        ),
    }
    result["summary"] = _plain_summary(result)
    destination = None
    if publish:
        artifacts = repo_root / cfg.artifacts_dir
        destination = publish_immutable_run(result, conservative, artifacts_root=artifacts)
    return result, conservative, destination


def synthetic_intraday_fixture(
    *,
    sessions: int = 80,
    symbols: tuple[str, ...] = ("SPY", "QQQ"),
    interval_minutes: int = 5,
    injected_fade: float = 0.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic random-walk fixture with an optional opening-fade effect."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-02", periods=sessions)
    bars = []
    daily = []
    prices = {symbol: 100.0 + index * 10 for index, symbol in enumerate(symbols)}
    bars_per_day = int((16 * 60 - (9 * 60 + 30)) / interval_minutes)
    for session in days:
        for symbol in symbols:
            previous = prices[symbol]
            gap = 0.003
            opening = previous * (1.0 + gap)
            current = opening
            session_rows: list[dict[str, Any]] = []
            for position in range(bars_per_day):
                timestamp_et = pd.Timestamp.combine(session.date(), time(9, 30)).tz_localize(
                    EXCHANGE_TZ
                ) + pd.Timedelta(minutes=position * interval_minutes)
                drift = rng.normal(0, 0.00035)
                if position < 3:
                    drift += 0.0008
                if 3 <= position < 8:
                    drift -= injected_fade / 5.0
                close = current * (1.0 + drift)
                high = max(current, close) * (1.0 + abs(rng.normal(0, 0.0001)))
                low = min(current, close) * (1.0 - abs(rng.normal(0, 0.0001)))
                session_rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": timestamp_et.tz_convert("UTC"),
                        "interval_minutes": interval_minutes,
                        "open": current,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": float(1_000_000 + rng.integers(0, 200_000)),
                    }
                )
                current = close
            bars.extend(session_rows)
            high = max(_to_float(item["high"]) for item in session_rows)
            low = min(_to_float(item["low"]) for item in session_rows)
            daily.append(
                {
                    "symbol": symbol,
                    "date": session,
                    "open": opening,
                    "high": high,
                    "low": low,
                    "close": current,
                    "volume": sum(_to_float(item["volume"]) for item in session_rows),
                    "adj_close": current,
                }
            )
            prices[symbol] = current
    return pd.DataFrame(bars), pd.DataFrame(daily)
