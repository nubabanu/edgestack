"""Configuration loading and validation.

The whole configuration tree is validated at startup. Invalid values fail
fast with a precise error instead of surfacing as wrong research results.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edgestack.exceptions import ConfigError
from edgestack.types import CostScenario


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProjectConfig(_Section):
    name: str = "EdgeStack"
    timezone: str = "America/New_York"
    random_seed: int = 42


class PathsConfig(_Section):
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")


class UniverseConfig(_Section):
    source: Literal["local", "stooq", "synthetic"] = "local"
    symbols: tuple[str, ...] = ()
    benchmark_symbol: str = "SPY"
    min_price: float = Field(default=5.0, gt=0)
    min_median_dollar_volume: float = Field(default=5_000_000, ge=0)
    min_history_sessions: int = Field(default=504, ge=1)
    include_etfs: bool = False
    point_in_time: bool = False

    @field_validator("symbols")
    @classmethod
    def _symbols_upper_unique(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for s in v:
            s = s.strip().upper()
            if not s.isascii() or not s.replace(".", "").replace("-", "").isalnum():
                raise ValueError(f"invalid symbol: {s!r}")
            seen[s] = None
        return tuple(seen)


class DataConfig(_Section):
    calendar: str = "XNYS"
    cache_ttl_days: int = Field(default=3, ge=0)
    request_timeout_seconds: float = Field(default=30, gt=0)
    max_retries: int = Field(default=3, ge=0)


class SignalsConfig(_Section):
    execution_delay_sessions: int = Field(default=1, ge=1)
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 40, 60)
    min_effective_sample_size: int = Field(default=100, ge=1)
    min_probability_of_profit: float = Field(default=0.55, ge=0.5, le=1.0)
    min_expected_net_return: float = Field(default=0.002, ge=0.0)
    min_conviction_score: float = Field(default=65, ge=0, le=100)
    min_reward_to_risk: float = Field(default=1.2, ge=0)
    abstain_on_missing_borrow: bool = True
    max_candidates_per_side: int = Field(default=20, ge=1)

    @field_validator("horizons")
    @classmethod
    def _horizons_positive_sorted(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v or any(h < 1 for h in v) or list(v) != sorted(set(v)):
            raise ValueError("horizons must be unique, ascending, positive integers")
        return v


class ValidationConfig(_Section):
    method: Literal["expanding_walk_forward", "rolling_walk_forward"] = "expanding_walk_forward"
    n_folds: int = Field(default=5, ge=2)
    train_min_sessions: int = Field(default=756, ge=60)
    test_sessions: int = Field(default=252, ge=20)
    embargo_sessions: int = Field(default=60, ge=0)
    final_test_start: date = date(2023, 1, 1)
    fdr_alpha: float = Field(default=0.05, gt=0, lt=1)
    fdr_method: Literal["benjamini_hochberg", "benjamini_yekutieli"] = "benjamini_hochberg"
    bootstrap_samples: int = Field(default=2000, ge=100)
    block_length_sessions: int = Field(default=20, ge=1)
    min_subperiod_consistency: float = Field(default=0.6, ge=0, le=1)


class DiscoveryConfig(_Section):
    max_condition_depth: int = Field(default=3, ge=1, le=5)
    min_support: int = Field(default=50, ge=10)
    max_rules_per_batch: int = Field(default=5000, ge=1)
    quantile_bins: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

    @field_validator("quantile_bins")
    @classmethod
    def _bins_valid(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if any(not 0 < q < 1 for q in v) or list(v) != sorted(set(v)):
            raise ValueError("quantile_bins must be strictly increasing in (0, 1)")
        return v


class CostsConfig(_Section):
    scenario: CostScenario = CostScenario.CONSERVATIVE
    commission_bps: float = Field(default=0.0, ge=0)
    half_spread_bps: float = Field(default=5.0, ge=0)
    base_slippage_bps: float = Field(default=5.0, ge=0)
    impact_coeff_bps: float = Field(default=10.0, ge=0)
    short_borrow_annualized: float = Field(default=0.05, ge=0)
    sec_fee_bps: float = Field(default=0.03, ge=0)


class ScoringConfig(_Section):
    neutral_conviction: float = Field(default=50.0, ge=0, le=100)
    logistic_slope: float = Field(default=6.0, gt=0)
    family_cap: float = Field(default=0.4, gt=0, le=1)
    within_family_lambda: float = Field(default=0.25, ge=0, le=1)
    shrinkage_min_sample: int = Field(default=30, ge=1)


class RiskConfig(_Section):
    max_position_weight: float = Field(default=0.03, gt=0, le=1)
    max_sector_weight: float = Field(default=0.20, gt=0, le=1)
    max_gross_exposure: float = Field(default=1.50, gt=0)
    max_net_exposure: float = Field(default=0.50, ge=0)
    max_positions: int = Field(default=40, ge=1)
    target_annualized_volatility: float = Field(default=0.12, gt=0)
    max_portfolio_drawdown: float = Field(default=0.15, gt=0, le=1)
    atr_stop_multiple: float = Field(default=2.0, gt=0)
    atr_target_multiples: tuple[float, ...] = (2.5, 4.0)
    atr_window: int = Field(default=14, ge=2)

    @model_validator(mode="after")
    def _net_le_gross(self) -> RiskConfig:
        if self.max_net_exposure > self.max_gross_exposure:
            raise ValueError("max_net_exposure cannot exceed max_gross_exposure")
        return self


class MonitoringConfig(_Section):
    rolling_window_signals: int = Field(default=60, ge=10)
    degrade_posterior_threshold: float = Field(default=0.40, gt=0, lt=1)
    suspend_posterior_threshold: float = Field(default=0.25, gt=0, lt=1)
    retire_after_suspended_sessions: int = Field(default=126, ge=1)
    calibration_ece_threshold: float = Field(default=0.10, gt=0, lt=1)

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> MonitoringConfig:
        if self.suspend_posterior_threshold >= self.degrade_posterior_threshold:
            raise ValueError("suspend threshold must be below degrade threshold")
        return self


class PaperConfig(_Section):
    live_trading_enabled: bool = False
    initial_cash: float = Field(default=100_000.0, gt=0)

    @field_validator("live_trading_enabled")
    @classmethod
    def _no_live_trading(cls, v: bool) -> bool:
        # Hard safety rail: this repository ships no live adapter. The flag
        # exists so a future adapter must be enabled deliberately — but the
        # bundled code refuses it outright.
        if v:
            raise ValueError(
                "live_trading_enabled=true is not supported: EdgeStack contains "
                "no live trading implementation."
            )
        return v


class EdgeStackConfig(_Section):
    project: ProjectConfig = ProjectConfig()
    paths: PathsConfig = PathsConfig()
    universe: UniverseConfig = UniverseConfig()
    data: DataConfig = DataConfig()
    signals: SignalsConfig = SignalsConfig()
    validation: ValidationConfig = ValidationConfig()
    discovery: DiscoveryConfig = DiscoveryConfig()
    costs: CostsConfig = CostsConfig()
    scoring: ScoringConfig = ScoringConfig()
    risk: RiskConfig = RiskConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    paper: PaperConfig = PaperConfig()

    @model_validator(mode="after")
    def _embargo_covers_horizons(self) -> EdgeStackConfig:
        max_horizon = max(self.signals.horizons)
        if self.validation.embargo_sessions < max_horizon:
            raise ValueError(
                f"validation.embargo_sessions ({self.validation.embargo_sessions}) must be "
                f">= the longest signal horizon ({max_horizon}) to prevent label leakage"
            )
        return self

    def config_hash(self) -> str:
        """SHA-256 over the canonicalized resolved configuration."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path | None = None) -> EdgeStackConfig:
    """Load and validate configuration from YAML.

    Resolution order: explicit ``path`` argument, ``EDGESTACK_CONFIG`` env var,
    ``configs/default.yaml`` relative to the working directory, built-in defaults.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    elif env := os.environ.get("EDGESTACK_CONFIG"):
        candidates.append(Path(env))
    else:
        candidates.append(Path("configs/default.yaml"))

    chosen = candidates[0]
    if not chosen.exists():
        if path is not None or os.environ.get("EDGESTACK_CONFIG"):
            raise ConfigError(f"config file not found: {chosen}")
        return EdgeStackConfig()

    try:
        raw: Any = yaml.safe_load(chosen.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - depends on malformed input
        raise ConfigError(f"could not parse {chosen}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"top level of {chosen} must be a mapping")
    try:
        return EdgeStackConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"invalid configuration in {chosen}: {exc}") from exc
