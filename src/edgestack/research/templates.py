"""Finite campaign templates registered before any evaluation occurs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from itertools import product
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.calendar import TradingCalendar
from edgestack.recommendation.hashing import stable_hash
from edgestack.recommendation.manifests import TrialKind, TrialRecordV2, TrialStatus
from edgestack.research.coverage import DataRequirement
from edgestack.research.schemas import CampaignLifecycle, CampaignSummaryV1
from edgestack.research.store import ResearchStore
from edgestack.research.universe import ALL_RESEARCH_ETFS, INVERSE_ETFS


@dataclass(frozen=True)
class CampaignTemplate:
    template_id: str
    name: str
    family: str
    evaluator: str
    candidate_family: tuple[dict[str, Any], ...]
    requirements: tuple[DataRequirement, ...]
    previously_accessed: bool = True

    @property
    def manifest_hash(self) -> str:
        return stable_hash(
            {
                "template_id": self.template_id,
                "name": self.name,
                "family": self.family,
                "evaluator": self.evaluator,
                "candidate_family": self.candidate_family,
                "requirements": [item.__dict__ for item in self.requirements],
                "previously_accessed": self.previously_accessed,
            }
        )


def _grid(
    rule: str, symbols: tuple[str, ...], **parameters: tuple[Any, ...]
) -> tuple[dict[str, Any], ...]:
    names = tuple(parameters)
    values = tuple(parameters[name] for name in names)
    return tuple(
        {"rule": rule, "symbol": symbol, **dict(zip(names, choice, strict=True))}
        for symbol in symbols
        for choice in product(*values)
    )


def _last_complete_session() -> date:
    now = datetime.now(ZoneInfo("America/New_York"))
    calendar = TradingCalendar()
    if calendar.is_session(now.date()) and now.time() >= time(16, 16):
        return now.date()
    return calendar.prev_session(now.date()).date()


def _monthly_liquid_stocks(
    cfg: EdgeStackConfig,
    end: date,
    *,
    limit: int = 25,
) -> tuple[str, ...]:
    directory = Path(cfg.paths.data_dir) / "curated" / "research_universe"
    eligible = []
    for path in directory.glob("????????-*.parquet") if directory.exists() else ():
        try:
            snapshot_date = date.fromisoformat(f"{path.name[:4]}-{path.name[4:6]}-{path.name[6:8]}")
        except ValueError:
            continue
        if snapshot_date <= end:
            eligible.append((snapshot_date, path))
    if not eligible:
        return ("AAPL", "MSFT", "NVDA", "AMZN")
    _, selected = max(eligible, key=lambda item: (item[0], item[1].name))
    frame = pd.read_parquet(selected)
    stocks = frame.loc[frame["asset_kind"] == "STOCK"].sort_values(["rank", "symbol"])
    return tuple(stocks["symbol"].astype(str).head(limit))


def registered_templates(
    cfg: EdgeStackConfig, *, through: date | None = None
) -> tuple[CampaignTemplate, ...]:
    if through is None:
        complete = _last_complete_session()
        month_start = complete.replace(day=1)
        end = TradingCalendar().prev_session(month_start).date()
    else:
        end = through
    cohort = end.strftime("%Y%m%d")
    daily_id = f"daily-liquid-multifamily-{cohort}-v1"
    intraday_id = f"intraday-opening-multifamily-{cohort}-v1"
    event_id = f"event-context-multifamily-{cohort}-v1"
    daily_start = date(2011, 1, 3)
    intraday_start = date(2016, 1, 4)
    liquid_stocks = _monthly_liquid_stocks(cfg, end)
    liquid = ("SPY", "QQQ", "IWM", "DIA", *liquid_stocks)
    daily_symbols = (*liquid, "TLT", "SHY", "GLD")
    daily_price_symbols = (*daily_symbols, *INVERSE_ETFS)
    daily_candidates = (
        *_grid("trend_dip", liquid, lookback=(100, 200), dip=(-0.01, -0.02), hold=(3, 5)),
        *_grid("breakout", liquid, lookback=(20, 60, 120), hold=(5, 10)),
        *_grid("mean_reversion", liquid, lookback=(2, 5), threshold=(-0.02, -0.04), hold=(1, 3)),
        *_grid("relative_strength", liquid, lookback=(20, 60), threshold=(0.0, 0.03), hold=(5, 10)),
        *_grid("volatility_regime", ("SPY", "QQQ"), lookback=(10, 20, 60), quantile=(0.25, 0.5)),
        *_grid("calendar", liquid, weekday=(0, 4), hold=(1,)),
        *_grid("inverse_trend", INVERSE_ETFS, lookback=(20, 60, 120), hold=(3, 5)),
    )
    from edgestack.research.opening_fade import load_campaign_config

    opening_config_path = Path("configs/opening_fade.yaml")
    opening_previously_accessed = False
    if opening_config_path.exists():
        opening_config = load_campaign_config(opening_config_path)
        opening_previously_accessed = opening_config.previously_accessed_start <= end
        intraday_candidates = tuple(
            item.model_dump(mode="json") for item in opening_config.candidates
        )
    else:
        intraday_candidates = tuple(
            {"rule": rule, "symbol": symbol, "opening_minutes": opening, "exit": exit_time}
            for rule, symbol, opening, exit_time in product(
                (
                    "opening_fade",
                    "opening_continuation",
                    "opening_range_breakout",
                    "vwap_failure",
                    "vwap_reclaim",
                    "cross_index_confirmation",
                ),
                ALL_RESEARCH_ETFS,
                (5, 15, 30),
                ("10:30", "11:00"),
            )
        )
    event_candidates = tuple(
        {"rule": rule, "horizon": horizon}
        for rule, horizon in product(
            ("earnings_drift", "macro_release", "short_sale_volume_reversal"),
            (1, 5, 20),
        )
    )
    if (
        max(len(daily_candidates), len(intraday_candidates), len(event_candidates))
        > cfg.research.max_trials_per_campaign
    ):
        raise ValueError("registered campaign exceeds research.max_trials_per_campaign")
    return (
        CampaignTemplate(
            template_id=daily_id,
            name="Daily liquid ETF and stock edge search",
            family="daily_multi_family",
            evaluator="daily",
            candidate_family=daily_candidates,
            requirements=(
                DataRequirement(
                    campaign_id=daily_id,
                    dataset="prices",
                    symbols=daily_price_symbols,
                    frequency="1d",
                    start=daily_start,
                    end=end,
                    required_observations=756,
                    preferred_providers=("alpaca", "yahoo"),
                ),
                DataRequirement(
                    campaign_id=daily_id,
                    dataset="macro",
                    symbols=("DGS3MO",),
                    frequency="daily",
                    start=daily_start,
                    end=end,
                    required_observations=756,
                    preferred_providers=("fred",),
                ),
            ),
            previously_accessed=False,
        ),
        CampaignTemplate(
            template_id=intraday_id,
            name="Opening sequence and VWAP edge search",
            family="intraday_opening_multi_family",
            evaluator="opening_fade",
            candidate_family=intraday_candidates,
            requirements=(
                DataRequirement(
                    campaign_id=intraday_id,
                    dataset="intraday",
                    symbols=ALL_RESEARCH_ETFS,
                    frequency="1m",
                    start=intraday_start,
                    end=end,
                    required_observations=252 * 390,
                    preferred_providers=("alpaca", "yahoo"),
                ),
            ),
            previously_accessed=opening_previously_accessed,
        ),
        CampaignTemplate(
            template_id=event_id,
            name="Earnings, macro and short-sale-volume search",
            family="event_context_multi_family",
            evaluator="events",
            candidate_family=event_candidates,
            requirements=(
                DataRequirement(
                    campaign_id=event_id,
                    dataset="prices",
                    symbols=daily_symbols,
                    frequency="1d",
                    start=daily_start,
                    end=end,
                    required_observations=756,
                    preferred_providers=("alpaca", "yahoo"),
                ),
                DataRequirement(
                    campaign_id=event_id,
                    dataset="macro",
                    symbols=("DGS3MO", "VIXCLS", "DFF"),
                    frequency="daily",
                    start=daily_start,
                    end=end,
                    required_observations=756,
                    preferred_providers=("fred",),
                ),
                DataRequirement(
                    campaign_id=event_id,
                    dataset="earnings",
                    symbols=liquid_stocks,
                    frequency="event",
                    start=daily_start,
                    end=end,
                    required_observations=40,
                    preferred_providers=("sec_edgar",),
                ),
                DataRequirement(
                    campaign_id=event_id,
                    dataset="short_sale_volume",
                    symbols=liquid_stocks,
                    frequency="1d",
                    start=date(2018, 8, 1),
                    end=end,
                    required_observations=252,
                    preferred_providers=("finra",),
                ),
            ),
            previously_accessed=False,
        ),
    )


def _trial_id(template: CampaignTemplate, index: int, parameters: dict[str, Any]) -> str:
    return stable_hash(
        {"campaign_id": template.template_id, "index": index, "parameters": parameters}
    )[:24]


def seed_registered_campaigns(
    store: ResearchStore,
    cfg: EdgeStackConfig,
    *,
    through: date | None = None,
) -> tuple[CampaignTemplate, ...]:
    templates = registered_templates(cfg, through=through)
    existing = {item.campaign_id: item for item in store.campaigns()}
    now = datetime.now(UTC)
    for template in templates:
        current = existing.get(template.template_id)
        if current is None:
            current = CampaignSummaryV1(
                campaign_id=template.template_id,
                manifest_hash=template.manifest_hash,
                name=template.name,
                family=template.family,
                lifecycle=CampaignLifecycle.NEEDS_DATA,
                created_at=now,
                updated_at=now,
                trial_count=len(template.candidate_family),
                data_requirements=tuple(item.requirement_id for item in template.requirements),
                next_action="Assess coverage and enqueue every satisfiable evidence gap.",
                previously_accessed=template.previously_accessed,
            )
            store.upsert_campaign(current)
        elif current.manifest_hash != template.manifest_hash:
            # Never mutate a campaign after evaluation; a changed template needs a new id.
            raise ValueError(
                f"registered template {template.template_id} changed without a version bump"
            )
        trials = []
        for index, parameters in enumerate(template.candidate_family):
            horizon = int(parameters.get("hold", parameters.get("horizon", 1)))
            trials.append(
                TrialRecordV2(
                    trial_id=_trial_id(template, index, parameters),
                    experiment_id=template.template_id,
                    kind=TrialKind.STANDALONE,
                    family=template.family,
                    horizon_sessions=max(1, horizon),
                    parameters=parameters,
                    status=TrialStatus.REGISTERED,
                )
            )
        with store.catalog.connect() as con:
            con.executemany(
                "INSERT OR IGNORE INTO trial_ledger_v2 VALUES (?, ?, ?, ?, ?)",
                [
                    [
                        trial.trial_id,
                        template.template_id,
                        now,
                        trial.status.value,
                        trial.model_dump_json(),
                    ]
                    for trial in trials
                ],
            )
    return templates
