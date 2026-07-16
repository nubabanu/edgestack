"""DGS3MO ingestion and calendar-day cash/funding accounting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from edgestack.exceptions import DataError, ProviderError

DGS3MO_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
FUNDING_STRESS_SPREADS_BPS = (200.0, 400.0, 800.0)


@dataclass(frozen=True)
class FundingRateObservation:
    series_id: str
    as_of: date
    annualized_rate: float
    fetched_at: datetime
    source: str = DGS3MO_CSV_URL

    def __post_init__(self) -> None:
        if self.series_id != "DGS3MO":
            raise DataError("funding observation must use DGS3MO")
        if not np.isfinite(self.annualized_rate) or self.annualized_rate < 0:
            raise DataError("DGS3MO rate must be finite and non-negative")

    def to_json(self) -> dict[str, str | float]:
        return {
            "series_id": self.series_id,
            "as_of": self.as_of.isoformat(),
            "annualized_rate": self.annualized_rate,
            "fetched_at": self.fetched_at.isoformat(),
            "source": self.source,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> FundingRateObservation:
        return cls(
            series_id=str(payload["series_id"]),
            as_of=date.fromisoformat(str(payload["as_of"])),
            annualized_rate=float(str(payload["annualized_rate"])),
            fetched_at=datetime.fromisoformat(str(payload["fetched_at"])),
            source=str(payload.get("source", DGS3MO_CSV_URL)),
        )


def load_cached_funding_rate(path: Path) -> FundingRateObservation | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("cache payload is not an object")
        return FundingRateObservation.from_json(payload)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError(f"invalid funding-rate cache {path}: {exc}") from exc


def fetch_dgs3mo(
    cache_path: Path,
    *,
    now: datetime | None = None,
    timeout_seconds: float = 30.0,
    session: requests.Session | None = None,
) -> FundingRateObservation:
    """Fetch the latest free FRED DGS3MO observation, with a daily local cache."""
    current = now or datetime.now(UTC)
    cached = load_cached_funding_rate(cache_path)
    if cached is not None and cached.fetched_at.date() == current.date():
        return cached
    client = session or requests.Session()
    try:
        response = client.get(DGS3MO_CSV_URL, timeout=timeout_seconds)
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
    except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
        if cached is not None:
            return cached
        raise ProviderError(f"could not fetch DGS3MO: {exc}") from exc
    if not {"DATE", "DGS3MO"}.issubset(frame.columns):
        raise ProviderError("DGS3MO response is missing DATE or DGS3MO")
    frame["DATE"] = pd.to_datetime(frame["DATE"], errors="coerce")
    frame["DGS3MO"] = pd.to_numeric(frame["DGS3MO"], errors="coerce")
    latest = frame.dropna(subset=["DATE", "DGS3MO"]).tail(1)
    if latest.empty:
        raise ProviderError("DGS3MO response has no resolved observations")
    observation = FundingRateObservation(
        series_id="DGS3MO",
        as_of=latest.iloc[0]["DATE"].date(),
        annualized_rate=float(latest.iloc[0]["DGS3MO"]) / 100.0,
        fetched_at=current,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    temporary.write_text(
        json.dumps(observation.to_json(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(cache_path)
    return observation


def funding_business_day_age(observation_date: date, session_date: date) -> int:
    if observation_date >= session_date:
        return 0
    return int(np.busday_count(observation_date.isoformat(), session_date.isoformat()))


def funding_rate_is_stale(
    observation_date: date, session_date: date, *, max_business_days: int = 5
) -> bool:
    return funding_business_day_age(observation_date, session_date) > max_business_days


def annualized_financing(
    leverage: float, base_rate: float, funding_spread_bps: float
) -> tuple[float, float, float]:
    """Return annualized positive-cash income, borrowing cost, and their net."""
    cash_income = max(0.0, 1.0 - leverage) * base_rate
    funding_cost = max(0.0, leverage - 1.0) * (base_rate + funding_spread_bps / 10_000)
    return cash_income, funding_cost, cash_income - funding_cost


def accrue_cash_financing(
    signed_cash: float,
    *,
    base_rate: float,
    funding_spread_bps: float,
    start: date,
    end: date,
) -> float:
    """Cash flow over the actual calendar-day gap; positive is an account credit."""
    days = (end - start).days
    if days < 0:
        raise DataError("financing accrual end precedes start")
    rate = base_rate if signed_cash >= 0 else base_rate + funding_spread_bps / 10_000
    return signed_cash * rate * days / 365.0


def funding_scenarios(leverage: float, base_rate: float) -> dict[str, float]:
    return {
        f"funding_{int(spread)}bps": annualized_financing(leverage, base_rate, spread)[1]
        for spread in FUNDING_STRESS_SPREADS_BPS
    }
