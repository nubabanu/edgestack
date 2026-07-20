"""FRED/ALFRED initial-release and vintage-aware macro observations."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from edgestack.data.catalog import atomic_write_bytes
from edgestack.data.providers.base import MacroDataProvider, ProviderMetadata
from edgestack.exceptions import ProviderError

_URL = "https://api.stlouisfed.org/fred/series/observations"


class FredVintageProvider(MacroDataProvider):
    def __init__(
        self,
        api_key: str | None,
        *,
        raw_dir: Path,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = api_key
        self.raw_dir = raw_dir
        self.timeout = timeout
        self.session = session or requests.Session()
        self.last_raw_hashes: tuple[str, ...] = ()
        self.metadata = ProviderMetadata(
            name="fred_alfred",
            kind="macro",
            supported_fields=(
                "series_id",
                "observation_date",
                "value",
                "realtime_start",
                "realtime_end",
            ),
            frequencies=("daily", "weekly", "monthly", "quarterly"),
            is_point_in_time=True,
            adjustment="raw",
            publication_delay="series-specific",
            limitations=(
                "an API key is required",
                "release times and revision schedules differ by series",
                "initial-release values may be missing for older observations",
            ),
        )

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def fetch_series(self, series_ids: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
        if not self._api_key:
            raise ProviderError("FRED is unconfigured; set EDGESTACK_FRED_API_KEY")
        rows: list[dict[str, Any]] = []
        raw_hashes: list[str] = []
        for series_id in dict.fromkeys(item.strip().upper() for item in series_ids if item.strip()):
            params = {
                "series_id": series_id,
                "api_key": self._api_key,
                "file_type": "json",
                "observation_start": start.isoformat(),
                "observation_end": end.isoformat(),
                "realtime_start": "1776-07-04",
                "realtime_end": "9999-12-31",
                "output_type": "4",
                "limit": "100000",
            }
            try:
                response = self.session.get(_URL, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                raise ProviderError("FRED request failed (network)") from exc
            if response.status_code == 429:
                raise ProviderError("FRED request failed (rate_limit)")
            if response.status_code in {400, 401}:
                raise ProviderError("FRED request failed (authentication_or_query)")
            if response.status_code >= 400:
                raise ProviderError(f"FRED request failed (HTTP {response.status_code})")
            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderError("FRED response was malformed") from exc
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            digest = hashlib.sha256(raw).hexdigest()
            atomic_write_bytes(self.raw_dir / f"{digest}.json", raw)
            raw_hashes.append(digest)
            for item in payload.get("observations", []):
                value = item.get("value")
                rows.append(
                    {
                        "series_id": series_id,
                        "observation_date": pd.to_datetime(item.get("date")),
                        "value": None if value in {None, "."} else float(value),
                        "realtime_start": pd.to_datetime(item.get("realtime_start")),
                        "realtime_end": pd.to_datetime(item.get("realtime_end")),
                        "retrieved_at": datetime.now(UTC),
                    }
                )
        self.last_raw_hashes = tuple(raw_hashes)
        return pd.DataFrame(
            rows,
            columns=[
                "series_id",
                "observation_date",
                "value",
                "realtime_start",
                "realtime_end",
                "retrieved_at",
            ],
        )
