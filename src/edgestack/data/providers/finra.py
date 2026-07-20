"""FINRA Reg SHO off-exchange daily short-sale volume.

This is deliberately named short-sale volume. It is not a borrow feed and it
does not represent outstanding short interest.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from edgestack.data.catalog import atomic_write_bytes
from edgestack.data.providers.base import ProviderMetadata
from edgestack.exceptions import ProviderError

_URL = "https://api.finra.org/data/group/otcMarket/name/regShoDaily"


class FinraShortSaleVolumeProvider:
    def __init__(
        self,
        *,
        raw_dir: Path,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.raw_dir = raw_dir
        self.timeout = timeout
        self.session = session or requests.Session()
        self.last_raw_hashes: tuple[str, ...] = ()
        self.metadata = ProviderMetadata(
            name="finra_reg_sho_daily",
            kind="short_sale_volume",
            supported_fields=("short_volume", "short_exempt_volume", "total_volume"),
            frequencies=("1d",),
            is_point_in_time=True,
            adjustment="raw",
            publication_delay="same day after close",
            limitations=(
                "off-exchange publicly disseminated trades only",
                "not consolidated exchange short-sale volume",
                "not short interest and not a borrow-availability signal",
                "the query API exposes a rolling historical window",
            ),
        )

    def fetch_daily_short_sale_volume(
        self,
        symbols: tuple[str, ...],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        wanted = tuple(dict.fromkeys(symbol.upper() for symbol in symbols))
        payload: dict[str, Any] = {
            "limit": 5000,
            "offset": 0,
            "fields": [
                "tradeReportDate",
                "securitiesInformationProcessorSymbolIdentifier",
                "shortParQuantity",
                "shortExemptParQuantity",
                "totalParQuantity",
            ],
            "dateRangeFilters": [
                {
                    "fieldName": "tradeReportDate",
                    "startDate": start.isoformat(),
                    "endDate": end.isoformat(),
                }
            ],
            "domainFilters": [
                {
                    "fieldName": "securitiesInformationProcessorSymbolIdentifier",
                    "values": list(wanted),
                }
            ],
        }
        rows: list[dict[str, Any]] = []
        raw_hashes: list[str] = []
        while True:
            try:
                response = self.session.post(
                    _URL,
                    json=payload,
                    headers={"Accept": "application/json"},
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                raise ProviderError("FINRA short-sale volume request failed (network)") from exc
            if response.status_code == 429:
                raise ProviderError("FINRA short-sale volume request failed (rate_limit)")
            if response.status_code >= 400:
                raise ProviderError(
                    f"FINRA short-sale volume request failed (HTTP {response.status_code})"
                )
            if response.status_code == 204:
                result: Any = []
            else:
                try:
                    result = response.json()
                except ValueError as exc:
                    raise ProviderError("FINRA short-sale volume response was malformed") from exc
            raw = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
            digest = hashlib.sha256(raw).hexdigest()
            atomic_write_bytes(self.raw_dir / f"{digest}.json", raw)
            raw_hashes.append(digest)
            page = result if isinstance(result, list) else []
            rows.extend(item for item in page if isinstance(item, dict))
            if len(page) < int(payload["limit"]):
                break
            payload["offset"] = int(payload["offset"]) + len(page)
        self.last_raw_hashes = tuple(raw_hashes)
        output = pd.DataFrame(
            {
                "symbol": [
                    str(item.get("securitiesInformationProcessorSymbolIdentifier", "")).upper()
                    for item in rows
                ],
                "date": [
                    pd.to_datetime(str(item.get("tradeReportDate", "")), errors="coerce")
                    for item in rows
                ],
                "short_sale_volume": [float(item.get("shortParQuantity") or 0) for item in rows],
                "short_exempt_volume": [
                    float(item.get("shortExemptParQuantity") or 0) for item in rows
                ],
                "total_reported_volume": [
                    float(item.get("totalParQuantity") or 0) for item in rows
                ],
                "retrieved_at": datetime.now(UTC),
            }
        )
        if output.empty:
            return output
        return (
            output.groupby(["symbol", "date"], as_index=False, observed=True)
            .agg(
                short_sale_volume=("short_sale_volume", "sum"),
                short_exempt_volume=("short_exempt_volume", "sum"),
                total_reported_volume=("total_reported_volume", "sum"),
                retrieved_at=("retrieved_at", "max"),
            )
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )
