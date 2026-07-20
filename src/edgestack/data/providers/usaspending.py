"""USAspending monthly prime-contract obligations for alternative-data research.

The public search API is useful for hypothesis testing, but it is a current,
revised view of historical awards.  It is therefore deliberately advertised
as non-point-in-time and cannot by itself support promotion.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from edgestack.data.catalog import atomic_write_bytes
from edgestack.data.providers.base import ProviderMetadata
from edgestack.exceptions import ProviderError
from edgestack.recommendation.hashing import canonical_json, stable_hash

_URL = "https://api.usaspending.gov/api/v2/search/spending_over_time/"


def _fiscal_month_end(fiscal_year: int, fiscal_month: int) -> pd.Timestamp:
    """Convert federal fiscal month 1=October through 12=September."""
    if not 1 <= fiscal_month <= 12:
        raise ProviderError(f"invalid USAspending fiscal month: {fiscal_month}")
    if fiscal_month <= 3:
        calendar_year = fiscal_year - 1
        calendar_month = fiscal_month + 9
    else:
        calendar_year = fiscal_year
        calendar_month = fiscal_month - 3
    return pd.Timestamp(calendar_year, calendar_month, 1) + pd.offsets.MonthEnd(0)


class USAspendingContractProvider:
    """Fetch a frozen snapshot of monthly obligations by recipient search text."""

    def __init__(
        self,
        *,
        raw_dir: Path,
        timeout: float = 60.0,
        max_retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        self.raw_dir = raw_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.last_raw_hashes: tuple[str, ...] = ()
        self.metadata = ProviderMetadata(
            name="usaspending",
            kind="government_contract_obligations",
            supported_fields=(
                "symbol",
                "event_time",
                "recipient_query",
                "contract_obligations",
                "source_version",
            ),
            frequencies=("monthly",),
            is_point_in_time=False,
            adjustment="raw",
            publication_delay="award actions are updated daily; historical revision lag varies",
            limitations=(
                "the search API is a current revised snapshot, not a historical vintage archive",
                "recipient text can include subsidiaries and requires point-in-time parent mapping",
                "monthly aggregation omits contract type and award-level implementation detail",
                "negative obligations are valid de-obligations and are retained",
            ),
        )

    @staticmethod
    def _payload(
        query: str,
        start: date,
        end: date,
        award_type_codes: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "group": "month",
            "filters": {
                "time_period": [{"start_date": start.isoformat(), "end_date": end.isoformat()}],
                "award_type_codes": list(award_type_codes),
                "recipient_search_text": [query],
            },
        }

    def _request(self, payload: dict[str, Any], *, refresh: bool) -> tuple[dict[str, Any], str]:
        request_hash = stable_hash({"url": _URL, "payload": payload})
        request_path = self.raw_dir / "requests" / f"{request_hash}.json"
        if request_path.exists() and not refresh:
            raw = request_path.read_bytes()
            try:
                result = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ProviderError("cached USAspending response is malformed") from exc
            return result, hashlib.sha256(raw).hexdigest()

        headers = {"User-Agent": "EdgeStack research-only alternative-data client/0.1"}
        response: requests.Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    _URL,
                    json=payload,
                    timeout=self.timeout,
                    headers=headers,
                )
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise ProviderError("USAspending request failed (network)") from exc
                time.sleep(min(8.0, 0.5 * 2**attempt))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= self.max_retries:
                    kind = "rate_limit" if response.status_code == 429 else "server"
                    raise ProviderError(f"USAspending request failed ({kind})")
                retry_after = response.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else 0.5 * 2**attempt
                )
                time.sleep(min(30.0, delay))
                continue
            if response.status_code >= 400:
                raise ProviderError(f"USAspending request failed (HTTP {response.status_code})")
            break
        if response is None:  # pragma: no cover - defensive guard
            raise ProviderError("USAspending request produced no response")
        try:
            result = response.json()
        except ValueError as exc:
            raise ProviderError("USAspending response was malformed") from exc
        if not isinstance(result, dict) or not isinstance(result.get("results"), list):
            raise ProviderError("USAspending response did not contain monthly results")
        raw = canonical_json(result)
        digest = hashlib.sha256(raw).hexdigest()
        atomic_write_bytes(self.raw_dir / "content" / f"{digest}.json", raw)
        atomic_write_bytes(request_path, raw)
        return result, digest

    def fetch_monthly_contract_obligations(
        self,
        recipient_map: dict[str, tuple[str, ...]],
        start: date,
        end: date,
        *,
        award_type_codes: tuple[str, ...] = ("A", "B", "C", "D"),
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return one deterministic aggregate row per listed parent and month."""
        alias_frames: list[pd.DataFrame] = []
        hashes: list[str] = []
        for symbol in sorted(recipient_map):
            aliases = tuple(dict.fromkeys(item.strip() for item in recipient_map[symbol]))
            if not aliases:
                raise ProviderError(f"USAspending mapping for {symbol} has no recipient query")
            for query in aliases:
                payload = self._payload(query, start, end, award_type_codes)
                result, digest = self._request(payload, refresh=refresh)
                hashes.append(digest)
                rows: list[dict[str, Any]] = []
                for item in result["results"]:
                    period = item.get("time_period") or {}
                    fiscal_year = period.get("fiscal_year")
                    fiscal_month = period.get("month")
                    try:
                        event_time = _fiscal_month_end(
                            int(str(fiscal_year)), int(str(fiscal_month))
                        )
                    except (TypeError, ValueError) as exc:
                        raise ProviderError(
                            "USAspending response has an invalid fiscal period"
                        ) from exc
                    amount = item.get("Contract_Obligations", item.get("aggregated_amount"))
                    rows.append(
                        {
                            "symbol": symbol.upper(),
                            "event_time": event_time,
                            "recipient_query": query,
                            "contract_obligations": (
                                float(amount) if amount is not None else float("nan")
                            ),
                            "source_version": digest,
                        }
                    )
                frame = pd.DataFrame(rows)
                if not frame.empty:
                    frame = frame.loc[
                        frame["event_time"].between(pd.Timestamp(start), pd.Timestamp(end))
                    ]
                alias_frames.append(frame)
        self.last_raw_hashes = tuple(sorted(set(hashes)))
        if not alias_frames:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "event_time",
                    "contract_obligations",
                    "recipient_queries",
                    "source_versions",
                    "missingness_reason",
                ]
            )
        alias_panel = pd.concat(alias_frames, ignore_index=True)
        complete_months = pd.date_range(
            pd.Timestamp(start) + pd.offsets.MonthEnd(0),
            pd.Timestamp(end) + pd.offsets.MonthEnd(0),
            freq="ME",
        )
        output_rows: list[dict[str, Any]] = []
        for symbol in sorted(recipient_map):
            mapped_aliases = tuple(dict.fromkeys(recipient_map[symbol]))
            selected = alias_panel.loc[alias_panel["symbol"] == symbol].copy()
            for event_time in complete_months:
                monthly = selected.loc[selected["event_time"] == event_time]
                observed = int(monthly["contract_obligations"].notna().sum())
                complete = observed == len(mapped_aliases)
                output_rows.append(
                    {
                        "symbol": symbol,
                        "event_time": event_time,
                        "contract_obligations": (
                            float(monthly["contract_obligations"].sum())
                            if complete
                            else float("nan")
                        ),
                        "recipient_queries": "|".join(mapped_aliases),
                        "source_versions": "|".join(
                            sorted(set(monthly["source_version"].astype(str)))
                        ),
                        "missingness_reason": "" if complete else "source_period_missing",
                    }
                )
        return (
            pd.DataFrame(output_rows).sort_values(["event_time", "symbol"]).reset_index(drop=True)
        )
