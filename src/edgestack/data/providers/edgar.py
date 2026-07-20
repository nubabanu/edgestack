"""Immutable SEC EDGAR bulk snapshots for submissions and company facts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from edgestack.data.catalog import atomic_write_bytes
from edgestack.data.providers.base import ProviderMetadata
from edgestack.exceptions import ProviderError

_BULK_URLS = {
    "submissions": "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip",
    "companyfacts": "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip",
}
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_ROOT = "https://data.sec.gov/submissions"


@dataclass(frozen=True)
class EdgarBulkSnapshot:
    dataset: str
    content_hash: str
    path: Path
    retrieved_at: datetime
    byte_count: int


class SecEdgarBulkProvider:
    def __init__(
        self,
        *,
        raw_dir: Path,
        user_agent: str,
        timeout: float = 120.0,
        session: requests.Session | None = None,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("SEC EDGAR requires a declared user agent")
        self.raw_dir = raw_dir
        self.user_agent = user_agent
        self.timeout = timeout
        self.session = session or requests.Session()
        self.last_raw_hashes: tuple[str, ...] = ()
        self.metadata = ProviderMetadata(
            name="sec_edgar_bulk",
            kind="fundamentals_and_events",
            supported_fields=("submissions", "companyfacts"),
            frequencies=("event", "quarterly"),
            is_point_in_time=True,
            adjustment="raw",
            publication_delay="real-time API; nightly bulk snapshot",
            limitations=(
                "issuer tagging quality varies",
                "standardized XBRL coverage begins primarily after 2009",
                "ticker-to-CIK history requires explicit point-in-time mapping",
            ),
        )

    def fetch_bulk_snapshot(self, dataset: str) -> EdgarBulkSnapshot:
        try:
            url = _BULK_URLS[dataset]
        except KeyError:
            raise ProviderError(f"unknown EDGAR bulk dataset {dataset!r}") from None
        try:
            response = self.session.get(
                url,
                headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ProviderError("SEC EDGAR bulk request failed (network)") from exc
        if response.status_code == 429:
            raise ProviderError("SEC EDGAR bulk request failed (rate_limit)")
        if response.status_code >= 400:
            raise ProviderError(f"SEC EDGAR bulk request failed (HTTP {response.status_code})")
        payload = response.content
        digest = hashlib.sha256(payload).hexdigest()
        path = self.raw_dir / dataset / f"{digest}.zip"
        atomic_write_bytes(path, payload)
        self.last_raw_hashes = (*self.last_raw_hashes, digest)
        return EdgarBulkSnapshot(
            dataset=dataset,
            content_hash=digest,
            path=path,
            retrieved_at=datetime.now(UTC),
            byte_count=len(payload),
        )

    def _json(self, url: str) -> dict:
        try:
            response = self.session.get(
                url,
                headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ProviderError("SEC EDGAR API request failed (network)") from exc
        if response.status_code == 429:
            raise ProviderError("SEC EDGAR API request failed (rate_limit)")
        if response.status_code >= 400:
            raise ProviderError(f"SEC EDGAR API request failed (HTTP {response.status_code})")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError("SEC EDGAR API response was malformed") from exc
        if not isinstance(payload, dict):
            raise ProviderError("SEC EDGAR API response was not an object")
        raw = response.content
        digest = hashlib.sha256(raw).hexdigest()
        atomic_write_bytes(self.raw_dir / "api" / f"{digest}.json", raw)
        self.last_raw_hashes = (*self.last_raw_hashes, digest)
        return payload

    def fetch_earnings_events(
        self,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return timestamped 8-K item 2.02 filings for the requested tickers."""
        tickers = self._json(_TICKERS_URL)
        cik_by_symbol = {
            str(item.get("ticker", "")).upper(): int(item["cik_str"])
            for item in tickers.values()
            if isinstance(item, dict) and item.get("ticker") and item.get("cik_str")
        }
        rows = []
        for symbol in dict.fromkeys(item.upper() for item in symbols):
            cik = cik_by_symbol.get(symbol)
            if cik is None:
                continue
            submission = self._json(f"{_SUBMISSIONS_ROOT}/CIK{cik:010d}.json")
            frames = [pd.DataFrame(submission.get("filings", {}).get("recent", {}))]
            for extra in submission.get("filings", {}).get("files", []):
                name = extra.get("name") if isinstance(extra, dict) else None
                if name:
                    frames.append(pd.DataFrame(self._json(f"{_SUBMISSIONS_ROOT}/{name}")))
            filings = pd.concat(frames, ignore_index=True)
            if filings.empty or not {"form", "items", "acceptanceDateTime"}.issubset(filings):
                continue
            accepted = pd.to_datetime(filings["acceptanceDateTime"], utc=True, errors="coerce")
            mask = (
                filings["form"].isin(["8-K", "8-K/A"])
                & filings["items"].astype(str).str.contains("2.02", na=False)
                & accepted.between(pd.Timestamp(start), pd.Timestamp(end))
            )
            for index in filings.index[mask]:
                rows.append(
                    {
                        "symbol": symbol,
                        "accepted_at": accepted.loc[index],
                        "form": str(filings.loc[index, "form"]),
                        "report_period": pd.to_datetime(
                            filings.loc[index, "reportDate"], errors="coerce"
                        )
                        if "reportDate" in filings
                        else pd.NaT,
                        "source": "SEC_EDGAR_8K_ITEM_2_02",
                        "retrieved_at": datetime.now(UTC),
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "symbol",
                "accepted_at",
                "form",
                "report_period",
                "source",
                "retrieved_at",
            ],
        )
