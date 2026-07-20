from __future__ import annotations

from datetime import date
from typing import Any

from edgestack.data.providers.finra import FinraShortSaleVolumeProvider


class _Response:
    status_code = 200

    def json(self) -> list[dict[str, Any]]:
        return [
            {
                "tradeReportDate": "2026-07-17",
                "securitiesInformationProcessorSymbolIdentifier": "SPY",
                "shortParQuantity": 40,
                "shortExemptParQuantity": 2,
                "totalParQuantity": 100,
            }
        ]


class _Session:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}

    def post(self, _url: str, **kwargs: Any) -> _Response:
        self.headers = dict(kwargs.get("headers", {}))
        return _Response()


def test_finra_requests_json_and_labels_short_sale_volume(tmp_path) -> None:
    session = _Session()
    provider = FinraShortSaleVolumeProvider(raw_dir=tmp_path, session=session)  # type: ignore[arg-type]
    frame = provider.fetch_daily_short_sale_volume(("SPY",), date(2026, 7, 17), date(2026, 7, 17))

    assert session.headers == {"Accept": "application/json"}
    assert frame.loc[0, "short_sale_volume"] == 40
    assert frame.loc[0, "total_reported_volume"] == 100
    assert "not short interest" in provider.metadata.limitations[2]
