"""Point-in-time liquid research universe and intraday storage tiers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from edgestack.data.universe_pit import PitSP500Universe

CORE_ETFS = (
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
    "TLT",
    "IEF",
    "SHY",
    "GLD",
    "USO",
)
INVERSE_ETFS = ("SH", "PSQ", "DOG", "RWM")
ALL_RESEARCH_ETFS = CORE_ETFS + INVERSE_ETFS


@dataclass(frozen=True)
class UniverseMember:
    as_of: pd.Timestamp
    symbol: str
    rank: int
    median_dollar_volume: float
    asset_kind: str
    intraday_interval_minutes: int


def build_monthly_liquid_universe(
    panel: pd.DataFrame,
    pit: PitSP500Universe,
    *,
    stock_count: int = 100,
    one_minute_stock_count: int = 25,
    min_price: float = 5.0,
    min_median_dollar_volume: float = 5_000_000.0,
) -> tuple[UniverseMember, ...]:
    """Rank only information available through each historical month end."""
    required = {"symbol", "date", "close", "volume"}
    missing = required - set(panel)
    if missing:
        raise ValueError(f"liquid universe panel missing columns: {sorted(missing)}")
    data = panel.loc[:, sorted(required)].copy()
    data["date"] = pd.to_datetime(data["date"])
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data["dollar_volume"] = data["close"].astype(float) * data["volume"].astype(float)
    month_ends = data.groupby(data["date"].dt.to_period("M"))["date"].max().sort_values()
    output: list[UniverseMember] = []
    for as_of in month_ends:
        history = data.loc[data["date"] <= as_of].sort_values(["symbol", "date"])
        trailing = history.groupby("symbol", observed=True).tail(60)
        stats = trailing.groupby("symbol", observed=True).agg(
            close=("close", "last"),
            median_dollar_volume=("dollar_volume", "median"),
            observations=("date", "nunique"),
        )
        members = set(pit.members(as_of.date()))
        eligible = stats.loc[
            stats.index.isin(members)
            & (stats["observations"] >= 40)
            & (stats["close"] >= min_price)
            & (stats["median_dollar_volume"] >= min_median_dollar_volume)
        ].sort_values("median_dollar_volume", ascending=False)
        for rank, (symbol, row) in enumerate(eligible.head(stock_count).iterrows(), start=1):
            output.append(
                UniverseMember(
                    as_of=pd.Timestamp(as_of),
                    symbol=str(symbol),
                    rank=rank,
                    median_dollar_volume=float(row["median_dollar_volume"]),
                    asset_kind="STOCK",
                    intraday_interval_minutes=1 if rank <= one_minute_stock_count else 5,
                )
            )
        for rank, symbol in enumerate(ALL_RESEARCH_ETFS, start=1):
            output.append(
                UniverseMember(
                    as_of=pd.Timestamp(as_of),
                    symbol=symbol,
                    rank=rank,
                    median_dollar_volume=float("nan"),
                    asset_kind="INVERSE_ETF" if symbol in INVERSE_ETFS else "ETF",
                    intraday_interval_minutes=1,
                )
            )
    return tuple(output)
