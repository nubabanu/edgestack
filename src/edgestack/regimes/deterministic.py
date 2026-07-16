"""Deterministic, transparent regime labels.

Thresholds are fixed domain constants (documented below), not fitted values,
so labels are identical in-sample and live and cannot leak.

- market trend: benchmark vs its 200-SMA (+/-2% dead zone -> SIDEWAYS)
- market volatility: annualized 20-session benchmark vol (12% / 20% cuts)
- stock trend: above/below its own 200-SMA
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from edgestack.types import Edge, RegimeContext

TREND_DEAD_ZONE = 0.02
VOL_LOW, VOL_HIGH = 0.12, 0.20


@dataclass(frozen=True)
class RegimeLabels:
    market_trend: str  # UPTREND | DOWNTREND | SIDEWAYS | UNKNOWN
    market_volatility: str  # LOW | MEDIUM | HIGH | UNKNOWN
    stock_trend: str  # UP | DOWN | UNKNOWN

    @property
    def market(self) -> str:
        return f"{self.market_trend}_{self.market_volatility}_VOLATILITY"


def label_row(row: pd.Series) -> RegimeLabels:
    """Regime labels from one symbol's feature row at signal time."""
    trend = row.get("bench_trend_200")
    if pd.isna(trend):
        market_trend = "UNKNOWN"
    elif trend > TREND_DEAD_ZONE:
        market_trend = "UPTREND"
    elif trend < -TREND_DEAD_ZONE:
        market_trend = "DOWNTREND"
    else:
        market_trend = "SIDEWAYS"

    vol = row.get("bench_vol_20")
    if pd.isna(vol):
        market_vol = "UNKNOWN"
    elif vol < VOL_LOW:
        market_vol = "LOW"
    elif vol > VOL_HIGH:
        market_vol = "HIGH"
    else:
        market_vol = "MEDIUM"

    above = row.get("above_sma200")
    stock_trend = "UNKNOWN" if pd.isna(above) else ("UP" if above >= 0.5 else "DOWN")
    return RegimeLabels(market_trend, market_vol, stock_trend)


def edge_regime_similarity(edge: Edge, labels: RegimeLabels) -> float:
    """How well the current regime matches where the edge historically worked.

    1.0  — edge was profitable in the current market-trend regime;
    0.4  — no regime breakdown recorded (unknown applicability);
    0.1  — edge historically LOST money in the current regime.
    """
    perf = edge.robustness.regime_performance
    key = "market_uptrend" if labels.market_trend == "UPTREND" else "market_downtrend"
    if labels.market_trend == "SIDEWAYS" or key not in perf:
        return 0.4
    return 1.0 if perf[key] > 0 else 0.1


def regime_context(row: pd.Series, similarity: float) -> RegimeContext:
    labels = label_row(row)
    return RegimeContext(
        market=labels.market,
        volatility=labels.market_volatility,
        stock_trend=labels.stock_trend,
        similarity_score=max(0.0, min(1.0, similarity)),
    )
