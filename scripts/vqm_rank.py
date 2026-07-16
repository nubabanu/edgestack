"""Value-Quality-Momentum composite from TODAY's fundamentals snapshot.

CURRENT-SNAPSHOT ranking only — Yahoo's free feed has no historical vintages,
so this composite is NOT backtestable here. Its use is confined to ranking
today's 12-month candidates, cross-checked against the (backtested)
cross-sectional momentum/low-vol evidence.

Pillars (each = mean of cross-sectional percentile ranks, missing-safe):
  VALUE    cheap forwardPE, priceToBook, enterpriseToEbitda; FCF yield high
  QUALITY  high ROE/ROA, high margins, low debtToEquity, currentRatio
  GROWTH   revenueGrowth, earningsGrowth
  MOMENTUM 52WeekChange, closeness to 52w high
  SENTIMENT analyst upside (targetMean/price - 1), recommendationMean (1=buy)

Composite = mean of the five pillar scores. Also flags: high short interest,
extreme leverage, negative FCF.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from edgestack.data.catalog import atomic_write_bytes


def main() -> int:
    with open("artifacts/fundamentals_snapshot.json", encoding="utf-8") as handle:
        snap = json.load(handle)
    df = pd.DataFrame.from_dict(snap["data"], orient="index")
    df["fcf_yield"] = df["freeCashflow"] / df["marketCap"]
    df["upside"] = df["targetMeanPrice"] / df["regularMarketPrice"] - 1.0
    df["pct_of_52w_high"] = df["regularMarketPrice"] / df["fiftyTwoWeekHigh"]

    def rank(col: str, ascending: bool) -> pd.Series:
        s = df[col]
        # guard nonsensical values (negative PE means losses -> worst bucket)
        if col in ("forwardPE", "trailingPE", "enterpriseToEbitda", "priceToBook"):
            s = s.where(s > 0)
        return s.rank(pct=True, ascending=ascending)

    pillars = {
        "VALUE": [
            rank("forwardPE", False),
            rank("priceToBook", False),
            rank("enterpriseToEbitda", False),
            rank("fcf_yield", True),
        ],
        "QUALITY": [
            rank("returnOnEquity", True),
            rank("returnOnAssets", True),
            rank("operatingMargins", True),
            rank("profitMargins", True),
            rank("debtToEquity", False),
            rank("currentRatio", True),
        ],
        "GROWTH": [rank("revenueGrowth", True), rank("earningsGrowth", True)],
        "MOMENTUM": [rank("52WeekChange", True), rank("pct_of_52w_high", True)],
        "SENTIMENT": [rank("upside", True), rank("recommendationMean", False)],
    }
    for name, cols in pillars.items():
        df[name] = pd.concat(cols, axis=1).mean(axis=1)
    df["COMPOSITE"] = df[list(pillars)].mean(axis=1)

    df["flags"] = ""
    df.loc[df["shortPercentOfFloat"] > 0.10, "flags"] += "HIGH_SHORT;"
    df.loc[df["debtToEquity"] > 300, "flags"] += "LEVERED;"
    df.loc[df["fcf_yield"] < 0, "flags"] += "NEG_FCF;"

    cols = [
        "shortName",
        "COMPOSITE",
        "VALUE",
        "QUALITY",
        "GROWTH",
        "MOMENTUM",
        "SENTIMENT",
        "forwardPE",
        "fcf_yield",
        "returnOnEquity",
        "revenueGrowth",
        "52WeekChange",
        "upside",
        "beta",
        "flags",
    ]
    ranked = df.sort_values("COMPOSITE", ascending=False)[cols]

    pd.set_option("display.width", 250)
    print("=== TOP 20 composite (value+quality+growth+momentum+sentiment) ===")
    print(ranked.head(20).round(3).to_string())
    print("\n=== BOTTOM 15 composite (avoid candidates) ===")
    print(ranked.tail(15).round(3).to_string())

    payload = {
        "as_of": snap["as_of"],
        "top30": ranked.head(30)
        .round(4)
        .reset_index()
        .rename(columns={"index": "symbol"})
        .to_dict("records"),
        "bottom20": ranked.tail(20)
        .round(4)
        .reset_index()
        .rename(columns={"index": "symbol"})
        .to_dict("records"),
    }
    atomic_write_bytes(
        Path("artifacts") / "vqm_rank.json", json.dumps(payload, default=str).encode()
    )
    print("\nsaved -> artifacts/vqm_rank.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
