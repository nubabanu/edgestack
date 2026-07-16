"""Deep-history test: ensemble4 and the calendar effects on 100 years of
US market returns (Ken French daily factor library, 1926->present).

The market index is reconstructed from daily total returns (Mkt-RF + RF);
all four ensemble families are close/adj-based, so they compute directly.
Eras deliberately include regimes our 1993+ sample never saw: the Great
Depression, WW2, the 1970s inflation.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import requests

from edgestack.data.catalog import atomic_write_bytes
from edgestack.strategies import backtest_exposure, ensemble_exposure
from edgestack.validation.advanced_tests import newey_west_alpha
from edgestack.validation.metrics import max_drawdown, sharpe_ratio

URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
ERAS = {
    "1927_1945_depression_ww2": ("1927-01-01", "1945-12-31"),
    "1946_1962_postwar": ("1946-01-01", "1962-12-31"),
    "1963_1979_inflation": ("1963-01-01", "1979-12-31"),
    "1980_1992_disinflation": ("1980-01-01", "1992-12-31"),
    "1993_2007_our_dev_era": ("1993-01-01", "2007-12-31"),
    "2008_2026_modern": ("2008-01-01", "2026-12-31"),
}
CACHE = Path("data/cache/zoo/ff_daily.parquet")


def load_french() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    r = requests.get(URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0]).decode("latin-1")
    rows = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and parts[0].isdigit() and len(parts[0]) == 8:
            rows.append((parts[0], float(parts[1]), float(parts[4])))
    df = pd.DataFrame(rows, columns=["date", "mkt_rf", "rf"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["mkt"] = (df["mkt_rf"] + df["rf"]) / 100.0
    df = df.set_index("date")[["mkt", "rf"]]
    df.to_parquet(CACHE)
    return df


def main() -> int:
    ff = load_french()
    index = (1 + ff["mkt"]).cumprod()
    bars = pd.DataFrame({"close": index, "adj": index, "open": index, "high": index, "low": index})
    ret = ff["mkt"]
    print(
        f"Ken French daily market factor: {ff.index[0].date()} -> "
        f"{ff.index[-1].date()} ({len(ff):,} sessions)"
    )

    pos = ensemble_exposure(bars)
    strat = backtest_exposure(pos, ret)

    ym = bars.index.to_period("M")
    tdom = pd.Series(bars.groupby(ym).cumcount() + 1, index=bars.index)
    tde = pd.Series(bars.groupby(ym).cumcount(ascending=False) + 1, index=bars.index)
    tom = (tdom <= 3) | (tde == 1)
    sep = bars.index.month == 9

    results = {}
    print(
        f"\n{'era':<26}{'ens Sh':>8}{'B&H Sh':>8}{'ens DD':>9}{'B&H DD':>9}"
        f"{'ToM bps':>9}{'rest':>6}{'Sep bps':>9}{'other':>7}"
    )
    for era, (lo, hi) in ERAS.items():
        s = strat.loc[lo:hi].dropna()
        b = ret.loc[s.index]
        if len(s) < 500:
            continue
        m = (bars.index >= lo) & (bars.index <= hi)
        era_row = {
            "ens_sharpe": round(sharpe_ratio(s.to_numpy()), 2),
            "bh_sharpe": round(sharpe_ratio(b.to_numpy()), 2),
            "ens_maxdd": round(max_drawdown(s.to_numpy()), 3),
            "bh_maxdd": round(max_drawdown(b.to_numpy()), 3),
            "tom_bps": round(float(ret[m & tom].mean() * 1e4), 1),
            "rest_bps": round(float(ret[m & ~tom].mean() * 1e4), 1),
            "sep_bps": round(float(ret[m & sep].mean() * 1e4), 1),
            "nonsep_bps": round(float(ret[m & ~sep].mean() * 1e4), 1),
        }
        results[era] = era_row
        print(
            f"{era:<26}{era_row['ens_sharpe']:>8.2f}{era_row['bh_sharpe']:>8.2f}"
            f"{era_row['ens_maxdd']:>9.1%}{era_row['bh_maxdd']:>9.1%}"
            f"{era_row['tom_bps']:>9.1f}{era_row['rest_bps']:>6.1f}"
            f"{era_row['sep_bps']:>9.1f}{era_row['nonsep_bps']:>7.1f}"
        )

    nw = newey_west_alpha(strat.dropna(), ret.loc[strat.dropna().index])
    pooled = {
        "alpha_ann": round(nw["alpha_ann"], 4),
        "alpha_t": round(nw["alpha_t"], 2),
        "beta": round(nw["beta"], 2),
        "eras_sharpe_ge_bh": sum(1 for r in results.values() if r["ens_sharpe"] >= r["bh_sharpe"]),
    }
    print(
        f"\nPOOLED 100y: alpha {nw['alpha_ann']:+.1%}/yr (t={nw['alpha_t']:.2f}, "
        f"beta={nw['beta']:.2f}); ensemble Sharpe >= B&H in "
        f"{pooled['eras_sharpe_ge_bh']}/{len(results)} eras"
    )
    atomic_write_bytes(
        Path("artifacts") / "french_deep_history.json",
        json.dumps({"eras": results, "pooled": pooled}, indent=1).encode(),
    )
    print("saved -> artifacts/french_deep_history.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
