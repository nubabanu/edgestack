"""Archived legacy ensemble and seasonal research, excluded from V2 actions.

The historical 4-family ensemble ("ensemble4") and the optional seasonal
multiplier — the distillation of every edge that survived this project's
campaigns (see docs/strategy-zoo.md and artifacts/strategy_zoo.json).

Validation summary (signals at close t earn session t+1, 2 bps per unit of
exposure change; splits dev 1999-2015 / val 2016-2023 / holdout 2024+):

    ensemble4 on SPY:  Sharpe 0.58/1.16/1.57 vs B&H 0.34/0.76/1.31,
                       pooled NW alpha +3.6%/yr (t=4.04), maxDD -13%
    ensemble4 on QQQ:  Sharpe 0.72/1.27/1.49 vs B&H 0.32/0.86/1.19,
                       pooled NW alpha +5.9%/yr (t=4.85), maxDD -17%

    x seasonal multiplier: raises pooled alpha (t up to 5.25) but FAILED the
    2024-26 holdout Sharpe comparison and roughly doubles max drawdown.
    It is therefore OFF by default and should be treated as unproven.

The four families (equal-weighted):
    trend_or_dip   hold above the 200-DMA; below it, hold only RSI(2)<10
                   panic days (survived independently on SPY, QQQ and XLK)
    breakout_20d   long for 10 sessions after a close above the prior
                   20-session high (survived on QQQ)
    vol_target     exposure = 10% / 20d realized vol, capped at 1.5x
    reversion_3dn  long the day after three consecutive down closes

Inputs are daily bar frames with columns open/high/low/close/adj indexed by
session date. All functions are pure and vectorized.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ENSEMBLE_FAMILIES = ("trend_or_dip", "breakout_20d", "vol_target", "reversion_3dn")
TRADE_COST = 0.0002


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def _hold_n(trigger: pd.Series, n: int) -> pd.Series:
    return trigger.astype(float).rolling(n, min_periods=1).max().fillna(0.0)


def family_positions(bars: pd.DataFrame) -> pd.DataFrame:
    """Position series (0..1.5) for each validated family, indexed like bars."""
    c = bars["close"]
    ret = bars["adj"].pct_change()
    sma200 = c.rolling(200).mean()
    rsi2 = _rsi(c, 2)
    hi20 = c.rolling(20).max()  # close-based: the validated zoo definition
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    return pd.DataFrame(
        {
            "trend_or_dip": np.maximum((c > sma200).astype(float), (rsi2 < 10).astype(float)),
            "breakout_20d": _hold_n((c > hi20.shift()).astype(float), 10),
            "vol_target": (0.10 / vol20).clip(upper=1.5).fillna(0.0),
            "reversion_3dn": ((ret < 0) & (ret.shift() < 0) & (ret.shift(2) < 0)).astype(float),
        },
        index=bars.index,
    )


def ensemble_exposure(bars: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """Equal-weight (or custom-weight) blend of the four family positions."""
    fams = family_positions(bars)
    if weights is None:
        return fams.mean(axis=1)
    w = pd.Series(weights).reindex(fams.columns).fillna(0.0)
    if w.sum() <= 0:
        raise ValueError("weights must sum to a positive value")
    return (fams * w).sum(axis=1) / w.sum()


def seasonal_multiplier(index: pd.DatetimeIndex) -> pd.Series:
    """Calendar overlay factors: Sep x0.5, Oct/Nov x1.5, Nov-td7 x0.

    WARNING: the ensemble x seasonal combination FAILED the 2024-26 holdout
    Sharpe test (see module docstring). Apply only with that caveat.
    """
    ym = index.to_period("M")
    counter = pd.DataFrame(index=index).groupby(ym).cumcount()
    tdom = pd.Series(counter + 1, index=index)
    month = pd.Series(index.month, index=index)
    mult = pd.Series(1.0, index=index)
    mult[month == 9] = 0.5
    mult[(month == 10) | (month == 11)] = 1.5
    mult[(month == 11) & (tdom == 7)] = 0.0
    return mult


def backtest_exposure(
    position: pd.Series, returns: pd.Series, cost: float = TRADE_COST
) -> pd.Series:
    """Net daily strategy returns: position decided at close t-1 earns t."""
    held = position.shift(1).fillna(0.0)
    return held * returns - cost * held.diff().abs().fillna(0.0)
