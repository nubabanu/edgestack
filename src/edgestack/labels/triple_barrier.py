"""Triple-barrier labels: volatility-scaled profit/stop barriers + time barrier.

Timing matches forward_returns: signal at close of T, entry at the open of
T+delay. Barriers are set from ATR measured at signal time (causal). Bar-level
resolution is conservative for longs: when a bar touches both barriers, the
stop is assumed to hit first; when the open gaps beyond a barrier, the fill is
at the open, not the barrier price.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from edgestack.exceptions import DataError
from edgestack.features.volatility import atr as atr_fn


@dataclass(frozen=True)
class TripleBarrierConfig:
    horizon: int = 10
    up_atr_multiple: float = 2.5
    down_atr_multiple: float = 2.0
    atr_window: int = 14
    execution_delay: int = 1


def triple_barrier_labels(panel: pd.DataFrame, cfg: TripleBarrierConfig) -> pd.DataFrame:
    """Long-side triple-barrier outcomes for every (symbol, signal date).

    Columns: symbol, date, entry_date, label_end, outcome (+1 profit barrier,
    -1 stop barrier, 0 time barrier), gross_ret, holding_sessions.
    """
    if cfg.execution_delay < 1:
        raise DataError("execution_delay must be >= 1 for close-observed signals")

    frames = []
    for symbol, group in panel.groupby("symbol", sort=True):
        df = group.set_index("date").sort_index()
        frames.append(_one_symbol(str(symbol), df, cfg))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def _one_symbol(symbol: str, df: pd.DataFrame, cfg: TripleBarrierConfig) -> pd.DataFrame:
    n = len(df)
    dates = df.index.to_numpy()
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    atr = atr_fn(df, cfg.atr_window).to_numpy()

    rows = []
    d = cfg.execution_delay
    for t in range(n):
        entry_idx = t + d
        last_idx = t + d + cfg.horizon - 1  # last session inside the window
        if last_idx >= n or np.isnan(atr[t]):
            continue
        entry = opens[entry_idx]
        up = entry + cfg.up_atr_multiple * atr[t]
        down = entry - cfg.down_atr_multiple * atr[t]

        outcome = 0
        exit_price = np.nan
        exit_idx = last_idx
        for k in range(entry_idx, last_idx + 1):
            open_k = opens[k]
            # Gap through a barrier at the open fills at the open.
            if open_k <= down:
                outcome, exit_price, exit_idx = -1, open_k, k
                break
            if open_k >= up:
                outcome, exit_price, exit_idx = 1, open_k, k
                break
            hit_down = lows[k] <= down
            hit_up = highs[k] >= up
            if hit_down:  # conservative: stop first when both touch
                outcome, exit_price, exit_idx = -1, down, k
                break
            if hit_up:
                outcome, exit_price, exit_idx = 1, up, k
                break
        if outcome == 0:
            # Time barrier: exit at the open after the window, if it exists.
            time_exit = last_idx + 1
            if time_exit >= n:
                continue
            exit_price, exit_idx = opens[time_exit], time_exit

        rows.append(
            (
                symbol,
                dates[t],
                dates[entry_idx],
                dates[exit_idx],
                outcome,
                exit_price / entry - 1.0,
                exit_idx - entry_idx + 1,
            )
        )

    return pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "date",
            "entry_date",
            "label_end",
            "outcome",
            "gross_ret",
            "holding_sessions",
        ],
    )
