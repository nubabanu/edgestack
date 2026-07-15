"""Maximum adverse / favorable excursion labels.

Secondary risk target for rule discovery: how far a long entry moved AGAINST
you (MAE, <= 0) and FOR you (MFE, >= 0) inside the holding window. Timing
matches forward_returns: signal at close of T, entry at open of T+delay,
window covers the ``horizon`` sessions starting at entry; ``label_end`` is
the same exit session forward_returns uses, so interval purging stays exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from edgestack.exceptions import DataError


def adverse_excursion_labels(
    panel: pd.DataFrame,
    horizon: int,
    *,
    execution_delay: int = 1,
) -> pd.DataFrame:
    """MAE/MFE per (symbol, signal date) for a long held ``horizon`` sessions."""
    if execution_delay < 1:
        raise DataError("execution_delay must be >= 1 for close-observed signals")
    frames = []
    for symbol, group in panel.groupby("symbol", sort=True):
        df = group.set_index("date").sort_index()
        n = len(df)
        window = horizon
        need = execution_delay + horizon  # exit session offset
        if n <= need:
            continue
        opens = df["open"].to_numpy()
        lows = df["low"].to_numpy()
        highs = df["high"].to_numpy()
        dates = df.index.to_numpy()

        # window of sessions [t+delay, t+delay+horizon-1] for each signal t
        low_windows = sliding_window_view(lows, window)    # starts at index i
        high_windows = sliding_window_view(highs, window)
        # valid signal indices: t such that t+delay+horizon <= n-1 (exit exists)
        t_max = n - need
        t_idx = np.arange(t_max)
        start = t_idx + execution_delay
        entry = opens[start]
        mae = low_windows[start].min(axis=1) / entry - 1.0
        mfe = high_windows[start].max(axis=1) / entry - 1.0

        frames.append(pd.DataFrame({
            "symbol": symbol,
            "date": dates[t_idx],
            "horizon": horizon,
            "entry_date": dates[start],
            "label_end": dates[t_idx + need],
            "mae": mae,
            "mfe": mfe,
        }))
    if not frames:
        raise DataError("panel too short for adverse-excursion labels")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["symbol", "date"]).reset_index(drop=True)
