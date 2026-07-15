"""Forward-return labels with explicit execution timing.

Timing contract (the safe daily default from the spec):

- the signal is observed at the close of session ``T``;
- entry happens at the open of session ``T + execution_delay`` (default 1);
- an ``h``-session holding exits at the open of session ``T + delay + h``.

Every label row carries ``entry_date`` and ``label_end`` so purged validation
can remove any training row whose outcome interval overlaps a test window.
Returns here are GROSS; costs are applied by the cost model at evaluation
time so one label set serves all cost scenarios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgestack.exceptions import DataError

LABEL_COLUMNS = (
    "symbol", "date", "horizon", "entry_date", "label_end",
    "gross_ret", "bench_ret", "excess_ret",
)


def session_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-session return decomposition for one symbol (sorted by date).

    close_to_close, close_to_open (overnight), open_to_close (intraday),
    open_to_open. "Performance while the market is closed" is close_to_open.
    """
    out = pd.DataFrame(index=bars.index)
    close, open_ = bars["close"], bars["open"]
    out["close_to_close"] = close.pct_change()
    out["close_to_open"] = open_ / close.shift() - 1.0
    out["open_to_close"] = close / open_ - 1.0
    out["open_to_open"] = open_.pct_change()
    return out


def forward_return_labels(
    panel: pd.DataFrame,
    horizons: tuple[int, ...],
    *,
    benchmark_symbol: str,
    execution_delay: int = 1,
    roundtrip_cost: float | None = None,
) -> pd.DataFrame:
    """Build open-to-open forward-return labels for every (symbol, date, horizon).

    Rows whose exit session does not exist yet (end of data) are dropped —
    an unknown outcome is not a label.
    """
    if execution_delay < 1:
        raise DataError(
            "execution_delay must be >= 1: a close-observed signal cannot be "
            "executed at that same close"
        )
    symbols = panel["symbol"].unique()
    bench_sym = benchmark_symbol if benchmark_symbol in symbols else str(sorted(symbols)[0])
    bench = panel.loc[panel["symbol"] == bench_sym].set_index("date").sort_index()

    frames = []
    for symbol, group in panel.groupby("symbol", sort=True):
        df = group.set_index("date").sort_index()
        for horizon in horizons:
            shift_in = -execution_delay
            shift_out = -(execution_delay + horizon)
            entry_open = df["open"].shift(shift_in)
            exit_open = df["open"].shift(shift_out)
            entry_date = df.index.to_series().shift(shift_in)
            label_end = df.index.to_series().shift(shift_out)
            gross = exit_open / entry_open - 1.0

            bench_entry = bench["open"].reindex(df.index).shift(shift_in)
            bench_exit = bench["open"].reindex(df.index).shift(shift_out)
            bench_ret = bench_exit / bench_entry - 1.0

            frame = pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": df.index,
                    "horizon": horizon,
                    "entry_date": entry_date.to_numpy(),
                    "label_end": label_end.to_numpy(),
                    "gross_ret": gross.to_numpy(),
                    "bench_ret": bench_ret.to_numpy(),
                }
            )
            frames.append(frame.dropna(subset=["gross_ret", "label_end"]))

    labels = pd.concat(frames, ignore_index=True)
    labels["excess_ret"] = labels["gross_ret"] - labels["bench_ret"]
    if roundtrip_cost is not None:
        # Primary learning target per the rules campaign: benchmark-adjusted
        # return net of estimated round-trip costs (never a raw win/lose flag).
        labels["excess_net_ret"] = labels["excess_ret"] - roundtrip_cost
    labels["entry_date"] = pd.to_datetime(labels["entry_date"])
    labels["label_end"] = pd.to_datetime(labels["label_end"])
    labels["vol_scale"] = np.nan  # filled by callers that need vol-adjusted targets
    return labels.sort_values(["symbol", "horizon", "date"]).reset_index(drop=True)
