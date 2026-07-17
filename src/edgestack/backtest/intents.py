"""Build trade intents by replaying validated edges OUT OF SAMPLE.

Intents are generated only inside walk-forward test windows, with quantile
thresholds fitted on each fold's training window — the backtest never trades
a rule on data that shaped it.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from edgestack.backtest.engine import TradeIntent
from edgestack.config import EdgeStackConfig
from edgestack.discovery.candidate_generation import CONTINUOUS_RULE_FEATURES
from edgestack.discovery.conditions import evaluate_condition
from edgestack.features.binning import QuantileBinner
from edgestack.types import Edge, Side
from edgestack.validation.splits import PurgedWalkForwardSplitter


def build_trade_intents(
    features: pd.DataFrame, panel: pd.DataFrame, edges: list[Edge], cfg: EdgeStackConfig
) -> list[TradeIntent]:
    if not edges:
        return []
    merged = features.merge(
        panel[["symbol", "date", "close"]], on=["symbol", "date"], how="inner"
    ).reset_index(drop=True)
    splitter = PurgedWalkForwardSplitter.from_config(cfg)
    # No labels are trained here; folds only carve OOS windows, so the outcome
    # interval is the signal date itself.
    folds = splitter.split_frame(merged["date"], merged["date"])
    binnable = tuple(
        c
        for c in (*CONTINUOUS_RULE_FEATURES, "bench_trend_200", "bench_vol_20")
        if c in merged.columns
    )

    # Strongest edge first so the (symbol, session) dedupe keeps the best one.
    ordered = sorted(edges, key=lambda e: -e.stats.net_mean_return)
    intents: dict[tuple[str, pd.Timestamp], TradeIntent] = {}
    for fold in folds:
        train = merged.iloc[fold.train_idx]
        test = merged.iloc[fold.test_idx]
        binner = QuantileBinner(quantiles=cfg.discovery.quantile_bins).fit(train, binnable)
        for edge in ordered:
            mask = evaluate_condition(edge.identity.condition, test, binner)
            for row in test.loc[mask].itertuples(index=False):
                record = cast(Any, row)
                symbol = str(record.symbol)
                session = pd.Timestamp(record.date)
                key = (symbol, session)
                if key in intents:
                    continue
                close = float(record.close)
                natr = getattr(record, "natr_14", np.nan)
                atr = float(natr) * close if np.isfinite(natr) and natr > 0 else 0.02 * close
                side = edge.identity.direction
                stop_mult = cfg.risk.atr_stop_multiple
                target_mult = cfg.risk.atr_target_multiples[0]
                if side is Side.LONG:
                    stop = close - stop_mult * atr
                    target = close + target_mult * atr
                else:
                    stop = close + stop_mult * atr
                    target = close - target_mult * atr
                intents[key] = TradeIntent(
                    symbol=symbol,
                    signal_session=session,
                    side=side,
                    horizon=edge.identity.holding_horizon,
                    stop_price=max(stop, 0.01),
                    target_price=max(target, 0.01),
                )
    return sorted(intents.values(), key=lambda i: (i.signal_session, i.symbol))
