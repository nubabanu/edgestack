"""Daily-reset inverse-ETF tracking, financing, and execution accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from edgestack.exceptions import ValidationError


@dataclass(frozen=True)
class InverseEtfAccounting:
    daily_returns: pd.Series
    compounded_return: float
    static_inverse_return: float
    daily_reset_decay: float
    total_drag: float


def account_inverse_etf(
    index_returns: pd.Series,
    *,
    inverse_multiple: float = -1.0,
    annual_tracking_difference: float = 0.005,
    annual_expense_ratio: float = 0.009,
    annual_financing_rate: float = 0.0,
    round_trip_cost_bps: float = 10.0,
) -> InverseEtfAccounting:
    """Model inverse exposure as a daily-reset ETF, never as a direct short."""
    if inverse_multiple >= 0:
        raise ValidationError("inverse_multiple must be negative")
    values = index_returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise ValidationError("inverse ETF accounting requires index returns")
    daily_drag = (annual_tracking_difference + annual_expense_ratio + annual_financing_rate) / 252.0
    daily = inverse_multiple * values - daily_drag
    if bool((daily <= -1.0).any()):
        raise ValidationError("modeled inverse ETF suffered an invalid total-loss day")
    compounded = float(np.prod(1.0 + daily) - 1.0 - round_trip_cost_bps / 1e4)
    static_inverse = float(inverse_multiple * (np.prod(1.0 + values) - 1.0))
    reset_without_drag = float(np.prod(1.0 + inverse_multiple * values) - 1.0)
    reset_decay = reset_without_drag - static_inverse
    return InverseEtfAccounting(
        daily_returns=daily,
        compounded_return=compounded,
        static_inverse_return=static_inverse,
        daily_reset_decay=reset_decay,
        total_drag=compounded - static_inverse,
    )
