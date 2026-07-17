package com.edgestack.app.domain

import com.edgestack.app.domain.model.TrackedPosition

/** Unrealized P&L for one tracked position at [lastPrice]; null when no quote. */
data class PositionPnl(
    val marketValue: Double,
    val profit: Double,
    val profitPercent: Double,
)

fun TrackedPosition.pnlAt(lastPrice: Double?): PositionPnl? {
    if (lastPrice == null || lastPrice <= 0.0 || entryPrice <= 0.0) return null
    val value = lastPrice * quantity
    val cost = entryPrice * quantity
    return PositionPnl(
        marketValue = value,
        profit = value - cost,
        profitPercent = (lastPrice / entryPrice - 1.0) * 100.0,
    )
}
