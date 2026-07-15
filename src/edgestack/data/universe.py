"""Universe construction with honest survivorship labeling."""

from __future__ import annotations

from datetime import date

from edgestack.config import EdgeStackConfig
from edgestack.types import UniverseSnapshot

SURVIVORSHIP_WARNING = (
    "Universe is a static symbol list, not point-in-time membership: results "
    "carry survivorship bias (delisted losers are absent). Treat backtests as "
    "optimistic."
)


def static_universe(cfg: EdgeStackConfig, as_of: date) -> UniverseSnapshot:
    """Universe from the configured symbol list.

    The bundled free data sources cannot provide point-in-time index
    membership, so this is honest about being survivor-biased.
    """
    limitations: tuple[str, ...] = ()
    if not cfg.universe.point_in_time:
        limitations = (SURVIVORSHIP_WARNING,)
    return UniverseSnapshot(
        as_of_date=as_of,
        symbols=cfg.universe.symbols,
        source=cfg.universe.source,
        methodology="static configured watchlist",
        is_point_in_time=cfg.universe.point_in_time,
        limitations=limitations,
    )
