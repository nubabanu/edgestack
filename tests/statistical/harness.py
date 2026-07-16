"""Shared harness for statistical acceptance tests.

Small three-symbol synthetic markets, a restricted feature set for speed, and
a fixed weekday/calendar condition menu. The Friday drift injected by
``CalendarEffect(is_friday)`` lands in the close of Friday sessions; a signal
observed at THURSDAY's close (weekday == 3) with a 1-session horizon enters at
Friday's open and exits at Monday's open, capturing that drift out of sample.
"""

from __future__ import annotations

from datetime import date

from edgestack.config import EdgeStackConfig
from edgestack.data.providers.synthetic import GBM, Effect, SyntheticMarket
from edgestack.discovery.candidate_generation import DiscoveryBatch, generate_candidates
from edgestack.features.registry import build_features, get_spec
from edgestack.labels.forward_returns import forward_return_labels
from edgestack.types import Condition, Edge, Predicate
from edgestack.validation.walk_forward import validate_batch

SYMBOLS = ("AAA", "BBB", "CCC")
START, END = date(2010, 1, 1), date(2020, 12, 31)

FEATURES = (
    "cal_weekday",
    "cal_is_monday",
    "cal_is_friday",
    "cal_turn_of_month",
    "cal_pre_holiday",
    "bench_trend_200",
    "bench_vol_20",
    "above_sma200",
    "rel_volume_20",
)

# The fixed research menu: five weekday rules + two calendar-window rules.
CONDITIONS: list[Condition] = [
    *[Predicate(feature="cal_weekday", op="==", value=float(wd)) for wd in range(5)],
    Predicate(feature="cal_turn_of_month", op="==", value=1.0),
    Predicate(feature="cal_pre_holiday", op="==", value=1.0),
]

THURSDAY_LONG_PREFIX = "long_h1: cal_weekday == 3.0"


def stat_cfg() -> EdgeStackConfig:
    return EdgeStackConfig.model_validate(
        {
            "universe": {"symbols": list(SYMBOLS), "benchmark_symbol": "AAA"},
            "signals": {"horizons": [1, 5], "min_effective_sample_size": 100},
            "validation": {
                "n_folds": 3,
                "test_sessions": 250,
                "train_min_sessions": 750,
                "embargo_sessions": 5,
                "final_test_start": "2022-01-01",
                "bootstrap_samples": 500,
            },
            "discovery": {"horizons": [1], "min_support": 50},
        }
    )


def build_frames(
    effects: tuple[Effect, ...],
    *,
    seed: int = 42,
    sigma: float = 0.05,
    cfg: EdgeStackConfig | None = None,
):
    """Synthetic market -> (cfg, features, labels) with the restricted feature set."""
    cfg = cfg or stat_cfg()
    market = SyntheticMarket(seed=seed, base=GBM(mu=0.0, sigma=sigma), effects=effects)
    panel = market.generate(SYMBOLS, START, END)
    specs = tuple(get_spec(name) for name in FEATURES)
    features = build_features(panel, cfg, specs)
    labels = forward_return_labels(
        panel,
        (1, 5),
        benchmark_symbol="AAA",
        execution_delay=cfg.signals.execution_delay_sessions,
    )
    return cfg, features, labels


def run_research(
    effects: tuple[Effect, ...],
    *,
    seed: int = 42,
    sigma: float = 0.05,
) -> tuple[EdgeStackConfig, DiscoveryBatch, list[Edge]]:
    """Generate market -> features -> labels -> discovery -> validation."""
    cfg, features, labels = build_frames(effects, seed=seed, sigma=sigma)
    batch = generate_candidates(
        features, labels, cfg, experiment_id="stat-test", conditions=CONDITIONS
    )
    edges = validate_batch(batch, features, labels, cfg)
    return cfg, batch, edges


def validated_names(edges: list[Edge]) -> list[str]:
    from edgestack.types import EdgeStatus

    return sorted(e.identity.name for e in edges if e.lifecycle.status is EdgeStatus.VALIDATED)


def edge_by_name_prefix(edges: list[Edge], prefix: str) -> Edge:
    matches = [e for e in edges if e.identity.name.startswith(prefix)]
    assert matches, f"no edge named like {prefix!r}; have {[e.identity.name for e in edges]}"
    return matches[0]


def friday_effect(drift_bps: float, *, start: date | None = None, end: date | None = None):
    from edgestack.data.providers.synthetic import CalendarEffect

    return CalendarEffect(
        facts_column="is_friday",
        drift_bps=drift_bps,
        active_start=start,
        active_end=end,
    )
