"""Edge health assessment and lifecycle transitions.

Rules (all thresholds from config):

    VALIDATED -> ACTIVE      recent posterior healthy (monitoring confirms)
    VALIDATED/ACTIVE -> DEGRADED    posterior P(mean>0) < degrade threshold
    DEGRADED -> SUSPENDED    posterior < suspend threshold
    DEGRADED -> ACTIVE       posterior recovered above degrade threshold
    SUSPENDED -> ACTIVE      posterior recovered (evidence-based reactivation)
    SUSPENDED -> RETIRED     suspended longer than retire_after_suspended_sessions

Transitions append to ``edge_events``; retired edges keep their full history.
With fewer than MIN_RECENT recent signals no transition fires — silence is
not evidence of decay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog
from edgestack.discovery.candidate_generation import CONTINUOUS_RULE_FEATURES
from edgestack.discovery.conditions import evaluate_condition
from edgestack.discovery.edge_store import current_statuses, load_edges, record_edge_event
from edgestack.execution.costs import CostModel
from edgestack.features.binning import QuantileBinner
from edgestack.monitoring.drift import PSI_MATERIAL, population_stability_index
from edgestack.types import Edge, EdgeStatus, Side
from edgestack.validation.bayes import prob_mean_positive

MIN_RECENT = 5
MONITORABLE = (EdgeStatus.VALIDATED, EdgeStatus.ACTIVE, EdgeStatus.DEGRADED, EdgeStatus.SUSPENDED)


@dataclass(frozen=True)
class EdgeHealth:
    edge_id: str
    n_recent: int
    recent_net_mean: float
    recent_hit_rate: float
    recent_posterior: float  # P(true mean > 0 | recent trades), shrunk
    max_feature_psi: float
    sessions_since_status: int


def assess_edge(
    edge: Edge,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cfg: EdgeStackConfig,
    *,
    sessions_since_status: int,
    window: int | None = None,
) -> EdgeHealth:
    """Health of one edge from its most recent out-of-window signals."""
    window = window or cfg.monitoring.rolling_window_signals
    horizon = edge.identity.holding_horizon
    merged = (
        features.merge(
            labels.loc[labels["horizon"] == horizon, ["symbol", "date", "label_end", "gross_ret"]],
            on=["symbol", "date"],
            how="inner",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Reference = first 70% of history (fits thresholds and PSI baselines);
    # recent behavior is judged on the remaining 30%.
    split = int(len(merged) * 0.7)
    reference, live = merged.iloc[:split], merged.iloc[split:]

    binnable = tuple(
        c
        for c in (*CONTINUOUS_RULE_FEATURES, "bench_trend_200", "bench_vol_20")
        if c in merged.columns
    )
    binner = QuantileBinner(quantiles=cfg.discovery.quantile_bins).fit(reference, binnable)
    mask = evaluate_condition(edge.identity.condition, live, binner)
    hits = live.loc[mask].tail(window)

    cost = CostModel.from_config(cfg).roundtrip_cost(edge.identity.direction, horizon)
    gross = hits["gross_ret"].to_numpy()
    directional = gross if edge.identity.direction is Side.LONG else -gross
    net = directional - cost

    psi_values = []
    for feature_name in edge.identity.feature_dependencies:
        if feature_name in merged.columns:
            try:
                psi_values.append(
                    population_stability_index(
                        reference[feature_name].to_numpy(), live[feature_name].to_numpy()
                    )
                )
            except Exception:
                continue
    max_psi = max(psi_values) if psi_values else 0.0

    if len(net) >= 2:
        posterior = prob_mean_positive(net, prior_pseudo_n=cfg.scoring.shrinkage_min_sample)
        hit_rate = float((net > 0).mean())
        mean = float(net.mean())
    else:
        posterior, hit_rate, mean = 0.5, 0.0, 0.0

    return EdgeHealth(
        edge_id=edge.identity.edge_id,
        n_recent=len(net),
        recent_net_mean=mean,
        recent_hit_rate=hit_rate,
        recent_posterior=posterior,
        max_feature_psi=max_psi,
        sessions_since_status=sessions_since_status,
    )


def decide_transition(
    status: EdgeStatus, health: EdgeHealth, cfg: EdgeStackConfig
) -> tuple[EdgeStatus, str] | None:
    m = cfg.monitoring
    if status is EdgeStatus.SUSPENDED and (
        health.sessions_since_status > m.retire_after_suspended_sessions
    ):
        return EdgeStatus.RETIRED, (
            f"suspended for {health.sessions_since_status} sessions "
            f"(> {m.retire_after_suspended_sessions})"
        )
    if health.n_recent < MIN_RECENT:
        return None

    posterior = health.recent_posterior
    drifted = health.max_feature_psi > PSI_MATERIAL

    if status in (EdgeStatus.VALIDATED, EdgeStatus.ACTIVE):
        if posterior < m.suspend_posterior_threshold:
            return EdgeStatus.SUSPENDED, f"posterior {posterior:.2f} below suspend threshold"
        if posterior < m.degrade_posterior_threshold or drifted:
            reason = (
                f"posterior {posterior:.2f} below degrade threshold"
                if posterior < m.degrade_posterior_threshold
                else f"feature drift PSI {health.max_feature_psi:.2f}"
            )
            return EdgeStatus.DEGRADED, reason
        if status is EdgeStatus.VALIDATED:
            return EdgeStatus.ACTIVE, f"monitoring confirms health (posterior {posterior:.2f})"
        return None

    if status is EdgeStatus.DEGRADED:
        if posterior < m.suspend_posterior_threshold:
            return EdgeStatus.SUSPENDED, f"posterior {posterior:.2f} below suspend threshold"
        if posterior >= m.degrade_posterior_threshold and not drifted:
            return EdgeStatus.ACTIVE, f"recovered (posterior {posterior:.2f})"
        return None

    if status is EdgeStatus.SUSPENDED:
        if posterior >= m.degrade_posterior_threshold and not drifted:
            return EdgeStatus.ACTIVE, f"reactivated on evidence (posterior {posterior:.2f})"
        return None
    return None


def run_monitoring(
    catalog: DataCatalog,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cfg: EdgeStackConfig,
) -> list[tuple[str, EdgeStatus, EdgeStatus, str]]:
    """Assess all monitorable edges, apply transitions, return them."""
    edges = load_edges(catalog, statuses=MONITORABLE)
    statuses = current_statuses(catalog).set_index("edge_id")["status"]

    with catalog.connect() as con:
        last_events = (
            con.execute("SELECT edge_id, max(ts) AS ts FROM edge_events GROUP BY edge_id")
            .df()
            .set_index("edge_id")["ts"]
        )

    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(features["date"]).unique()))
    latest = sessions[-1]

    transitions = []
    for edge in edges:
        edge_id = edge.identity.edge_id
        status = EdgeStatus(statuses.get(edge_id, EdgeStatus.VALIDATED.value))
        last_ts = last_events.get(edge_id)
        since = (
            int(
                np.searchsorted(sessions.values, np.datetime64(latest))
                - np.searchsorted(sessions.values, np.datetime64(pd.Timestamp(last_ts)))
            )
            if last_ts is not None
            else 0
        )
        health = assess_edge(edge, features, labels, cfg, sessions_since_status=max(0, since))
        decision = decide_transition(status, health, cfg)
        if decision is None:
            continue
        new_status, rule = decision
        record_edge_event(
            catalog,
            edge_id,
            status,
            new_status,
            rule,
            metrics_json=pd.Series(health.__dict__).to_json(),
        )
        transitions.append((edge_id, status, new_status, rule))
    return transitions
