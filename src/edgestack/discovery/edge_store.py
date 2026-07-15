"""Persistence for candidates, edges and lifecycle events (DuckDB)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pandas as pd

from edgestack.data.catalog import DataCatalog
from edgestack.discovery.candidate_generation import DiscoveryBatch
from edgestack.exceptions import ValidationError
from edgestack.types import Edge, EdgeStatus


def save_batch(catalog: DataCatalog, batch: DiscoveryBatch) -> None:
    """Persist EVERY evaluated candidate — the honest trial count."""
    now = datetime.now(UTC)
    with catalog.connect() as con:
        con.executemany(
            "INSERT OR REPLACE INTO candidates VALUES (?, ?, ?, ?, ?)",
            [
                [c.candidate_id, batch.batch_id, batch.experiment_id, now,
                 c.model_dump_json()]
                for c in batch.candidates
            ],
        )
    catalog.update_trial_count(batch.experiment_id, batch.trial_count)


def load_batch(catalog: DataCatalog, batch_id: str | None = None) -> DiscoveryBatch:
    """Load a candidate batch (latest by default)."""
    from edgestack.types import CandidateEdge

    with catalog.connect() as con:
        if batch_id is None:
            row = con.execute(
                "SELECT batch_id FROM candidates ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise ValidationError("no discovery batches found; run `edges discover` first")
            batch_id = row[0]
        rows = con.execute(
            "SELECT candidate_id, experiment_id, payload FROM candidates WHERE batch_id = ?",
            [batch_id],
        ).fetchall()
    if not rows:
        raise ValidationError(f"unknown discovery batch: {batch_id}")
    candidates = tuple(CandidateEdge.model_validate_json(payload) for _, _, payload in rows)
    return DiscoveryBatch(
        batch_id=batch_id, experiment_id=rows[0][1], candidates=candidates
    )


def save_edges(catalog: DataCatalog, edges: list[Edge]) -> None:
    now = datetime.now(UTC)
    with catalog.connect() as con:
        for edge in edges:
            con.execute(
                "INSERT OR REPLACE INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    edge.identity.edge_id,
                    edge.identity.edge_version,
                    now,
                    edge.lifecycle.experiment_id,
                    edge.lifecycle.discovery_batch_id,
                    edge.identity.name,
                    edge.identity.family.value,
                    edge.identity.direction.value,
                    edge.identity.holding_horizon,
                    edge.stats.net_mean_return,
                    edge.stats.q_value,
                    edge.stats.deflated_sharpe_ratio,
                    edge.model_dump_json(),
                ],
            )
            con.execute(
                "INSERT INTO edge_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    uuid.uuid4().hex,
                    edge.identity.edge_id,
                    now,
                    EdgeStatus.CANDIDATE.value,
                    edge.lifecycle.status.value,
                    "walk_forward_validation",
                    edge.stats.model_dump_json(),
                ],
            )


def record_edge_event(
    catalog: DataCatalog, edge_id: str, from_status: EdgeStatus | None,
    to_status: EdgeStatus, rule_fired: str, metrics_json: str = "{}",
) -> None:
    with catalog.connect() as con:
        con.execute(
            "INSERT INTO edge_events VALUES (?, ?, ?, ?, ?, ?, ?)",
            [uuid.uuid4().hex, edge_id, datetime.now(UTC),
             from_status.value if from_status else None, to_status.value,
             rule_fired, metrics_json],
        )


def current_statuses(catalog: DataCatalog) -> pd.DataFrame:
    """Latest lifecycle status per edge (event-sourced, never updated in place)."""
    with catalog.connect() as con:
        return con.execute(
            """
            SELECT edge_id, to_status AS status
            FROM (
                SELECT edge_id, to_status,
                       ROW_NUMBER() OVER (PARTITION BY edge_id ORDER BY ts DESC) AS rn
                FROM edge_events
            )
            WHERE rn = 1
            """
        ).df()


def load_edges(
    catalog: DataCatalog, statuses: tuple[EdgeStatus, ...] = (EdgeStatus.VALIDATED,
                                                              EdgeStatus.ACTIVE)
) -> list[Edge]:
    """Load full Edge payloads whose CURRENT status is one of ``statuses``."""
    wanted = {s.value for s in statuses}
    status_df = current_statuses(catalog)
    keep = set(status_df.loc[status_df["status"].isin(wanted), "edge_id"])
    if not keep:
        return []
    with catalog.connect() as con:
        rows = con.execute(
            """
            SELECT payload FROM (
                SELECT payload, edge_id,
                       ROW_NUMBER() OVER (PARTITION BY edge_id ORDER BY edge_version DESC) AS rn
                FROM edges
            ) WHERE rn = 1
            """
        ).fetchall()
    edges = [Edge.model_validate_json(payload) for (payload,) in rows]
    return [e for e in edges if e.identity.edge_id in keep]
