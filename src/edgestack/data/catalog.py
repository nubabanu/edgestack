"""Data catalog: Parquet price store, DuckDB metadata, and the TestPeriodGuard.

All modeling-facing data access flows through :meth:`DataCatalog.load_panel`,
which silently truncates everything at the final-test boundary. The only way
to see the configured guarded test period is an explicit, audited unlock:

    with catalog.guard.unlock(reason="final evaluation v1.0") as key:
        panel = catalog.load_panel(..., unlock_key=key)

There is deliberately no configuration flag that disables the guard.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import duckdb
import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.schemas import (
    BAR_COLUMNS,
    CORPORATE_ACTION_COLUMNS,
    validate_bars,
    validate_corporate_actions,
)
from edgestack.exceptions import DataError, TestPeriodLockedError
from edgestack.logging import get_logger, log_event

log = get_logger("catalog")

_DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    kind TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    code_hash TEXT,
    data_hash TEXT,
    seed BIGINT,
    start_date DATE,
    end_date DATE,
    feature_set TEXT,
    cost_scenario TEXT,
    trial_count BIGINT,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    payload JSON NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
    edge_id TEXT NOT NULL,
    edge_version INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL,
    experiment_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    name TEXT NOT NULL,
    family TEXT NOT NULL,
    direction TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    net_mean_return DOUBLE,
    q_value DOUBLE,
    deflated_sharpe DOUBLE,
    payload JSON NOT NULL,
    PRIMARY KEY (edge_id, edge_version)
);
CREATE TABLE IF NOT EXISTS edge_events (
    event_id TEXT PRIMARY KEY,
    edge_id TEXT NOT NULL,
    ts TIMESTAMP NOT NULL,
    from_status TEXT,
    to_status TEXT NOT NULL,
    rule_fired TEXT,
    metrics JSON
);
CREATE TABLE IF NOT EXISTS backtests (
    run_id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    config_hash TEXT NOT NULL,
    cost_scenario TEXT NOT NULL,
    payload JSON NOT NULL
);
CREATE TABLE IF NOT EXISTS signal_reports (
    as_of_date DATE NOT NULL,
    created_at TIMESTAMP NOT NULL,
    config_hash TEXT NOT NULL,
    payload JSON NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id TEXT PRIMARY KEY,
    ts TIMESTAMP NOT NULL,
    event TEXT NOT NULL,
    reason TEXT,
    experiment_id TEXT,
    config_hash TEXT,
    detail JSON
);
CREATE TABLE IF NOT EXISTS experiment_manifests_v2 (
    manifest_hash TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    payload JSON NOT NULL
);
CREATE TABLE IF NOT EXISTS trial_ledger_v2 (
    trial_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    status TEXT NOT NULL,
    payload JSON NOT NULL
);
CREATE TABLE IF NOT EXISTS frozen_artifacts_v2 (
    content_hash TEXT PRIMARY KEY,
    manifest_hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    payload JSON NOT NULL
);
CREATE TABLE IF NOT EXISTS promotion_decisions_v2 (
    sleeve_id TEXT NOT NULL,
    artifact_hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    promoted BOOLEAN NOT NULL,
    payload JSON NOT NULL,
    PRIMARY KEY (sleeve_id, artifact_hash)
);
CREATE TABLE IF NOT EXISTS prospective_evidence_v2 (
    sleeve_id TEXT NOT NULL,
    frozen_artifact_hash TEXT NOT NULL,
    recorded_at TIMESTAMP NOT NULL,
    payload JSON NOT NULL,
    PRIMARY KEY (sleeve_id, frozen_artifact_hash, recorded_at)
);
CREATE TABLE IF NOT EXISTS publications_v2 (
    run_id TEXT PRIMARY KEY,
    published_at TIMESTAMP NOT NULL,
    bundle_hash TEXT NOT NULL,
    payload JSON NOT NULL
);
"""


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes atomically: temp file in the same directory, then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def safe_child_path(root: Path, name: str) -> Path:
    """Join ``name`` under ``root`` rejecting traversal outside the root."""
    posix_name = name.replace("\\", "/")
    posix_path = PurePosixPath(posix_name)
    windows_path = PureWindowsPath(name)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
    ):
        raise DataError(f"unsafe path outside {root}: {name!r}")
    candidate = root.joinpath(*posix_path.parts).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise DataError(f"unsafe path outside {root}: {name!r}")
    return candidate


class TestPeriodGuard:
    """Enforces the configured guarded test period.

    Frames served through the guard are truncated to dates strictly before
    ``test_start`` unless a live single-use unlock key is presented; unlocking
    writes a ``test_set_accessed`` audit record *before* any data is served.
    """

    def __init__(self, test_start: date, audit_fn: Any) -> None:
        self.test_start = pd.Timestamp(test_start)
        self._audit = audit_fn
        self._live_keys: set[str] = set()

    @contextmanager
    def unlock(self, reason: str, experiment_id: str | None = None) -> Iterator[str]:
        if not reason or not reason.strip():
            raise TestPeriodLockedError("an explicit reason is required to unlock the test period")
        key = uuid.uuid4().hex
        # Audit BEFORE yielding: access is recorded even if the caller crashes.
        self._audit("test_set_accessed", reason=reason, experiment_id=experiment_id)
        self._live_keys.add(key)
        try:
            yield key
        finally:
            self._live_keys.discard(key)

    def apply(self, df: pd.DataFrame, unlock_key: str | None = None) -> pd.DataFrame:
        if unlock_key is not None:
            if unlock_key not in self._live_keys:
                raise TestPeriodLockedError("invalid or expired test-period unlock key")
            return df
        return df.loc[pd.to_datetime(df["date"]) < self.test_start].reset_index(drop=True)


class DataCatalog:
    """Filesystem + DuckDB storage for prices, metadata and audit records."""

    def __init__(self, cfg: EdgeStackConfig) -> None:
        self.cfg = cfg
        self.data_dir = Path(cfg.paths.data_dir)
        self.artifacts_dir = Path(cfg.paths.artifacts_dir)
        self.prices_dir = self.data_dir / "curated" / "prices"
        self.corporate_actions_dir = self.data_dir / "curated" / "corporate_actions"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.artifacts_dir / "edgestack.duckdb"
        self._ensure_schema()
        self.guard = TestPeriodGuard(cfg.validation.final_test_start, self.audit)

    # -- connections ---------------------------------------------------------

    def _ensure_schema(self) -> None:
        with duckdb.connect(str(self.db_path)) as con:
            con.execute(_DDL)

    @contextmanager
    def connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        con = duckdb.connect(str(self.db_path))
        try:
            yield con
        finally:
            con.close()

    # -- price storage ---------------------------------------------------------

    def write_bars(self, df: pd.DataFrame, *, provider: str) -> dict[str, int]:
        """Merge validated bars into per-symbol Parquet files (atomic writes).

        On overlapping (symbol, date) rows the incoming data wins — providers
        occasionally restate history and restatements are assumed corrections.
        Returns rows written per symbol.
        """
        bars = validate_bars(df, context=f"write_bars[{provider}]")
        written: dict[str, int] = {}
        for symbol, group in bars.groupby("symbol", sort=True):
            path = safe_child_path(self.prices_dir, f"{symbol}.parquet")
            merged = group
            if path.exists():
                existing = pd.read_parquet(path)
                merged = (
                    pd.concat([existing, group], ignore_index=True)
                    .drop_duplicates(subset=["symbol", "date"], keep="last")
                    .sort_values("date")
                    .reset_index(drop=True)
                )
            import io

            buf = io.BytesIO()
            merged[list(BAR_COLUMNS)].to_parquet(buf, index=False)
            atomic_write_bytes(path, buf.getvalue())
            written[str(symbol)] = len(group)
        log_event(log, 20, "bars written", provider=provider, symbols=len(written))
        return written

    def list_symbols(self) -> tuple[str, ...]:
        if not self.prices_dir.exists():
            return ()
        return tuple(sorted(p.stem for p in self.prices_dir.glob("*.parquet")))

    def write_corporate_actions(self, df: pd.DataFrame, *, provider: str) -> dict[str, int]:
        if df.empty:
            return {}
        actions = validate_corporate_actions(df, context=f"corporate_actions[{provider}]")
        written: dict[str, int] = {}
        for symbol, group in actions.groupby("symbol", sort=True):
            path = safe_child_path(self.corporate_actions_dir, f"{symbol}.parquet")
            merged = group
            if path.exists():
                merged = (
                    pd.concat([pd.read_parquet(path), group], ignore_index=True)
                    .drop_duplicates(["symbol", "date", "action_type"], keep="last")
                    .sort_values(["date", "action_type"])
                    .reset_index(drop=True)
                )
            import io

            buffer = io.BytesIO()
            merged.loc[:, list(CORPORATE_ACTION_COLUMNS)].to_parquet(buffer, index=False)
            atomic_write_bytes(path, buffer.getvalue())
            written[str(symbol)] = len(group)
        return written

    def load_corporate_actions(
        self,
        symbols: tuple[str, ...] | None = None,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        available = tuple(
            sorted(path.stem for path in self.corporate_actions_dir.glob("*.parquet"))
        )
        wanted = symbols or available
        frames = [
            pd.read_parquet(safe_child_path(self.corporate_actions_dir, f"{symbol}.parquet"))
            for symbol in wanted
            if safe_child_path(self.corporate_actions_dir, f"{symbol}.parquet").exists()
        ]
        if not frames:
            return pd.DataFrame(columns=CORPORATE_ACTION_COLUMNS)
        output = pd.concat(frames, ignore_index=True)
        if start is not None:
            output = output.loc[pd.to_datetime(output["date"]) >= pd.Timestamp(start)]
        if end is not None:
            output = output.loc[pd.to_datetime(output["date"]) <= pd.Timestamp(end)]
        return output.sort_values(["date", "symbol", "action_type"]).reset_index(drop=True)

    def load_panel(
        self,
        symbols: tuple[str, ...] | None = None,
        start: date | None = None,
        end: date | None = None,
        *,
        unlock_key: str | None = None,
    ) -> pd.DataFrame:
        """Load the daily-bar panel, truncated at the final-test boundary.

        This is the single modeling-facing entry point for price data.
        """
        available = self.list_symbols()
        wanted = tuple(symbols) if symbols else available
        missing = sorted(set(wanted) - set(available))
        frames = []
        for symbol in wanted:
            path = safe_child_path(self.prices_dir, f"{symbol}.parquet")
            if path.exists():
                frames.append(pd.read_parquet(path))
        if not frames:
            raise DataError(
                f"no price data in catalog for {wanted!r}; run `edgestack data download` first"
            )
        panel = pd.concat(frames, ignore_index=True)
        if missing:
            log_event(log, 30, "symbols missing from catalog", missing=",".join(missing))
        if start is not None:
            panel = panel.loc[panel["date"] >= pd.Timestamp(start)]
        if end is not None:
            panel = panel.loc[panel["date"] <= pd.Timestamp(end)]
        panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
        return self.guard.apply(panel, unlock_key=unlock_key)

    def data_manifest_hash(self) -> str:
        """Deterministic content fingerprint of the stored price data."""
        import hashlib

        h = hashlib.sha256()
        for kind, directory in (
            ("prices", self.prices_dir),
            ("corporate_actions", self.corporate_actions_dir),
        ):
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.parquet")):
                h.update(f"{kind}/{path.name}".encode())
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        h.update(chunk)
        return h.hexdigest()

    # -- metadata --------------------------------------------------------------

    def record_experiment(
        self,
        kind: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        feature_set: str | None = None,
        trial_count: int | None = None,
        notes: str | None = None,
    ) -> str:
        experiment_id = uuid.uuid4().hex[:16]
        with self.connect() as con:
            con.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    experiment_id,
                    datetime.now(UTC),
                    kind,
                    self.cfg.config_hash(),
                    _git_head(),
                    self.data_manifest_hash(),
                    self.cfg.project.random_seed,
                    start_date,
                    end_date,
                    feature_set,
                    self.cfg.costs.scenario.value,
                    trial_count,
                    notes,
                ],
            )
        log_event(log, 20, "experiment recorded", experiment_id=experiment_id, kind=kind)
        return experiment_id

    def update_trial_count(self, experiment_id: str, trial_count: int) -> None:
        with self.connect() as con:
            con.execute(
                "UPDATE experiments SET trial_count = ? WHERE experiment_id = ?",
                [trial_count, experiment_id],
            )

    def audit(
        self,
        event: str,
        *,
        reason: str | None = None,
        experiment_id: str | None = None,
        **detail: Any,
    ) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO audit_log VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    uuid.uuid4().hex,
                    datetime.now(UTC),
                    event,
                    reason,
                    experiment_id,
                    self.cfg.config_hash(),
                    json.dumps(detail, default=str),
                ],
            )

    def audit_events(self, event: str) -> pd.DataFrame:
        with self.connect() as con:
            return con.execute("SELECT * FROM audit_log WHERE event = ? ORDER BY ts", [event]).df()


def _git_head() -> str | None:
    """Current commit hash, if running inside a git checkout."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except OSError:  # pragma: no cover
        return None
