"""Cross-process coordination between nightly publication and research work."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from edgestack.exceptions import DataError

NIGHTLY_LOCK_NAME = "nightly.lock"
STALE_AFTER = timedelta(hours=4)


def nightly_is_active(artifacts_dir: Path, *, now: datetime | None = None) -> bool:
    path = artifacts_dir / NIGHTLY_LOCK_NAME
    if not path.exists():
        return False
    instant = now or datetime.now(UTC)
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return instant - modified <= STALE_AFTER


@contextmanager
def nightly_lock(artifacts_dir: Path) -> Iterator[Path]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / NIGHTLY_LOCK_NAME
    token = uuid.uuid4().hex
    encoded = json.dumps(
        {"token": token, "pid": os.getpid(), "started_at": datetime.now(UTC).isoformat()},
        sort_keys=True,
    ).encode()
    if path.exists() and not nightly_is_active(artifacts_dir):
        path.unlink(missing_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise DataError("another nightly publication holds the research coordination lock") from exc
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)
    try:
        yield path
    finally:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        if payload.get("token") == token:
            path.unlink(missing_ok=True)
