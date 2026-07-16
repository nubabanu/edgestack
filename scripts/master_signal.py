"""Deprecated master exporter containing canonical target weights only."""

from __future__ import annotations

import json

from edgestack.config import load_config
from edgestack.data.catalog import DataCatalog, atomic_write_bytes
from edgestack.recommendation.compatibility import master_projection
from edgestack.recommendation.service import CanonicalBundleRepository


def main() -> int:
    cfg = load_config("configs/live.yaml")
    catalog = DataCatalog(cfg)
    bundle = CanonicalBundleRepository(catalog.artifacts_dir).latest()
    payload = master_projection(bundle)
    path = catalog.artifacts_dir / "master_signal.json"
    atomic_write_bytes(path, json.dumps(payload, indent=2).encode())
    print(f"deprecated master projection -> {path}; bundle={bundle.bundle_hash[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
