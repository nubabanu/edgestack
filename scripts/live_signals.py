"""Deprecated board exporter sourced exclusively from the canonical V2 bundle."""

from __future__ import annotations

import json

from edgestack.config import load_config
from edgestack.data.catalog import DataCatalog, atomic_write_bytes
from edgestack.recommendation.compatibility import board_projection
from edgestack.recommendation.service import CanonicalBundleRepository


def main() -> int:
    cfg = load_config("configs/live.yaml")
    catalog = DataCatalog(cfg)
    bundle = CanonicalBundleRepository(catalog.artifacts_dir).latest()
    payload = board_projection(bundle)
    path = catalog.artifacts_dir / "live_board.json"
    atomic_write_bytes(path, json.dumps(payload, indent=2).encode())
    print(
        f"deprecated board projection -> {path}; rows=0, "
        f"watchlist={len(payload['watchlist_v2'])}, bundle={bundle.bundle_hash[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
