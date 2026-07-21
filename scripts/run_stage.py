"""Run one nightly-chain stage, record its outcome, propagate its exit code.

Wraps a script invocation so the bat chain stops discarding exit codes:

    python scripts/run_stage.py --stage tranche_watch -- scripts/tranche_watch.py [args...]

The child runs in its own process, so even a native crash (access violation,
GIL abort) is captured as a nonzero exit and recorded in
``artifacts/nightly_status.json``. ``--core`` only documents intent; core vs
auxiliary classification for the aggregate exit code lives in
``nightly_status.CORE_STAGES``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

try:
    from scripts import nightly_status
except ImportError:  # running as `python scripts/run_stage.py`
    import nightly_status  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, help="stage name for the status ledger")
    parser.add_argument(
        "--core", action="store_true", help="documentation only; see nightly_status.CORE_STAGES"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- script.py [args...]")
    args = parser.parse_args()
    command = [c for c in args.command if c != "--"]
    if not command:
        parser.error("no child command given (expected: -- script.py [args...])")
    started = time.monotonic()
    try:
        proc = subprocess.run([sys.executable, *command], cwd=ROOT)
        code = proc.returncode
    except OSError as exc:
        print(f"run_stage: failed to launch {command}: {exc}")
        code = 1
    nightly_status.record_stage(
        args.stage,
        "ok" if code == 0 else "failed",
        exit_code=code,
        duration_s=time.monotonic() - started,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
