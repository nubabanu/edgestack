"""Final nightly-chain step: summarize the status ledger and alert on failure.

Prints a one-screen stage summary, sends ONE alert through both channels
(Telegram + Windows toast; each a graceful no-op when unavailable) when the
run is not fully ok, and exits with the aggregate code (0 ok / 1 degraded /
2 core failure) so Task Scheduler sees the truth.
"""

from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path

try:
    from scripts import nightly_status
except ImportError:  # running as `python scripts/nightly_status_report.py`
    import nightly_status  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
TOAST_SCRIPT = ROOT / "scripts" / "notify_toast.ps1"


def _alert(text: str) -> None:
    """Best-effort push to both channels; never changes the exit code."""
    with contextlib.suppress(Exception):
        try:
            from scripts import notify_telegram
        except ImportError:
            import notify_telegram  # type: ignore[no-redef]
        notify_telegram.send(text)
    with contextlib.suppress(Exception):
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(TOAST_SCRIPT),
                "-Title",
                "EdgeStack nightly status",
                "-Body",
                text[:250],
            ],
            timeout=30,
        )


def main() -> int:
    overall, code = nightly_status.aggregate()
    data = nightly_status.load_status()
    stages = data.get("stages", {})
    print(f"nightly chain {data.get('run_date', '?')}: {overall} (exit {code})")
    for name, entry in stages.items():
        duration = entry.get("duration_s")
        suffix = f" [{duration:.0f}s]" if isinstance(duration, int | float) else ""
        print(f"  {entry.get('status', '?'):>7}  {name}{suffix}")
    if code != 0:
        failed = [n for n, e in stages.items() if e.get("status") != "ok"]
        _alert(
            f"EdgeStack nightly {overall.upper()}: "
            f"failed stages: {', '.join(failed) or 'none recorded'}"
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
