"""Claim-withdrawal and authoritative-path regression tests."""

from __future__ import annotations

from pathlib import Path


def test_previously_accessed_period_is_not_described_as_never_seen() -> None:
    roots = (Path("README.md"), Path("docs"), Path("src"), Path("scripts"), Path("android/app/src"))
    prohibited = ("un" + "touched", "independent confirmation", "production-ready")
    offenders: list[str] = []
    for root in roots:
        paths = (root,) if root.is_file() else tuple(root.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in {".md", ".py", ".kt", ".json"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for phrase in prohibited:
                if phrase in text:
                    offenders.append(f"{path}:{phrase}")
    assert offenders == []


def test_actionable_surfaces_do_not_import_legacy_signal_engines() -> None:
    paths = (
        Path("src/edgestack/api/app.py"),
        Path("src/edgestack/dashboard/app.py"),
        Path("src/edgestack/paper/session.py"),
        Path("src/edgestack/recommendation/nightly.py"),
    )
    forbidden = (
        "reporting.signal_report",
        "strategies.composite",
        "features.calendar",
        "fundamentals_snapshot",
        "calendar_overlay",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert all(item not in text for item in forbidden), path


def test_legacy_exporters_are_canonical_projections_only() -> None:
    expected = {
        "live_signals.py": "board_projection",
        "make_picks.py": "picks_projection",
        "master_signal.py": "master_projection",
    }
    for name, projection in expected.items():
        text = (Path("scripts") / name).read_text(encoding="utf-8")
        assert projection in text
        assert "CanonicalBundleRepository" in text
        assert "signal_engine" not in text
