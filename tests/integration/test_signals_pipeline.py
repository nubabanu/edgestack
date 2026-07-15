"""End-to-end signal generation on a synthetic market with an injected edge.

Full path: synthetic panel -> catalog -> feature build -> restricted discovery
-> walk-forward validation -> edge store -> signal engine -> persisted report.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog
from edgestack.pipelines import run_signals_generate, run_signals_rank
from edgestack.types import CandidateStatus


def _last_thursday(cfg: EdgeStackConfig) -> date:
    from edgestack.pipelines import load_feature_frame

    features = load_feature_frame(cfg)
    assert isinstance(features, pd.DataFrame)
    dates = pd.to_datetime(features["date"]).drop_duplicates().sort_values()
    thursdays = dates[dates.dt.dayofweek == 3]
    return thursdays.iloc[-1].date()


def test_signals_generate_end_to_end(prepared: EdgeStackConfig, capsys) -> None:
    cfg = prepared
    as_of = _last_thursday(cfg)
    run_signals_generate(cfg, as_of=as_of)
    out = capsys.readouterr().out
    assert "RESEARCH OUTPUT" in out
    assert "survivorship bias" in out

    from edgestack.reporting.signal_report import load_report

    report = load_report(DataCatalog(cfg), as_of)
    assert report.as_of_date == as_of

    # The injected Thursday-signal edge should produce long candidates.
    assert report.long_candidates, "expected long candidates on a Thursday"
    top = report.long_candidates[0]
    assert top.status is CandidateStatus.RESEARCH_CANDIDATE
    assert 45 <= top.conviction_score <= 100
    assert 0 <= top.calibrated_probability_of_positive_net_return <= 1
    assert top.recommended_holding_sessions == 1
    assert top.risk.stop_price < top.entry.ideal_high  # long stop below entry zone
    assert top.entry.earliest_timestamp.date() > as_of  # never same-close execution
    assert top.evidence and top.evidence[0].q_value <= 0.05
    assert "cal_weekday == 3.0" in top.evidence[0].edge
    assert top.explanation and "not investment advice" in top.explanation.lower()

    # Abstentions are explicit and carry reasons.
    assert report.abstentions
    assert all(a.reasons for a in report.abstentions)

    # Shorts, if any, must be research-only.
    for c in report.short_candidates:
        assert c.status is CandidateStatus.SHORT_RESEARCH_CANDIDATE


def test_non_thursday_dates_mostly_abstain(prepared: EdgeStackConfig, capsys) -> None:
    cfg = prepared
    from edgestack.pipelines import load_feature_frame

    features = load_feature_frame(cfg)
    dates = pd.to_datetime(features["date"]).drop_duplicates().sort_values()
    tuesday = dates[dates.dt.dayofweek == 1].iloc[-1].date()
    run_signals_generate(cfg, as_of=tuesday)
    capsys.readouterr()

    from edgestack.reporting.signal_report import load_report

    report = load_report(DataCatalog(cfg), tuesday)
    assert not report.long_candidates  # the validated edge fires on Thursdays only
    assert report.abstentions
    reasons = {r for a in report.abstentions for r in a.reasons}
    assert any("no validated edge applies" in r for r in reasons)


def test_signals_rank_renders(prepared: EdgeStackConfig, capsys) -> None:
    run_signals_rank(prepared, top=5)
    out = capsys.readouterr().out
    assert "LONG candidates" in out
    assert "abstentions" in out


def test_signal_side_effects_are_audited(prepared: EdgeStackConfig) -> None:
    catalog = DataCatalog(prepared)
    events = catalog.audit_events("signals_generate")
    assert len(events) >= 2
    # No test-period access should have happened anywhere in this pipeline.
    assert len(catalog.audit_events("test_set_accessed")) == 0
