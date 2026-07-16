"""Read-only dashboard over the atomically published canonical V2 bundle."""

from __future__ import annotations


def main() -> None:  # pragma: no cover - interactive UI
    import pandas as pd
    import streamlit as st

    from edgestack.config import load_config
    from edgestack.data.catalog import DataCatalog
    from edgestack.exceptions import DataError
    from edgestack.paper.canonical import load_monitoring_payload, load_paper_state
    from edgestack.recommendation.service import CanonicalBundleRepository

    st.set_page_config(page_title="EdgeStack V2", layout="wide")
    st.title("EdgeStack — canonical recommendation")
    st.warning(
        "Research and paper-trading output only. Not investment advice. "
        "Leverage can cause losses exceeding invested capital."
    )
    cfg = load_config()
    catalog = DataCatalog(cfg)
    repository = CanonicalBundleRepository(catalog.artifacts_dir)
    try:
        bundle = repository.latest()
    except DataError as exc:
        st.error(f"No verified canonical publication: {exc}")
        return

    base = bundle.base_recommendation
    recommendation = bundle.default_recommendation
    st.header(f"{recommendation.status.value} — session {bundle.session}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Effective leverage", f"{recommendation.effective_leverage:.2f}x")
    col2.metric("Expected volatility", f"{recommendation.expected_volatility:.1%}")
    col3.metric("One-day stress", f"{recommendation.one_day_stress_loss:.2%}")
    col4.metric("20-session stress", f"{recommendation.multi_session_stress_loss:.2%}")
    st.caption(
        f"Evidence {recommendation.evidence_grade.value} • fresh "
        f"{recommendation.freshness.is_fresh} • data {bundle.data_version[:12]} • "
        f"artifact {bundle.artifact_version[:12]}"
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Unlevered canonical base")
        st.dataframe(pd.DataFrame([weight.model_dump() for weight in base.unlevered_base_weights]))
    with right:
        st.subheader("Personalized target weights")
        st.dataframe(
            pd.DataFrame(
                [weight.model_dump() for weight in recommendation.personalized_target_weights]
            )
        )

    st.subheader("Binding constraints")
    st.dataframe(
        pd.DataFrame([constraint.model_dump() for constraint in recommendation.constraints])
    )
    if base.promoted_sleeves or base.promoted_compound_sleeves:
        st.subheader("Promoted sleeves")
        st.json(
            [
                sleeve.model_dump(mode="json")
                for sleeve in (*base.promoted_sleeves, *base.promoted_compound_sleeves)
            ]
        )
    if base.watchlist:
        st.subheader("Watchlist only — zero portfolio weight")
        st.dataframe(pd.DataFrame([entry.model_dump() for entry in base.watchlist]))

    state = recommendation.output_risk_state
    st.subheader("Drawdown state")
    st.write(
        {
            "state": state.drawdown_state.value,
            "drawdown": state.current_drawdown,
            "cash_latched": state.cash_latched,
            "reset_eligible": state.reset_eligible,
            "sessions_since_latch": state.sessions_since_latch,
        }
    )
    try:
        paper = load_paper_state(
            repository,
            initial_equity=bundle.default_risk_profile.account_equity,
            risk_state=state,
        )
        st.subheader("Canonical paper account")
        st.write(
            {
                "equity": paper.current_equity,
                "cash": paper.cash,
                "positions": len(paper.positions),
                "fills": len(paper.fills),
                "last_session": paper.last_session,
            }
        )
        if paper.realized_returns:
            st.line_chart(
                pd.DataFrame(
                    {
                        "session": [item.session for item in paper.realized_returns],
                        "equity": [item.ending_equity for item in paper.realized_returns],
                    }
                ).set_index("session")
            )
        st.subheader("Monitoring")
        st.json(load_monitoring_payload(repository))
    except DataError as exc:
        st.error(f"Atomic paper/monitoring read failed: {exc}")

    for warning in (*base.warnings, *recommendation.warnings):
        st.warning(warning)


if __name__ == "__main__":  # pragma: no cover
    main()
