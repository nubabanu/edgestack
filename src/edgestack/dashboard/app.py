"""Minimal Streamlit dashboard: latest signals, edges, lifecycle states.

Run with:  streamlit run src/edgestack/dashboard/app.py
Everything shown is research / paper-trading output, not investment advice.
"""

from __future__ import annotations


def main() -> None:  # pragma: no cover - interactive UI
    import pandas as pd
    import streamlit as st

    from edgestack.config import load_config
    from edgestack.data.catalog import DataCatalog
    from edgestack.discovery.edge_store import current_statuses, load_edges
    from edgestack.exceptions import DataError
    from edgestack.reporting.signal_report import load_report
    from edgestack.types import EdgeStatus

    st.set_page_config(page_title="EdgeStack", layout="wide")
    st.title("EdgeStack — research dashboard")
    st.warning(
        "Research / paper-trading output only. Nothing here is investment "
        "advice; free data sources carry survivorship bias."
    )

    cfg = load_config()
    catalog = DataCatalog(cfg)

    try:
        report = load_report(catalog)
    except DataError:
        st.info("No signal reports yet — run `edgestack signals generate`.")
        report = None

    if report is not None:
        st.header(f"Signals for {report.as_of_date}")
        for warning in report.universe_warnings:
            st.caption(f"⚠ {warning}")
        col_long, col_short = st.columns(2)

        def _frame(candidates):
            return pd.DataFrame(
                [
                    {
                        "symbol": c.symbol,
                        "conviction": c.conviction_score,
                        "P(net>0)": c.calibrated_probability_of_positive_net_return,
                        "E[net]": c.expected_net_return,
                        "hold": c.recommended_holding_sessions,
                        "stop": c.risk.stop_price,
                        "target": c.risk.target_1,
                        "R:R": c.risk.reward_to_risk,
                    }
                    for c in candidates
                ]
            )

        with col_long:
            st.subheader("Long candidates")
            st.dataframe(_frame(report.long_candidates))
        with col_short:
            st.subheader("Short candidates (research only)")
            st.dataframe(_frame(report.short_candidates))

        st.subheader(f"Abstentions ({len(report.abstentions)})")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "symbol": a.symbol,
                        "side": a.side.value if a.side else "",
                        "reasons": "; ".join(a.reasons),
                    }
                    for a in report.abstentions
                ]
            )
        )
        if report.long_candidates:
            st.subheader("Top candidate explanation")
            st.text(report.long_candidates[0].explanation)

    st.header("Edges")
    edges = load_edges(catalog, statuses=tuple(EdgeStatus))
    if edges:
        statuses = current_statuses(catalog).set_index("edge_id")["status"].to_dict()
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "name": e.identity.name,
                        "status": statuses.get(e.identity.edge_id, e.lifecycle.status.value),
                        "net/trade": e.stats.net_mean_return,
                        "q": e.stats.q_value,
                        "DSR": e.stats.deflated_sharpe_ratio,
                        "n": e.stats.sample_size,
                        "folds+": e.robustness.stability_score,
                    }
                    for e in edges
                ]
            )
        )
    else:
        st.info("No edges yet — run discovery and validation.")


if __name__ == "__main__":  # pragma: no cover
    main()
