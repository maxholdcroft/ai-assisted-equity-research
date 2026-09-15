"""Streamlit interface for the AI-Assisted Equity Research Platform."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.research_pipeline import ResearchRun, run_research


st.set_page_config(
    page_title="AI Equity Research",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {max-width: 1280px; padding-top: 2.5rem;}
        [data-testid="stMetric"] {
            background: #111827;
            border: 1px solid #273449;
            border-radius: 10px;
            padding: 14px;
        }
        [data-testid="stMetricLabel"] {color: #9ca3af;}
        [data-testid="stMetricValue"] {color: #f9fafb;}
        .status-banner {
            border-radius: 10px;
            padding: 14px 18px;
            margin: 12px 0 22px 0;
            font-weight: 600;
        }
        .status-approved {background: #123b2a; color: #b7f7cf;}
        .status-qualified {background: #473512; color: #fde68a;}
        .status-review {background: #4a1d25; color: #fecdd3;}
        .source-note {color: #9ca3af; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def format_percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def format_number(value: float | None) -> str:
    if value is None:
        return "—"

    absolute_value = abs(value)
    if absolute_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}bn"
    if absolute_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}m"
    if absolute_value >= 1_000:
        return f"{value:,.0f}"
    return f"{value:.2f}"


def render_evidence_items(items: list) -> None:
    for item in items:
        st.markdown(f"- {item.statement}")
        st.caption(f"Evidence: {', '.join(item.evidence_paths)}")


def status_css_class(status: str) -> str:
    return {
        "APPROVED": "status-approved",
        "QUALIFIED": "status-qualified",
        "REQUIRES_REVIEW": "status-review",
    }[status]


def price_chart(research_run: ResearchRun) -> None:
    prices = research_run.raw_data.price_history.copy()
    column = "Adj Close" if "Adj Close" in prices.columns else "Close"

    figure = px.line(
        prices,
        x=prices.index,
        y=column,
        title="Two-Year Adjusted Price History",
        labels={"x": "", column: "Adjusted close"},
        template="plotly_dark",
    )
    figure.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=45, b=0),
        showlegend=False,
    )
    st.plotly_chart(figure, use_container_width=True)


def drawdown_chart(research_run: ResearchRun) -> None:
    prices = research_run.raw_data.price_history.copy()
    column = "Adj Close" if "Adj Close" in prices.columns else "Close"
    drawdown = prices[column] / prices[column].cummax() - 1

    figure = px.area(
        x=drawdown.index,
        y=drawdown,
        title="Drawdown from Prior Peak",
        labels={"x": "", "y": "Drawdown"},
        template="plotly_dark",
    )
    figure.update_traces(line_color="#ef4444", fillcolor="rgba(239, 68, 68, 0.22)")
    figure.update_yaxes(tickformat=".0%")
    figure.update_layout(height=320, margin=dict(l=0, r=0, t=45, b=0))
    st.plotly_chart(figure, use_container_width=True)


def render_report(research_run: ResearchRun) -> None:
    report = research_run.report
    snapshot = report.snapshot
    analyst = report.analyst_run.assessment
    critic = report.critic_run.assessment

    st.divider()

    company_name = snapshot.company_name or snapshot.ticker
    st.subheader(f"{company_name} · {snapshot.ticker}")
    listing = " · ".join(
        value for value in [snapshot.exchange, snapshot.trading_currency] if value
    )
    if listing:
        st.caption(listing)

    st.markdown(
        (
            f'<div class="status-banner {status_css_class(report.publication_status)}">'
            f"{report.publication_status.replace('_', ' ')} — "
            f"{report.publication_reason}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    overview_tab, financials_tab, quantitative_tab, thesis_tab, review_tab = st.tabs(
        [
            "Overview",
            "Financials",
            "Quantitative Analysis",
            "AI Investment Thesis",
            "Risk Review",
        ]
    )

    with overview_tab:
        market = snapshot.market_performance
        financials = snapshot.financial_performance
        profitability = snapshot.profitability

        metric_columns = st.columns(4)
        metric_columns[0].metric(
            "Latest adjusted close",
            (
                f"{market.latest_adjusted_close:,.2f} "
                f"{snapshot.trading_currency or ''}"
            )
            if market.latest_adjusted_close is not None
            else "—",
        )
        metric_columns[1].metric("1Y return", format_percent(market.one_year_return))
        metric_columns[2].metric(
            "Revenue growth",
            format_percent(financials.revenue_growth),
        )
        metric_columns[3].metric("Net margin", format_percent(profitability.net_margin))

        price_chart(research_run)
        st.caption(
            "Price data are retrieved from Yahoo Finance. Returns use adjusted "
            "closing prices and are calculated in Python."
        )

    with financials_tab:
        st.warning(
            "Financial-statement currency is not independently verified. "
            "Amounts are shown as provider-reported units with no currency conversion."
        )

        financial_table = pd.DataFrame(
            {
                "Metric": [
                    "Revenue",
                    "Revenue growth",
                    "Net income",
                    "Net income growth",
                    "Operating margin",
                    "Net margin",
                    "Operating cash flow",
                    "Free cash flow",
                    "Free cash flow margin",
                    "Cash and short-term investments",
                    "Total debt",
                    "Debt to equity",
                ],
                "Value": [
                    format_number(snapshot.financial_performance.revenue),
                    format_percent(snapshot.financial_performance.revenue_growth),
                    format_number(snapshot.financial_performance.net_income),
                    format_percent(snapshot.financial_performance.net_income_growth),
                    format_percent(snapshot.profitability.operating_margin),
                    format_percent(snapshot.profitability.net_margin),
                    format_number(snapshot.cash_flow.operating_cash_flow),
                    format_number(snapshot.cash_flow.free_cash_flow),
                    format_percent(snapshot.cash_flow.free_cash_flow_margin),
                    format_number(snapshot.balance_sheet.cash_and_equivalents),
                    format_number(snapshot.balance_sheet.total_debt),
                    format_percent(snapshot.balance_sheet.debt_to_equity),
                ],
            }
        )
        st.dataframe(financial_table, hide_index=True, use_container_width=True)
        st.caption(
            f"Latest income-statement fiscal period: "
            f"{snapshot.financial_performance.latest_fiscal_period or 'unavailable'}"
        )

    with quantitative_tab:
        market = snapshot.market_performance
        risk = snapshot.risk_metrics

        metric_columns = st.columns(4)
        metric_columns[0].metric(
            "3M return",
            format_percent(market.three_month_return),
        )
        metric_columns[1].metric(
            "Annualised volatility",
            format_percent(risk.annualised_volatility),
        )
        metric_columns[2].metric(
            "Maximum drawdown",
            format_percent(risk.maximum_drawdown),
        )
        metric_columns[3].metric(
            "50/200-day signal",
            market.moving_average_signal.upper()
            if market.moving_average_signal
            else "—",
        )

        drawdown_chart(research_run)
        st.caption(
            "Volatility and drawdown are calculated from the latest one trading "
            "year of adjusted daily closes."
        )

    with thesis_tab:
        recommendation_column, confidence_column = st.columns(2)
        recommendation_column.metric("Analyst recommendation", analyst.recommendation)
        confidence_column.metric("Analyst confidence", analyst.confidence)

        st.subheader("Investment thesis")
        st.write(analyst.investment_thesis)

        left, right = st.columns(2)
        with left:
            st.subheader("Positives")
            render_evidence_items(analyst.positives)

            st.subheader("Potential catalysts")
            render_evidence_items(analyst.catalysts)

        with right:
            st.subheader("Negatives")
            render_evidence_items(analyst.negatives)

            st.subheader("Key risks")
            render_evidence_items(analyst.key_risks)

        st.subheader("Valuation assessment")
        st.write(analyst.valuation_assessment)

        with st.expander("Recommendation rationale and limitations"):
            st.write(analyst.recommendation_rationale)
            st.markdown("**What could change the view**")
            for factor in analyst.factors_that_change_recommendation:
                st.markdown(f"- {factor}")
            st.markdown("**Data limitations**")
            for limitation in analyst.data_limitations:
                st.markdown(f"- {limitation}")

    with review_tab:
        st.subheader("Deterministic evidence validation")
        validation = report.deterministic_validation
        if validation.passed:
            st.success("Passed: every AI evidence reference maps to the snapshot.")
        else:
            st.error("Failed: the analyst output contains invalid or missing support.")
            for finding in validation.findings:
                st.markdown(
                    f"- **{finding.severity}** · `{finding.location}` · "
                    f"{finding.explanation}"
                )

        st.subheader("Independent AI risk review")
        review_columns = st.columns(2)
        review_columns[0].metric(
            "Recommendation supported?",
            critic.analyst_recommendation_supported,
        )
        review_columns[1].metric(
            "Critic confidence",
            critic.suggested_confidence,
        )

        st.write(critic.final_review_commentary)

        for heading, findings in [
            ("Unsupported claims", critic.unsupported_claims),
            ("Overlooked risks or limitations", critic.overlooked_risks_or_limitations),
            ("Disagreements", critic.disagreements),
        ]:
            with st.expander(heading):
                if not findings:
                    st.write("None identified.")
                for finding in findings:
                    st.markdown(
                        f"**{finding.severity} — {finding.claim_or_issue}**"
                    )
                    st.write(finding.explanation)
                    if finding.relevant_snapshot_paths:
                        st.caption(
                            "Relevant evidence: "
                            + ", ".join(finding.relevant_snapshot_paths)
                        )

        st.info(
            "Educational research demonstration only. This application does not "
            "provide investment advice, and a publication gate is not a substitute "
            "for professional investment due diligence."
        )


st.title("AI-Assisted Equity Research")
st.caption(
    "Deterministic financial analysis, evidence-constrained local AI interpretation, "
    "and an independent AI risk review."
)

with st.form("research_form"):
    ticker = st.text_input(
        "Ticker",
        value="AZN.L",
        help="Examples: MSFT, AAPL, AZN.L. UK listings typically use the .L suffix.",
    )
    submitted = st.form_submit_button("Run research", type="primary")

if submitted:
    try:
        with st.spinner("Retrieving data, calculating metrics, and running local review..."):
            st.session_state.research_run = run_research(ticker)
    except Exception as exc:
        st.error(f"Analysis could not be completed: {exc}")

if "research_run" in st.session_state:
    render_report(st.session_state.research_run)
else:
    st.info("Enter a ticker and select Run research to begin.")