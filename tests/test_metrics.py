from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.analysis.metrics import calculate_metrics
from src.data.yahoo_finance import RawCompanyData


def make_company_data() -> RawCompanyData:
    dates = pd.date_range("2025-01-01", periods=253, freq="B")
    prices = pd.DataFrame(
        {"Adj Close": np.linspace(100.0, 200.0, len(dates))},
        index=dates,
    )

    current_period = pd.Timestamp("2025-12-31")
    previous_period = pd.Timestamp("2024-12-31")

    income_statement = pd.DataFrame(
        {
            current_period: [200.0, 50.0, 40.0],
            previous_period: [100.0, 25.0, 20.0],
        },
        index=["Total Revenue", "Operating Income", "Net Income"],
    )

    cash_flow = pd.DataFrame(
        {
            current_period: [60.0, 30.0],
            previous_period: [40.0, 20.0],
        },
        index=["Operating Cash Flow", "Free Cash Flow"],
    )

    balance_sheet = pd.DataFrame(
        {
            current_period: [40.0, 50.0, 100.0],
            previous_period: [30.0, 60.0, 90.0],
        },
        index=[
            "Cash And Cash Equivalents",
            "Total Debt",
            "Stockholders Equity",
        ],
    )

    return RawCompanyData(
        ticker="TEST",
        fetched_at=datetime.now(timezone.utc).isoformat(),
        price_history=prices,
        income_statement=income_statement,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
    )


def test_calculates_market_and_fundamental_metrics() -> None:
    metrics = calculate_metrics(make_company_data())

    assert metrics["market_performance"]["one_year_return"] == pytest.approx(1.0)
    assert metrics["market_performance"]["three_month_return"] > 0
    assert metrics["market_performance"]["moving_average_signal"] == "positive"

    assert metrics["financial_performance"]["revenue"] == 200.0
    assert metrics["financial_performance"]["revenue_growth"] == pytest.approx(1.0)
    assert metrics["financial_performance"]["net_income_growth"] == pytest.approx(1.0)

    assert metrics["profitability"]["operating_margin"] == pytest.approx(0.25)
    assert metrics["profitability"]["net_margin"] == pytest.approx(0.20)
    assert metrics["cash_flow"]["free_cash_flow_margin"] == pytest.approx(0.15)
    assert metrics["balance_sheet"]["debt_to_equity"] == pytest.approx(0.50)


def test_missing_input_remains_unavailable() -> None:
    data = make_company_data()
    data.balance_sheet = data.balance_sheet.drop(index="Total Debt")

    metrics = calculate_metrics(data)

    assert metrics["balance_sheet"]["total_debt"] is None
    assert metrics["balance_sheet"]["debt_to_equity"] is None