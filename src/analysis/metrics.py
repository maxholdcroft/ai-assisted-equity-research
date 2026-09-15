"""Deterministic market, risk, and fundamental metric calculations."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data.yahoo_finance import RawCompanyData


TRADING_DAYS_PER_YEAR = 252
THREE_MONTH_TRADING_DAYS = 63
FIFTY_DAY_WINDOW = 50
TWO_HUNDRED_DAY_WINDOW = 200


def calculate_metrics(data: RawCompanyData) -> dict[str, Any]:
    """Calculate metrics from raw provider data without using LLMs."""

    prices = _adjusted_close_series(data.price_history)

    return {
        "market_performance": _market_performance(prices),
        "risk_metrics": _risk_metrics(prices),
        "financial_performance": _financial_performance(data.income_statement),
        "profitability": _profitability(data.income_statement),
        "cash_flow": _cash_flow_metrics(data.cash_flow, data.income_statement),
        "balance_sheet": _balance_sheet_metrics(data.balance_sheet),
    }


def _adjusted_close_series(price_history: pd.DataFrame) -> pd.Series:
    """Use adjusted close so returns account for splits and dividends."""

    column = "Adj Close" if "Adj Close" in price_history.columns else "Close"
    return pd.to_numeric(price_history[column], errors="coerce").dropna()


def _market_performance(prices: pd.Series) -> dict[str, Any]:
    return {
        "latest_adjusted_close": _latest_value(prices),
        "one_year_return": _period_return(prices, TRADING_DAYS_PER_YEAR),
        "three_month_return": _period_return(prices, THREE_MONTH_TRADING_DAYS),
        "moving_average_50": _moving_average(prices, FIFTY_DAY_WINDOW),
        "moving_average_200": _moving_average(prices, TWO_HUNDRED_DAY_WINDOW),
        "moving_average_signal": _moving_average_signal(prices),
    }


def _risk_metrics(prices: pd.Series) -> dict[str, Any]:
    daily_returns = prices.pct_change().dropna().tail(TRADING_DAYS_PER_YEAR)

    if daily_returns.empty:
        annualised_volatility = None
    else:
        annualised_volatility = float(
            daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

    one_year_prices = prices.tail(TRADING_DAYS_PER_YEAR + 1)
    if one_year_prices.empty:
        maximum_drawdown = None
    else:
        drawdowns = one_year_prices / one_year_prices.cummax() - 1
        maximum_drawdown = float(drawdowns.min())

    return {
        "annualised_volatility": annualised_volatility,
        "maximum_drawdown": maximum_drawdown,
    }


def _financial_performance(income_statement: pd.DataFrame) -> dict[str, Any]:
    latest, previous = _latest_two_periods(income_statement)

    revenue_latest = _statement_value(
        income_statement,
        ["Total Revenue", "Operating Revenue"],
        latest,
    )
    revenue_previous = _statement_value(
        income_statement,
        ["Total Revenue", "Operating Revenue"],
        previous,
    )
    net_income_latest = _statement_value(
        income_statement,
        ["Net Income", "Net Income Common Stockholders"],
        latest,
    )
    net_income_previous = _statement_value(
        income_statement,
        ["Net Income", "Net Income Common Stockholders"],
        previous,
    )

    return {
        "latest_fiscal_period": _format_period(latest),
        "revenue": revenue_latest,
        "revenue_growth": _growth_rate(revenue_latest, revenue_previous),
        "net_income": net_income_latest,
        "net_income_growth": _growth_rate(net_income_latest, net_income_previous),
    }


def _profitability(income_statement: pd.DataFrame) -> dict[str, Any]:
    latest, _ = _latest_two_periods(income_statement)

    revenue = _statement_value(
        income_statement,
        ["Total Revenue", "Operating Revenue"],
        latest,
    )
    operating_income = _statement_value(
        income_statement,
        ["Operating Income", "EBIT"],
        latest,
    )
    net_income = _statement_value(
        income_statement,
        ["Net Income", "Net Income Common Stockholders"],
        latest,
    )

    return {
        "operating_margin": _ratio(operating_income, revenue),
        "net_margin": _ratio(net_income, revenue),
    }


def _cash_flow_metrics(
    cash_flow: pd.DataFrame,
    income_statement: pd.DataFrame,
) -> dict[str, Any]:
    latest_cash_flow_period, _ = _latest_two_periods(cash_flow)
    latest_income_period, _ = _latest_two_periods(income_statement)

    operating_cash_flow = _statement_value(
        cash_flow,
        ["Operating Cash Flow", "Total Cash From Operating Activities"],
        latest_cash_flow_period,
    )
    free_cash_flow = _statement_value(
        cash_flow,
        ["Free Cash Flow"],
        latest_cash_flow_period,
    )
    revenue = _statement_value(
        income_statement,
        ["Total Revenue", "Operating Revenue"],
        latest_income_period,
    )

    return {
        "operating_cash_flow": operating_cash_flow,
        "free_cash_flow": free_cash_flow,
        "free_cash_flow_margin": _ratio(free_cash_flow, revenue),
    }


def _balance_sheet_metrics(balance_sheet: pd.DataFrame) -> dict[str, Any]:
    latest, _ = _latest_two_periods(balance_sheet)

    cash = _statement_value(
        balance_sheet,
        [
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents",
            "Cash Financial",
        ],
        latest,
    )
    total_debt = _statement_value(
        balance_sheet,
        ["Total Debt"],
        latest,
    )
    shareholders_equity = _statement_value(
        balance_sheet,
        ["Stockholders Equity", "Common Stock Equity"],
        latest,
    )

    return {
        "cash_and_equivalents": cash,
        "total_debt": total_debt,
        "debt_to_equity": _ratio(total_debt, shareholders_equity),
    }


def _latest_two_periods(
    statement: pd.DataFrame,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if statement.empty:
        return None, None

    periods = sorted(statement.columns, reverse=True)
    latest = periods[0] if periods else None
    previous = periods[1] if len(periods) > 1 else None
    return latest, previous


def _statement_value(
    statement: pd.DataFrame,
    candidate_rows: Iterable[str],
    period: pd.Timestamp | None,
) -> float | None:
    if statement.empty or period is None:
        return None

    for row_name in candidate_rows:
        if row_name in statement.index:
            value = pd.to_numeric(statement.loc[row_name, period], errors="coerce")
            if pd.notna(value):
                return float(value)

    return None


def _period_return(prices: pd.Series, trading_days: int) -> float | None:
    if len(prices) <= trading_days:
        return None

    return float(prices.iloc[-1] / prices.iloc[-(trading_days + 1)] - 1)


def _moving_average(prices: pd.Series, window: int) -> float | None:
    if len(prices) < window:
        return None

    return float(prices.tail(window).mean())


def _moving_average_signal(prices: pd.Series) -> str | None:
    average_50 = _moving_average(prices, FIFTY_DAY_WINDOW)
    average_200 = _moving_average(prices, TWO_HUNDRED_DAY_WINDOW)

    if average_50 is None or average_200 is None:
        return None

    return "positive" if average_50 > average_200 else "negative"


def _growth_rate(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous <= 0:
        return None

    return float(current / previous - 1)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None

    return float(numerator / denominator)


def _latest_value(values: pd.Series) -> float | None:
    if values.empty:
        return None

    return float(values.iloc[-1])


def _format_period(period: pd.Timestamp | None) -> str | None:
    if period is None:
        return None

    return pd.Timestamp(period).date().isoformat()