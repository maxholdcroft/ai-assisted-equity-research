"""Validated, JSON-ready company snapshot used as LLM evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from src.analysis.metrics import calculate_metrics
from src.data.yahoo_finance import RawCompanyData


class MarketPerformance(BaseModel):
    latest_adjusted_close: float | None
    one_year_return: float | None
    three_month_return: float | None
    moving_average_50: float | None
    moving_average_200: float | None
    moving_average_signal: str | None


class RiskMetrics(BaseModel):
    annualised_volatility: float | None
    maximum_drawdown: float | None


class FinancialPerformance(BaseModel):
    latest_fiscal_period: str | None
    revenue: float | None
    revenue_growth: float | None
    net_income: float | None
    net_income_growth: float | None


class Profitability(BaseModel):
    operating_margin: float | None
    net_margin: float | None


class CashFlow(BaseModel):
    operating_cash_flow: float | None
    free_cash_flow: float | None
    free_cash_flow_margin: float | None


class BalanceSheet(BaseModel):
    cash_and_equivalents: float | None
    total_debt: float | None
    debt_to_equity: float | None


class DataQuality(BaseModel):
    provider: str
    fetched_at: str
    price_history_rows: int
    available_statements: dict[str, bool]
    missing_metric_count: int
    warnings: list[str] = Field(default_factory=list)


class CompanySnapshot(BaseModel):
    """The sole factual input available to the LLM stages."""

    ticker: str
    company_name: str | None
    exchange: str | None
    trading_currency: str | None
    financial_statement_currency: str | None
    analysis_timestamp: str
    market_performance: MarketPerformance
    risk_metrics: RiskMetrics
    financial_performance: FinancialPerformance
    profitability: Profitability
    cash_flow: CashFlow
    balance_sheet: BalanceSheet
    metric_provenance: dict[str, str]
    data_quality: DataQuality


def build_company_snapshot(data: RawCompanyData) -> CompanySnapshot:
    metrics = calculate_metrics(data)
    missing_metrics = _count_missing_values(metrics)

    return CompanySnapshot(
        ticker=data.ticker,
        company_name=data.metadata.get("company_name"),
        exchange=data.metadata.get("exchange"),
        trading_currency=data.metadata.get("trading_currency"),
        financial_statement_currency=None,
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        market_performance=metrics["market_performance"],
        risk_metrics=metrics["risk_metrics"],
        financial_performance=metrics["financial_performance"],
        profitability=metrics["profitability"],
        cash_flow=metrics["cash_flow"],
        balance_sheet=metrics["balance_sheet"],
        metric_provenance={
            "market_performance": (
                "Calculated in Python from Yahoo Finance adjusted closing prices."
            ),
            "risk_metrics": (
                "Calculated in Python from daily adjusted closing-price returns."
            ),
            "financial_performance": (
                "Retrieved from Yahoo Finance annual income statements; "
                "growth calculated in Python."
            ),
            "profitability": (
                "Calculated in Python from annual income-statement values."
            ),
            "cash_flow": (
                "Retrieved from Yahoo Finance annual cash-flow statements; "
                "margin calculated in Python."
            ),
            "balance_sheet": (
                "Retrieved from Yahoo Finance annual balance-sheet values; "
                "debt-to-equity calculated in Python."
            ),
        },
        data_quality=DataQuality(
            provider="Yahoo Finance via yfinance",
            fetched_at=data.fetched_at,
            price_history_rows=len(data.price_history),
            available_statements={
                "income_statement": not data.income_statement.empty,
                "balance_sheet": not data.balance_sheet.empty,
                "cash_flow_statement": not data.cash_flow.empty,
            },
            missing_metric_count=missing_metrics,
            warnings=data.warnings,
        ),
    )


def _count_missing_values(value: Any) -> int:
    """Count unavailable metric values recursively for visible data-quality reporting."""

    if isinstance(value, dict):
        return sum(_count_missing_values(item) for item in value.values())

    return int(value is None)