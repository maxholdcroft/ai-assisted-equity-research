"""Yahoo Finance data retrieval for the equity-research MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


@dataclass
class RawCompanyData:
    """Raw provider output, deliberately separate from calculated metrics."""

    ticker: str
    fetched_at: str
    price_history: pd.DataFrame
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    metadata: dict[str, str | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class YahooFinanceProvider:
    """Fetches raw market and annual financial-statement data from yfinance."""

    PRICE_HISTORY_PERIOD = "2y"

    def fetch_company_data(self, ticker: str) -> RawCompanyData:
        ticker = ticker.strip().upper()

        if not ticker:
            raise ValueError("Ticker cannot be empty.")

        instrument = yf.Ticker(ticker)
        warnings: list[str] = []

        price_history = self._fetch_price_history(instrument, ticker, warnings)
        income_statement = self._fetch_statement(
            instrument.get_income_stmt,
            "income statement",
            warnings,
        )
        balance_sheet = self._fetch_statement(
            instrument.get_balance_sheet,
            "balance sheet",
            warnings,
        )
        cash_flow = self._fetch_statement(
            instrument.get_cashflow,
            "cash-flow statement",
            warnings,
        )

        metadata = self._fetch_metadata(instrument, warnings)

        return RawCompanyData(
            ticker=ticker,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            price_history=price_history,
            income_statement=income_statement,
            balance_sheet=balance_sheet,
            cash_flow=cash_flow,
            metadata=metadata,
            warnings=warnings,
        )
    
    @staticmethod
    def _fetch_metadata(
        instrument: yf.Ticker,
        warnings: list[str],
    ) -> dict[str, str | None]:
        """Retrieve lightweight listing metadata after price history is loaded."""

        try:
            raw_metadata = instrument.get_history_metadata()
        except Exception:
            warnings.append("Listing metadata could not be retrieved from the provider.")
            return {}

        return {
            "company_name": raw_metadata.get("longName")
            or raw_metadata.get("shortName"),
            "exchange": raw_metadata.get("exchangeName"),
            "trading_currency": raw_metadata.get("currency"),
        }

    def _fetch_price_history(
        self,
        instrument: yf.Ticker,
        ticker: str,
        warnings: list[str],
    ) -> pd.DataFrame:
        try:
            history = instrument.history(
                period=self.PRICE_HISTORY_PERIOD,
                auto_adjust=False,
            )
        except Exception as exc:
            raise ValueError(
                f"Could not retrieve price history for '{ticker}'. "
                "Check the ticker and your internet connection."
            ) from exc

        if history.empty:
            raise ValueError(
                f"No price history was returned for '{ticker}'. "
                "Check the ticker format (for example, AZN.L for a London listing)."
            )

        if "Close" not in history.columns:
            raise ValueError(
                f"Price history for '{ticker}' does not contain a Close column."
            )

        if len(history) < 252:
            warnings.append(
                "Less than one trading year of price history was returned; "
                "one-year metrics may be unavailable."
            )

        return history

    @staticmethod
    def _fetch_statement(
        fetch_function,
        statement_name: str,
        warnings: list[str],
    ) -> pd.DataFrame:
        try:
            statement = fetch_function(freq="yearly", pretty=True)
        except Exception:
            warnings.append(
                f"{statement_name.capitalize()} could not be retrieved from the provider."
            )
            return pd.DataFrame()

        if statement.empty:
            warnings.append(
                f"No annual {statement_name} was returned by the provider."
            )

        return statement