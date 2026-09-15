# AI-Assisted Equity Research Platform

A local-first equity-research MVP that combines deterministic Python analysis with evidence-constrained local LLM interpretation and an independent AI risk review.

The project is designed to demonstrate responsible AI use in an investment-research setting:

> Python is used to calculate facts and metrics. The LLM only interprets them.

> Educational research demonstration only. This application does not provide investment advice.

## Features

- Retrieves two years of market data and annual financial statements through `yfinance`
- Calculates returns, volatility, drawdown, moving-average signal, growth, margins, cash flow, and leverage in Python
- Creates a validated, JSON-ready `CompanySnapshot` as the sole factual source for AI stages
- Uses a local `qwen3:8b` model via Ollama for investment interpretation
- Requires structured JSON responses validated with Pydantic
- Runs deterministic evidence-path validation before publishing AI output
- Uses an independent local AI critic to challenge unsupported claims, missing limitations, and confidence levels
- Applies a publication gate: `APPROVED`, `QUALIFIED`, or `REQUIRES_REVIEW`
- Displays data, AI thesis, and risk review separately in Streamlit

## Architecture

```text
Ticker
  ↓
Yahoo Finance / yfinance
  ↓
RawCompanyData + validation
  ↓
Deterministic Python metrics
  ↓
Validated CompanySnapshot
  ├── Local AI Investment Analyst
  └── Deterministic evidence validation
          ↓
    Independent Local AI Critic
          ↓
Publication gate + Streamlit research report
```

## Technology

- Python 3.11
- Streamlit
- Pandas, NumPy, Plotly
- yfinance / Yahoo Finance
- Pydantic
- Ollama with Qwen3 8B
- Pytest

## Installation

Clone the repository, then create and activate the project environment:

```zsh
conda create -n equity-research python=3.11 -y
conda activate equity-research
pip install -r requirements.txt
```

Install the required local model:

```zsh
ollama pull qwen3:8b
```

Ollama must be running locally at `http://localhost:11434`. If necessary:

```zsh
ollama serve
```

Run the application:

```zsh
streamlit run app.py
```

## Usage

Enter a ticker and select **Run research**.

Examples:

```text
AAPL
JPM
TSLA
AZN.L
HSBA.L
```

UK listings generally require the `.L` suffix.

## Methodology

### Deterministic analysis

The application calculates the following in Python rather than asking the LLM to calculate them:

| Area | Metrics |
|---|---|
| Market performance | 1Y return, 3M return, 50/200-day moving averages |
| Risk | Annualised volatility, maximum drawdown |
| Financial performance | Revenue, revenue growth, net income, net-income growth |
| Profitability | Operating margin, net margin |
| Cash flow | Operating cash flow, free cash flow, free-cash-flow margin |
| Balance sheet | Cash and short-term investments, total debt, debt-to-equity |

Unavailable provider fields remain `null`; they are never converted to zero or inferred by the model.

### AI controls

The local analyst receives only the validated `CompanySnapshot`. It is instructed not to use external knowledge, invent figures, calculate metrics, or infer valuation where financial-statement currency is unverified.

The application then applies two checks:

1. **Deterministic evidence validation** verifies that every cited evidence path exists in the snapshot.
2. **Independent AI risk review** challenges unsupported claims, qualitative overreach, missing limitations, and recommendation confidence.

The publication gate does not overwrite the analyst. It preserves the full audit trail and marks the result:

- `APPROVED` — deterministic validation passed and the critic supports the recommendation
- `QUALIFIED` — the critic partially supports it
- `REQUIRES_REVIEW` — deterministic validation fails, the critic rejects the recommendation, or high-severity concerns are found

## Testing

### Automated tests

```zsh
pytest -q
```

The current tests cover deterministic metric calculations and ensure missing inputs remain unavailable rather than being fabricated.

### Cross-company batch testing

The data and deterministic pipeline were tested successfully across ten companies:

| Segment | Tickers |
|---|---|
| US technology | AAPL |
| US financials | JPM |
| US industrials | CAT |
| US consumer | COST |
| US high-growth / volatile | TSLA |
| US healthcare | UNH |
| UK healthcare | AZN.L |
| UK financials | HSBA.L |
| UK energy | BP.L |
| UK materials | RIO.L |

All ten completed data retrieval and deterministic analysis without runtime failure. JPM and HSBA.L returned no generic operating margin, which is retained as unavailable because this metric is not directly comparable for banks.

A full local-AI batch run also completed for all ten companies. Most analyst recommendations were `HOLD` with low critic confidence and were gated as `REQUIRES_REVIEW`; this is an intentional result of the limited, valuation-free evidence set rather than a forced directional recommendation.

Run the checks locally:

```zsh
python -m work.batch_test
python -m work.batch_test --full-ai
```

## Limitations

- Yahoo Finance data may be incomplete, delayed, or unsuitable for production investment workflows.
- Financial-statement currency is not independently verified in this MVP, so valuation multiples are deliberately withheld.
- The application does not include forecasts, news, estimates, peer comparisons, transcripts, or SEC/RNS filings.
- Generic operating margin is not meaningful for banks; a production version would use bank-specific measures such as net interest margin, CET1, ROE, and loan-loss provisions.
- Local LLM output remains probabilistic. Structured output, deterministic validation, and an AI critic reduce risk but do not eliminate it.
- This is not financial advice.

## Future Improvements

- Add a licensed production-grade financial-data provider
- Verify financial-statement currency and add carefully aligned valuation metrics
- Add sector-specific metric frameworks, starting with banks
- Add SEC/RNS filing retrieval and citation-level evidence
- Compare the target company with a peer group
- Persist analyses and model/audit metadata
- Add GitHub Actions for automated testing
- Add benchmark datasets for evaluating analyst and critic quality
