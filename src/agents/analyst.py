"""Evidence-constrained local LLM investment analyst."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import requests
from pydantic import BaseModel, ConfigDict, Field

from src.analysis.snapshot import CompanySnapshot

EVIDENCE_SECTIONS = ("positives", "negatives", "catalysts", "key_risks")

def canonicalise_evidence_path(path: str) -> str:
    """Convert harmless LLM path formatting variants to dot notation."""

    return path.strip().replace("/", ".")


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(
        description="Interpretation or factual statement grounded in the snapshot."
    )
    evidence_paths: list[str] = Field(
        min_length=1,
        description=(
            "At least one exact snapshot path supporting the statement, "
            "for example 'financial_performance.revenue_growth'."
        ),
    )


class AnalystAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    investment_thesis: str
    positives: list[EvidenceItem]
    negatives: list[EvidenceItem]
    catalysts: list[EvidenceItem]
    key_risks: list[EvidenceItem]
    valuation_assessment: str
    recommendation: Literal["BUY", "HOLD", "SELL"]
    confidence: Literal["LOW", "MEDIUM", "HIGH"]
    recommendation_rationale: str
    factors_that_change_recommendation: list[str]
    data_limitations: list[str]


class AnalystRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    generated_at: str
    temperature: float
    assessment: AnalystAssessment


class OllamaConnectionError(RuntimeError):
    """Raised when the local Ollama service cannot be reached."""


class InvestmentAnalyst:
    """Produces an evidence-constrained assessment from a CompanySnapshot."""

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def analyse(self, snapshot: CompanySnapshot) -> AnalystRun:
        prompt = self._build_prompt(snapshot)

        payload = {
            "model": self.model_name,
            "stream": False,
            "think": False,
            "format": AnalystAssessment.model_json_schema(),
            "options": {
                "temperature": self.temperature,
                "seed": 42,
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a cautious equity-research analyst. "
                        "Return only JSON that satisfies the supplied schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaConnectionError(
                "Could not reach Ollama. Confirm it is running at "
                "http://localhost:11434."
            ) from exc

        content = response.json().get("message", {}).get("content")
        if not content:
            raise ValueError("Ollama returned no analyst content.")

        try:
            assessment = AnalystAssessment.model_validate_json(content)
            for section_name in EVIDENCE_SECTIONS:
                for item in getattr(assessment, section_name):
                    item.evidence_paths = [
                        canonicalise_evidence_path(path)
                        for path in item.evidence_paths
                    ]
        except Exception as exc:
            raise ValueError(
                "Ollama returned content that did not match the analyst schema."
            ) from exc

        return AnalystRun(
            model_name=self.model_name,
            generated_at=datetime.now(timezone.utc).isoformat(),
            temperature=self.temperature,
            assessment=assessment,
        )

    @staticmethod
    def _build_prompt(snapshot: CompanySnapshot) -> str:
        snapshot_json = snapshot.model_dump_json(indent=2)

        return f"""
Analyse the company snapshot below as an investment-research demonstration.

Rules:
1. Treat the supplied snapshot as the complete factual record. Do not use
   outside knowledge, prices, news, forecasts, peer comparisons, or assumptions.
2. Never invent, calculate, or round a financial number.
3. Every factual claim must cite one or more exact snapshot paths in
   `evidence_paths`.
4. Clearly distinguish interpretation from facts.
4a. Every item in positives, negatives, catalysts, and key_risks must include
    at least one evidence path. Do not introduce catalysts based on drug
    approvals, market expansion, competition, forecasts, or other external facts.
4b. Do not attach a currency symbol or currency name to a financial-statement
    value. The snapshot does not verify statement currency.
5. `financial_statement_currency` is unverified. Do not compare the share price
   with revenue, earnings, cash flow, or debt; do not infer P/E, price-to-sales,
   or any valuation multiple. State this limitation in `valuation_assessment`.
6. Missing or unavailable values are evidence limitations, not zeroes.
7. Use BUY, HOLD, or SELL only. Prefer HOLD when the evidence is insufficient.
8. Confidence refers only to the recommendation under this incomplete dataset,
   not a probability of investment performance.

Company snapshot:
{snapshot_json}
""".strip()