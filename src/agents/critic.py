"""Independent local LLM critic for analyst-output validation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import requests
from pydantic import BaseModel, ConfigDict, Field

from src.agents.analyst import AnalystRun
from src.analysis.snapshot import CompanySnapshot


class ReviewFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_or_issue: str
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    explanation: str
    relevant_snapshot_paths: list[str] = Field(default_factory=list)


class CriticAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analyst_recommendation_supported: Literal["YES", "PARTIALLY", "NO"]
    unsupported_claims: list[ReviewFinding]
    overlooked_risks_or_limitations: list[ReviewFinding]
    disagreements: list[ReviewFinding]
    suggested_confidence: Literal["LOW", "MEDIUM", "HIGH"]
    final_review_commentary: str


class CriticRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    generated_at: str
    temperature: float
    assessment: CriticAssessment


class RiskCritic:
    """Challenges the analyst using the original structured evidence."""

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def review(
        self,
        snapshot: CompanySnapshot,
        analyst_run: AnalystRun,
    ) -> CriticRun:
        payload = {
            "model": self.model_name,
            "stream": False,
            "think": False,
            "format": CriticAssessment.model_json_schema(),
            "options": {
                "temperature": self.temperature,
                "seed": 42,
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an independent, sceptical equity-research "
                        "quality reviewer. Return only JSON matching the schema."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_prompt(snapshot, analyst_run),
                },
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
            raise RuntimeError(
                "Could not reach Ollama for the critic review."
            ) from exc

        content = response.json().get("message", {}).get("content")
        if not content:
            raise ValueError("Ollama returned no critic content.")

        try:
            assessment = CriticAssessment.model_validate_json(content)
        except Exception as exc:
            raise ValueError(
                "Ollama returned content that did not match the critic schema."
            ) from exc

        return CriticRun(
            model_name=self.model_name,
            generated_at=datetime.now(timezone.utc).isoformat(),
            temperature=self.temperature,
            assessment=assessment,
        )

    @staticmethod
    def _build_prompt(
        snapshot: CompanySnapshot,
        analyst_run: AnalystRun,
    ) -> str:
        return f"""
Review the analyst assessment against the original company snapshot.

Your role is to challenge the analyst, not agree by default.

Review rules:
1. The snapshot is the complete factual record. Do not use outside knowledge.
2. Flag any claim unsupported by the snapshot, including company-specific
   claims about drugs, markets, competition, forecasts, or events.
3. `financial_statement_currency` being null explicitly means unknown or
   unverified. This supports a limitation on valuation analysis; do not flag
   that limitation itself as unsupported.
4. Flag any attached currency symbol/name for a financial-statement value,
   because the snapshot does not verify that currency.
5. Flag qualitative labels such as "healthy", "substantial", "manageable",
   "moderate", or "significant" when the snapshot supplies no benchmark.
6. Flag catalysts based on market sentiment, external events, forecasts,
   competition, approvals, or other information absent from the snapshot.
7. Check whether recommendation confidence is proportionate to the missing
   valuation, forecast, peer-comparison, and external-information evidence.
8. Check whether the recommendation and confidence follow from the evidence.
9. Identify material data limitations the analyst has overlooked.
10. Be concise and evidence-specific. A low-confidence HOLD can be appropriate.

Original company snapshot:
{snapshot.model_dump_json(indent=2)}

Analyst assessment under review:
{analyst_run.assessment.model_dump_json(indent=2)}
""".strip()