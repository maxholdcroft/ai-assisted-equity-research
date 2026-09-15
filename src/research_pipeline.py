"""End-to-end, review-gated equity-research pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.agents.analyst import AnalystRun, InvestmentAnalyst
from src.agents.critic import CriticRun, RiskCritic
from src.agents.validation import AnalystValidation, validate_analyst_assessment
from src.analysis.snapshot import CompanySnapshot, build_company_snapshot
from src.data.yahoo_finance import YahooFinanceProvider

from dataclasses import dataclass

@dataclass
class ResearchRun:
    """One analysis run, retaining raw data for charts outside the LLM evidence."""

    raw_data: RawCompanyData
    report: "ResearchReport"

class ResearchReport(BaseModel):
    """Auditable output: facts, AI analysis, deterministic checks, and review."""

    model_config = ConfigDict(extra="forbid")

    snapshot: CompanySnapshot
    analyst_run: AnalystRun
    deterministic_validation: AnalystValidation
    critic_run: CriticRun
    publication_status: Literal["APPROVED", "QUALIFIED", "REQUIRES_REVIEW"]
    publication_reason: str


def run_research(ticker: str) -> ResearchRun:
    """Run the full MVP workflow for one listed-company ticker."""

    raw_data = YahooFinanceProvider().fetch_company_data(ticker)
    snapshot = build_company_snapshot(raw_data)

    analyst_run = InvestmentAnalyst().analyse(snapshot)
    deterministic_validation = validate_analyst_assessment(
        snapshot,
        analyst_run.assessment,
    )

    critic_run = RiskCritic().review(snapshot, analyst_run)

    status, reason = _determine_publication_status(
        deterministic_validation,
        critic_run,
    )

    report = ResearchReport(
        snapshot=snapshot,
        analyst_run=analyst_run,
        deterministic_validation=deterministic_validation,
        critic_run=critic_run,
        publication_status=status,
        publication_reason=reason,
    )

    return ResearchRun(
        raw_data=raw_data,
        report=report,
    )

def _determine_publication_status(
    validation: AnalystValidation,
    critic_run: CriticRun,
) -> tuple[Literal["APPROVED", "QUALIFIED", "REQUIRES_REVIEW"], str]:
    """Gate publication without overwriting either AI agent's output."""

    if not validation.passed:
        return (
            "REQUIRES_REVIEW",
            "Deterministic evidence validation identified invalid or missing support.",
        )

    critic = critic_run.assessment
    critic_findings = (
        critic.unsupported_claims
        + critic.overlooked_risks_or_limitations
        + critic.disagreements
    )
    has_high_severity_finding = any(
        finding.severity == "HIGH" for finding in critic_findings
    )

    if (
        critic.analyst_recommendation_supported == "NO"
        or has_high_severity_finding
    ):
        return (
            "REQUIRES_REVIEW",
            "The independent critic identified high-severity concerns; "
            "the analyst recommendation must not be treated as approved research.",
        )

    if critic.analyst_recommendation_supported == "PARTIALLY":
        return (
            "QUALIFIED",
            "The critic only partially supports the analyst recommendation. "
            "Read the risk review before relying on it.",
        )

    return (
        "APPROVED",
        "The analyst output passed deterministic checks and received critic support.",
    )