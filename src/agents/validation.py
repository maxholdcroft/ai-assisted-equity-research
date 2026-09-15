"""Deterministic checks on LLM analyst output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.agents.analyst import (
    AnalystAssessment,
    EVIDENCE_SECTIONS,
    canonicalise_evidence_path,
    )
from src.analysis.snapshot import CompanySnapshot


class ValidationFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: str
    rule: str
    explanation: str
    location: str


class AnalystValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    findings: list[ValidationFinding] = Field(default_factory=list)



EXTERNAL_FACT_PHRASES = (
    "drug approval",
    "drug approvals",
    "market expansion",
    "market saturation",
    "competitive pressure",
    "competitive pressures",
)


def validate_analyst_assessment(
    snapshot: CompanySnapshot,
    assessment: AnalystAssessment,
) -> AnalystValidation:
    """Check evidence references and simple factual-safety rules."""

    valid_paths = _leaf_paths(snapshot.model_dump())
    findings: list[ValidationFinding] = []

    for section_name in EVIDENCE_SECTIONS:
        for index, item in enumerate(getattr(assessment, section_name)):
            location = f"{section_name}[{index}]"

            if not item.evidence_paths:
                findings.append(
                    ValidationFinding(
                        severity="HIGH",
                        rule="evidence_required",
                        explanation="Every research claim needs at least one evidence path.",
                        location=location,
                    )
                )

            for evidence_path in item.evidence_paths:
                evidence_path = canonicalise_evidence_path(evidence_path)

            statement_lower = item.statement.lower()
            for phrase in EXTERNAL_FACT_PHRASES:
                if phrase in statement_lower:
                    findings.append(
                        ValidationFinding(
                            severity="HIGH",
                            rule="no_external_company_facts",
                            explanation=(
                                f"The claim includes '{phrase}', which is not available "
                                "in the supplied snapshot."
                            ),
                            location=location,
                        )
                    )

    for location, text in _string_values(assessment.model_dump()):
        if any(symbol in text for symbol in ("£", "$", "€")):
            findings.append(
                ValidationFinding(
                    severity="HIGH",
                    rule="no_unverified_financial_currency",
                    explanation=(
                        "The snapshot does not verify financial-statement currency, "
                        "so the response cannot attach a currency symbol to a "
                        "financial-statement value."
                    ),
                    location=location,
                )
            )

    return AnalystValidation(
        passed=not findings,
        findings=findings,
    )


def _leaf_paths(value: Any, prefix: str = "") -> set[str]:
    if not isinstance(value, dict):
        return {prefix}

    paths: set[str] = set()
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else key
        paths.update(_leaf_paths(child, child_prefix))

    return paths


def _string_values(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        results: list[tuple[str, str]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            results.extend(_string_values(child, child_prefix))
        return results

    if isinstance(value, list):
        results = []
        for index, child in enumerate(value):
            results.extend(_string_values(child, f"{prefix}[{index}]"))
        return results

    return [(prefix, value)] if isinstance(value, str) else []