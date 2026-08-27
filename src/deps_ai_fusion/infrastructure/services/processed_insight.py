from dataclasses import dataclass

from deps_ai_fusion.infrastructure.services.llm_extraction.confidence import (
    ConfidenceTrace,
    Evidence,
)

__all__ = ["ProcessedInsight", "ProcessedValueItem"]


@dataclass
class ProcessedValueItem:
    value: str
    confidence: float | None
    evidence: Evidence | None
    confidence_trace: ConfidenceTrace | None
    key: str | None = None
    key_confidence: float | None = None
    key_evidence: Evidence | None = None
    key_confidence_trace: ConfidenceTrace | None = None
    value_confidence: float | None = None
    value_evidence: Evidence | None = None
    value_confidence_trace: ConfidenceTrace | None = None


@dataclass
class ProcessedInsight:
    content: str
    error_occurred: bool
    confidence: float | None
    evidence: Evidence | None
    confidence_trace: ConfidenceTrace | None
    values: list[ProcessedValueItem] | None = None
    key_confidence: float | None = None
    key_evidence: Evidence | None = None
    key_confidence_trace: ConfidenceTrace | None = None
    value_confidence: float | None = None
    value_evidence: Evidence | None = None
    value_confidence_trace: ConfidenceTrace | None = None
