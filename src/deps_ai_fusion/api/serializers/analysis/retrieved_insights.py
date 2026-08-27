from deps_gen_ai.common import ElementCode
from pydantic import Field

from deps_ai_fusion.api.serializers.base import ConfiguredBaseModel
from deps_ai_fusion.api.serializers.confidence_result import (
    SerializedConfidenceTrace,
    SerializedEvidence,
    SerializedValueItem,
)
from deps_ai_fusion.infrastructure.services.processed_insight import ProcessedInsight

__all__ = ["SerializedRetrievedInsights"]


class InsightMetadata(ConfiguredBaseModel):
    confidence_trace: SerializedConfidenceTrace | None = Field(None, alias="confidenceTrace")
    values: list[SerializedValueItem] | None = None
    key_confidence: float | None = Field(None, alias="keyConfidence")
    key_evidence: SerializedEvidence | None = Field(None, alias="keyEvidence")
    key_confidence_trace: SerializedConfidenceTrace | None = Field(None, alias="keyConfidenceTrace")
    value_confidence: float | None = Field(None, alias="valueConfidence")
    value_evidence: SerializedEvidence | None = Field(None, alias="valueEvidence")
    value_confidence_trace: SerializedConfidenceTrace | None = Field(None, alias="valueConfidenceTrace")

    @classmethod
    def from_processed(cls, processed: ProcessedInsight) -> "InsightMetadata":
        return cls(
            confidence_trace=SerializedConfidenceTrace.from_trace(processed.confidence_trace)
            if processed.confidence_trace
            else None,
            values=[SerializedValueItem.from_processed_item(v) for v in processed.values] if processed.values else None,
            key_confidence=processed.key_confidence,
            key_evidence=SerializedEvidence(text=processed.key_evidence.text, page=processed.key_evidence.page)
            if processed.key_evidence
            else None,
            key_confidence_trace=SerializedConfidenceTrace.from_trace(processed.key_confidence_trace)
            if processed.key_confidence_trace
            else None,
            value_confidence=processed.value_confidence,
            value_evidence=SerializedEvidence(text=processed.value_evidence.text, page=processed.value_evidence.page)
            if processed.value_evidence
            else None,
            value_confidence_trace=SerializedConfidenceTrace.from_trace(processed.value_confidence_trace)
            if processed.value_confidence_trace
            else None,
        )


class Insight(ConfiguredBaseModel):
    content: str
    error_occurred: bool = Field(..., alias="errorOccurred")
    confidence: float | None
    evidence: SerializedEvidence | None = None
    metadata: InsightMetadata | None = None

    @classmethod
    def from_processed(cls, processed: ProcessedInsight) -> "Insight":
        return cls(
            content=processed.content,
            error_occurred=processed.error_occurred,
            confidence=processed.confidence,
            evidence=SerializedEvidence(text=processed.evidence.text, page=processed.evidence.page)
            if processed.evidence
            else None,
            metadata=InsightMetadata.from_processed(processed),
        )


class SerializedRetrievedInsights(ConfiguredBaseModel):
    elements: dict[str, Insight]

    @classmethod
    def from_processed_insights(cls, insights: dict[ElementCode, ProcessedInsight]) -> "SerializedRetrievedInsights":
        return cls(elements={code: Insight.from_processed(processed) for code, processed in insights.items()})
