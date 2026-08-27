from pydantic import Field

from deps_ai_fusion.api.serializers.base import ConfiguredBaseModel
from deps_ai_fusion.infrastructure.services.llm_extraction.confidence import (
    ConfidenceTrace,
)
from deps_ai_fusion.infrastructure.services.processed_insight import ProcessedValueItem

__all__ = [
    "SerializedEvidence",
    "SerializedConfidenceTrace",
    "SerializedValueItem",
]


class SerializedEvidence(ConfiguredBaseModel):
    text: str
    page: int


class SerializedConfidenceTrace(ConfiguredBaseModel):
    level: str
    self_confidence: str | None = Field(None, alias="selfConfidence")
    reason_codes: list[str] = Field(default_factory=list, alias="reasonCodes")
    aggregation_rule_id: str = Field(..., alias="aggregationRuleId")

    @classmethod
    def from_trace(cls, trace: ConfidenceTrace) -> "SerializedConfidenceTrace":
        return cls(
            level=trace.c_final.value,
            self_confidence=trace.self_confidence.value if trace.self_confidence else None,
            reason_codes=[rc.value for rc in trace.reason_codes],
            aggregation_rule_id=trace.aggregation_rule_id,
        )


class SerializedValueItem(ConfiguredBaseModel):
    value: str
    confidence: float | None
    evidence: SerializedEvidence | None = None
    confidence_trace: SerializedConfidenceTrace | None = Field(None, alias="confidenceTrace")
    key: str | None = None
    key_confidence: float | None = Field(None, alias="keyConfidence")
    key_evidence: SerializedEvidence | None = Field(None, alias="keyEvidence")
    key_confidence_trace: SerializedConfidenceTrace | None = Field(None, alias="keyConfidenceTrace")
    value_confidence: float | None = Field(None, alias="valueConfidence")
    value_evidence: SerializedEvidence | None = Field(None, alias="valueEvidence")
    value_confidence_trace: SerializedConfidenceTrace | None = Field(None, alias="valueConfidenceTrace")

    @classmethod
    def from_processed_item(cls, item: ProcessedValueItem) -> "SerializedValueItem":
        return cls(
            value=item.value,
            confidence=item.confidence,
            evidence=SerializedEvidence(text=item.evidence.text, page=item.evidence.page) if item.evidence else None,
            confidence_trace=SerializedConfidenceTrace.from_trace(item.confidence_trace)
            if item.confidence_trace
            else None,
            key=item.key,
            key_confidence=item.key_confidence,
            key_evidence=SerializedEvidence(text=item.key_evidence.text, page=item.key_evidence.page)
            if item.key_evidence
            else None,
            key_confidence_trace=SerializedConfidenceTrace.from_trace(item.key_confidence_trace)
            if item.key_confidence_trace
            else None,
            value_confidence=item.value_confidence,
            value_evidence=SerializedEvidence(text=item.value_evidence.text, page=item.value_evidence.page)
            if item.value_evidence
            else None,
            value_confidence_trace=SerializedConfidenceTrace.from_trace(item.value_confidence_trace)
            if item.value_confidence_trace
            else None,
        )
