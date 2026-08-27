from datetime import datetime

from pydantic import Field

from deps_ai_fusion.api.serializers.base import ConfiguredBaseModel
from deps_ai_fusion.api.serializers.confidence_result import (
    SerializedConfidenceTrace,
    SerializedEvidence,
)
from deps_ai_fusion.api.serializers.page_span import SerializedPageSpan
from deps_ai_fusion.application.types import CompletionWithInsight
from deps_ai_fusion.domain.model.conversation import Completion
from deps_ai_fusion.infrastructure.services.processed_insight import ProcessedInsight

__all__ = ["SerializedCompletion", "CreateCompletionRequest"]


class CreateCompletionRequest(ConfiguredBaseModel):
    question: str
    model: str
    provider: str
    page_span: SerializedPageSpan | None = Field(default=None, alias="pageSpan")
    files: list[str] | None = None


class SerializedCompletion(ConfiguredBaseModel):
    code: str
    question: str
    response: str
    model: str
    provider: str
    confidence: float | None
    evidence: SerializedEvidence | None = None
    confidence_trace: SerializedConfidenceTrace | None = Field(None, alias="confidenceTrace")
    created_at: datetime = Field(..., alias="createdAt")

    @classmethod
    def from_completion_with_insight(cls, result: CompletionWithInsight) -> "SerializedCompletion":
        return cls._build(completion=result.completion, processed=result.processed)

    @classmethod
    def from_stored(cls, completion: Completion, processed: ProcessedInsight) -> "SerializedCompletion":
        return cls._build(completion=completion, processed=processed)

    @classmethod
    def _build(cls, completion: Completion, processed: ProcessedInsight) -> "SerializedCompletion":
        return cls(
            code=completion.code,
            question=completion.question,
            response=processed.content,
            model=completion.llm_reference.model,
            provider=completion.llm_reference.provider,
            confidence=processed.confidence,
            evidence=SerializedEvidence(text=processed.evidence.text, page=processed.evidence.page)
            if processed.evidence
            else None,
            confidence_trace=SerializedConfidenceTrace.from_trace(processed.confidence_trace)
            if processed.confidence_trace
            else None,
            created_at=completion.created_at,
        )
