from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from deps_extracted_data.model import (
    MAX_ALIAS_LENGTH,
    ExtractedData,
    ExtractedFieldFactory,
    FieldDataFactory,
)
from deps_gen_ai.common import LLMResponse
from pydantic import BaseModel

from deps_ai_fusion.domain.model import Query
from deps_ai_fusion.infrastructure.services.llm_extraction.confidence import (
    ConfidenceEvaluator,
    ConfidenceLevel,
    Evidence,
    SchemaHint,
)

__all__ = ["AbstractInsightsRecorder"]

TResponseType = TypeVar("TResponseType", bound=BaseModel)


class AbstractInsightsRecorder(ABC, Generic[TResponseType]):
    extracted_field_factory = ExtractedFieldFactory()
    field_data_factory = FieldDataFactory()
    _confidence_evaluator = ConfidenceEvaluator()

    @abstractmethod
    def record_insights(
        self,
        edata: ExtractedData,
        for_query: Query,
        insight: LLMResponse[TResponseType],
    ) -> None:
        ...

    def _compute_confidence(
        self,
        field_code: str,
        value_for_check: str | bool | None,
        evidence: Evidence | None,
        self_confidence: ConfidenceLevel | None,
        is_list: bool = False,
        schema_hint: SchemaHint | None = None,
        label: str | None = None,
    ) -> float:
        return self._confidence_evaluator.evaluate(
            field_code=field_code,
            value=value_for_check,
            evidence=evidence,
            self_confidence=self_confidence,
            is_list=is_list,
            schema_hint=schema_hint,
            label=label,
        ).as_float()

    @staticmethod
    def _ev_from_raw(raw: object) -> Evidence | None:
        if not isinstance(raw, dict):
            return None
        try:
            return Evidence(text=str(raw["text"]), page=int(raw["page"]))
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _sc_from_raw(raw: object) -> ConfidenceLevel | None:
        return ConfidenceLevel(raw) if raw in {c.value for c in ConfidenceLevel} else None

    @staticmethod
    def _truncate_alias(alias: str) -> str:
        if len(alias) > MAX_ALIAS_LENGTH:
            return f"{alias[:MAX_ALIAS_LENGTH - 3]}..."
        return alias
