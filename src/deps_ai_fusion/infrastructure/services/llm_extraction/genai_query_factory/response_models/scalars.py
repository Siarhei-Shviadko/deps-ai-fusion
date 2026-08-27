import json
import logging
from typing import ClassVar, Self

from deps_gen_ai.common import LLMResponse
from pydantic import Field

from deps_ai_fusion.infrastructure.services.llm_extraction.confidence import (
    ConfidenceLevel,
    Evidence,
)

from .base import ConfiguredBaseResponseModel
from .types import RawKeyValuePair

__all__ = [
    "StringResponse",
    "BooleanResponse",
    "KeyValuePairResponse",
]


class StringResponse(ConfiguredBaseResponseModel):
    _logger: ClassVar[logging.Logger] = logging.getLogger(__qualname__)

    reasoning: str = Field(
        ...,
        description="Step-by-step explanation on how and why this value was selected from the context.",
    )
    value: str = Field(..., description="The extracted string value.")
    evidence: Evidence | None = Field(
        None,
        description="Exact verbatim quote from the document supporting the value, and the page number.",
    )
    self_confidence: ConfidenceLevel | None = Field(
        None,
        description="Model's own confidence assessment: HIGH, MEDIUM, or LOW.",
    )

    @classmethod
    def from_json_content(cls, content: str) -> "StringResponse | None":
        try:
            data = cls._parse_data_item(content)
            if data is None:
                return None
            value = data.get("value")
            if value is None:
                return None
            return cls(
                reasoning="",
                value=str(value),
                evidence=cls._parse_evidence(data.get("evidence")),
                self_confidence=cls._parse_self_confidence(data.get("self_confidence")),
            )
        except (TypeError, KeyError, ValueError) as exc:
            cls._logger.warning("Failed to parse StringResponse from JSON content", exc_info=exc)
            return None

    @classmethod
    def parse_llm_response(cls, response: LLMResponse[Self]) -> str:
        if not response.success:
            return response.content

        if response.parsed and isinstance(response.parsed, StringResponse):
            return response.parsed.value
        if response.parsed and isinstance(response.parsed, dict):
            return str(response.parsed.get("value", response.content))

        parsed = cls.from_json_content(response.content)
        if parsed is not None:
            return parsed.value
        return response.content

    @staticmethod
    def _parse_data_item(content: str) -> dict | None:
        data = json.loads(content)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
        return data if isinstance(data, dict) else None

    @staticmethod
    def _parse_evidence(evidence_data: dict | None) -> Evidence | None:
        return Evidence(**evidence_data) if evidence_data is not None else None

    @staticmethod
    def _parse_self_confidence(sc: str | None) -> ConfidenceLevel | None:
        return ConfidenceLevel(sc) if sc in {cl.value for cl in ConfidenceLevel} else None


class BooleanResponse(ConfiguredBaseResponseModel):
    reasoning: str = Field(
        ...,
        description="Step-by-step explanation on how and why this value was selected from the context.",
    )
    value: bool = Field(..., description="The extracted boolean value.")
    evidence: Evidence | None = Field(
        None,
        description="Exact verbatim quote from the document supporting the value, and the page number.",
    )
    self_confidence: ConfidenceLevel | None = Field(
        None,
        description="Model's own confidence assessment: HIGH, MEDIUM, or LOW.",
    )

    @classmethod
    def parse_llm_response(cls, response: LLMResponse[Self]) -> bool | None:
        if not response.success:
            return None

        if response.parsed and isinstance(response.parsed, BooleanResponse):
            return response.parsed.value
        if response.parsed and isinstance(response.parsed, dict):
            return response.parsed.get("value", None)

        return None


class KeyValuePairResponse(ConfiguredBaseResponseModel):
    reasoning: str = Field(
        ...,
        description="Step-by-step explanation on how and why this key/value pair was selected from the context.",
    )
    key: str = Field(..., description="The key/name.")
    value: str = Field(..., description="The value associated with the key.")
    key_evidence: Evidence | None = Field(
        None,
        description="Exact verbatim quote from the document supporting the key, and the page number.",
    )
    key_self_confidence: ConfidenceLevel | None = Field(
        None,
        description="Model's own confidence assessment for the key: HIGH, MEDIUM, or LOW.",
    )
    value_evidence: Evidence | None = Field(
        None,
        description="Exact verbatim quote from the document supporting the value, and the page number.",
    )
    value_self_confidence: ConfidenceLevel | None = Field(
        None,
        description="Model's own confidence assessment for the value: HIGH, MEDIUM, or LOW.",
    )

    @classmethod
    def parse_llm_response(cls, response: LLMResponse[Self]) -> RawKeyValuePair:
        if not response.success:
            return {"key": "Error occurred...", "value": response.content}

        if response.parsed and isinstance(response.parsed, KeyValuePairResponse):
            return {"key": response.parsed.key, "value": response.parsed.value}
        if response.parsed and isinstance(response.parsed, dict):
            return {
                "key": str(response.parsed.get("key", "Error occurred...")),
                "value": str(response.parsed.get("value", response.content)),
            }

        return {"key": "Error occurred...", "value": response.content}
