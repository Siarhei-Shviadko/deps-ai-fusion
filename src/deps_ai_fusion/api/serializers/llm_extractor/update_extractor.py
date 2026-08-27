from typing import Any

from pydantic import Field

from deps_ai_fusion.domain.model import ContextAttachments, RawLLMExtractionParams

from ..base import ConfiguredBaseModel
from ..base_llm_params import BaseLLMParams
from ..page_span import SerializedPageSpan

__all__ = ["UpdateExtractorRequest", "UpdateExtractorResponse"]


class UpdateExtractorParams(BaseLLMParams):
    custom_instruction: str = Field(..., alias="customInstruction")
    page_span: SerializedPageSpan | None = Field(None, alias="pageSpan")
    grouping_factor: int = Field(..., ge=1, alias="groupingFactor")
    context_attachments: ContextAttachments | None = Field(None, alias="contextAttachments")
    coordinates_enabled: bool = Field(False, alias="coordinatesEnabled")
    temperature: float = Field(
        default=0.5,
        ge=0,
        le=2,
        examples=[0.7],
        description="""Controls the randomness of text generation.
        Lower temperatures make the model more deterministic and repetitive, while higher temperatures make the model more creative and random.
        """,
    )

    def to_dict(self) -> RawLLMExtractionParams:
        extra_llm_params = {
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "seed": self.seed,
            "logprobs": self.logprobs,
        }
        extra_llm_params.update(self.extra_model_params or {})
        return {
            "custom_instruction": self.custom_instruction,
            "grouping_factor": self.grouping_factor,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "extra_llm_params": extra_llm_params,
            "page_span": self.page_span.to_dict() if self.page_span else None,
            "context_attachments": self.context_attachments if self.context_attachments else None,
            "coordinates_enabled": self.coordinates_enabled,
        }


class UpdateExtractorRequest(ConfiguredBaseModel):
    name: str
    extraction_params: UpdateExtractorParams = Field(..., alias="extractionParams")

    def to_dict(self) -> dict[str, Any]:
        extraction_params = self.extraction_params.to_dict()
        return {
            "name": self.name,
            **extraction_params,
        }


class UpdateExtractorResponse(ConfiguredBaseModel):
    extractor_id: str = Field(..., alias="extractorId")
    document_type_id: str = Field(..., alias="documentTypeId")
