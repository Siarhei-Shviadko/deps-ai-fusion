from pydantic import Field

from deps_ai_fusion.domain.model import (
    DEFAULT_CUSTOM_INSTRUCTION,
    DEFAULT_GROUPING_FACTOR,
    ContextAttachments,
    ExtractionParams,
    RawLLMExtractionParams,
)

from ..base_llm_params import BaseLLMParams
from ..page_span import SerializedPageSpan

__all__ = ["SerializedLLMExtractionParams"]


class SerializedLLMExtractionParams(BaseLLMParams):
    custom_instruction: str = Field(DEFAULT_CUSTOM_INSTRUCTION, alias="customInstruction")
    grouping_factor: int = Field(DEFAULT_GROUPING_FACTOR, alias="groupingFactor", ge=1)
    page_span: SerializedPageSpan | None = Field(None, alias="pageSpan")
    context_attachments: ContextAttachments | None = Field(None, alias="contextAttachments")

    @classmethod
    def from_model(cls, extraction_params: ExtractionParams) -> "SerializedLLMExtractionParams":
        llm_params = extraction_params.llm_params
        return cls(
            custom_instruction=extraction_params.custom_instruction,
            grouping_factor=extraction_params.grouping_factor,
            temperature=llm_params.pop("temperature"),
            top_p=llm_params.pop("top_p", None),
            max_tokens=llm_params.pop("max_tokens", None),
            stop=llm_params.pop("stop", None),
            seed=llm_params.pop("seed", None),
            logprobs=llm_params.pop("logprobs", False),
            extra_model_params=llm_params,
            page_span=SerializedPageSpan(**extraction_params.page_span.to_dict())
            if extraction_params.page_span
            else None,
            context_attachments=extraction_params.context_attachments,
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
        }
