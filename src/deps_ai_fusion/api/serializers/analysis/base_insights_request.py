from typing import Any

from deps_gen_ai.common import Query
from pydantic import Field, field_validator

from deps_ai_fusion.application.structured_outputs import ReasoningResponse

from ..base import ConfiguredBaseModel
from ..base_llm_params import BaseLLMParams
from ..page_span import SerializedPageSpan
from .genai_query import SerializedGenAIQuery

__all__ = ["BaseInsightsRequest"]


class RequestParams(BaseLLMParams):
    grouping_factor: int | None = Field(
        None,
        alias="groupingFactor",
        description="""Defines how many elements are grouped per LLM request.
        The higher this value, the fewer requests will be made to the LLM decreasing costs (as document context is loaded into each request).
        On the other hand, a large number of elements grouped in a single request potentially decreases the quality of the insights retrieved.
        """,
    )
    page_span: SerializedPageSpan | None = Field(
        None,
        alias="pageSpan",
        description="Inclusive range of pages to process (e.g. PageSpan(start=1, end=5)). If omitted, all pages will be processed.",
    )

    @field_validator("grouping_factor")
    @classmethod
    def validate_grouping_factor(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("Grouping Factor must be greater than 0")

        return value

    def to_raw_llm_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.stop is not None:
            params["stop"] = self.stop
        if self.seed is not None:
            params["seed"] = self.seed
        if self.logprobs is not None:
            params["logprobs"] = self.logprobs
        if self.extra_model_params is not None:
            params["model_kwargs"] = self.extra_model_params
        return params


class BaseInsightsRequest(ConfiguredBaseModel):
    llm_reference: str = Field(
        ...,
        alias="model",
        description="LLM Reference contains information about Provider and Model in the format of 'provider@model'",
    )
    requested_insights: dict[str, str | SerializedGenAIQuery] = Field(
        ...,
        alias="requestedInsights",
        description="Mapping between an ElementCode to retrieve insights for, and a Prompt to use for that ElementCode",
    )
    custom_instructions: str | None = Field(
        None,
        alias="customInstructions",
        description="""Custom instructions are appended to the system prompt to guide the model’s behavior.
        Useful for adapting the model to specific use cases, such as summarization, extraction, classification, etc.
        They also help to tailor the responses by specifying constraints, response style, or additional context.
        """,
    )
    params: RequestParams = Field(
        ...,
        description="Additional parameters to be used for the insights retrieval.",
    )
    files: list[str] | None = None

    @field_validator("llm_reference")
    @classmethod
    def validate_llm_reference(cls, value: str) -> str:
        if value.count("@") > 1:
            raise ValueError("LLM Reference can't contain extra '@' symbol!")

        return value

    def requested_insights_as_queries(self) -> dict[str, Query]:
        type_to_query_factory = {
            str: lambda request: Query.from_raw(prompts=[request], response_model=ReasoningResponse),
            SerializedGenAIQuery: lambda request: Query.from_raw(
                prompts=[p.content for p in request.workflow.prompts],
                response_model=request.workflow.response_model,
            ),
        }

        return {
            element_code: type_to_query_factory[type(request)](request)
            for element_code, request in self.requested_insights.items()
        }
