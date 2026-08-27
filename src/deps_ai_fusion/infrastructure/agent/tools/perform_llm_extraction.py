from typing import Annotated

from deps_gen_ai.common import Query
from deps_gen_ai.providers_aggregate import ProvidersAggregate
from langchain_core.tools import ArgsSchema, BaseTool, InjectedToolCallId
from langgraph.prebuilt import InjectedState

from deps_ai_fusion.application import LLMExtractionService
from deps_ai_fusion.domain.model import (
    LLMExtractor,
    LLMExtractorFactory,
    RawLLMExtractionParams,
)
from deps_ai_fusion.infrastructure.services import ModelCapabilitiesService
from deps_ai_fusion.infrastructure.services.llm_extraction import GenAIQueryFactory

from ..settings import settings
from ..state import AgentState
from ..text_sanitizer import sanitize_text
from .schemas import DataShape, ExecuteLLMExtractionRequest

__all__ = ["PerformLLMExtractionTool"]


class PerformLLMExtractionTool(BaseTool):
    name: str = "perform-llm-extraction"
    description: str = (
        "Validate a prompts_chain against the current document without persisting anything. "
        "Use to check response shape and prompt quality before creating a field. "
        "Do not use for direct document questions that do not require structured extraction."
    )
    args_schema: ArgsSchema | None = ExecuteLLMExtractionRequest

    llm_providers: ProvidersAggregate
    extractors_app: LLMExtractionService
    capabilities_service: ModelCapabilitiesService
    query_factory: GenAIQueryFactory = GenAIQueryFactory()

    def _run(
        self,
        reasoning: str,
        prompts_chain: list[str],
        response_model: DataShape,
        tool_call_id: Annotated[str, InjectedToolCallId()],
        state: Annotated[AgentState, InjectedState()],
    ) -> str:
        extractor: LLMExtractor = self._resolve_extractor(state)

        provider = extractor.llm_reference.provider
        model = extractor.llm_reference.model

        filtered = self.capabilities_service.filter_to_supported_parameters(
            provider, model, extractor.extraction_params.llm_params
        )

        retrieved = self.llm_providers.retrieve_insights(
            provider=provider,
            model=model,
            entity_id=state.document_id,
            elements={
                "static-field-code": self._genai_queries_from_request(prompts_chain, response_model),
            },
            raw_llm_params=filtered,
            retrival_group_size=extractor.extraction_params.grouping_factor,
        )

        raw = f"LLM extraction results: ```{retrieved.insights['static-field-code'].content}```"
        return sanitize_text(raw)

    def _resolve_extractor(self, state: AgentState) -> LLMExtractor:
        if (document_type_id := state.document_type_id) is not None:
            existing_extractors = self.extractors_app.get_llm_extractors_for_document_type(
                document_type_id=document_type_id,
                tenant_id=state.tenant_id,
            )

            if existing_extractors:
                return existing_extractors[0]

        return LLMExtractorFactory.create(
            tenant_id="tenant_id",
            document_type_id="document_type_id",
            provider=settings.default_extractor_provider,
            name="It doesn't matter",
            model=settings.default_extractor_model,
            extraction_params=RawLLMExtractionParams(
                custom_instruction=settings.default_extractor_custom_instruction,
                grouping_factor=settings.default_extractor_grouping_factor,
                temperature=settings.default_extractor_temperature,
                top_p=settings.default_extractor_top_p,
                page_span=None,
                context_attachments=None,
                extra_llm_params=settings.default_extractor_extra_llm_params,
                coordinates_enabled=False,
            ),
        )

    def _genai_queries_from_request(self, prompts_chain: list[str], response_model: DataShape) -> Query:
        return Query.from_raw(
            prompts=prompts_chain,
            response_model=self.query_factory.response_model_for_dtype[
                (response_model.data_type, response_model.cardinality, response_model.include_aliases)
            ],
        )
