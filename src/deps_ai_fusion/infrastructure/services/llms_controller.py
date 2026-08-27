import logging

from deps_extracted_data.model import ExtractedData, ExtractedDataFactory
from deps_extracted_data.serializers.v2 import SerializedExtractedData
from deps_gen_ai.common import PageSpan as GenAIPageSpan
from deps_gen_ai.common import Query as GenAIQuery
from deps_gen_ai.common import RetrievedInsights
from deps_gen_ai.providers import ProviderCode
from deps_gen_ai.providers_aggregate import ProvidersAggregate

from deps_ai_fusion.application.i_control_llms import IControlLLMs
from deps_ai_fusion.application.types import ModelName, ProviderName
from deps_ai_fusion.domain.model import Code as QueryCode
from deps_ai_fusion.domain.model import (
    ContextAttachments,
    IModelCapabilitiesService,
    LLMExtractor,
    PageSpan,
    Query,
)

from ..proxies import ExtractionProxy, UnifierProxy
from .coordinates_processor import CoordinatesProcessor
from .llm_extraction import GenAIQueryFactory, InsightsRecorder

__all__ = ["LLMsController"]


class LLMsController(IControlLLMs):
    _DEFAULT_PROVIDER = ProviderCode.EPAM_DIAL.value

    def __init__(
        self,
        providers: ProvidersAggregate,
        extraction: ExtractionProxy,
        unifier: UnifierProxy,
        coordinates_processor: CoordinatesProcessor,
        capabilities_service: IModelCapabilitiesService,
    ) -> None:
        self._providers = providers
        self._extraction = extraction
        self._unifier = unifier
        self._coordinates_processor = coordinates_processor
        self._capabilities_service = capabilities_service
        self._genai_query_factory = GenAIQueryFactory()
        self._insights_recorder = InsightsRecorder()

        self._logger = logging.getLogger(self.__class__.__name__)

    def execute_extractor(
        self,
        llm_extractor: LLMExtractor,
        document_id: str,
        override_llm: str | None = None,
    ) -> None:
        provider, model = (
            self._split_llm_type(override_llm)
            if override_llm
            else (llm_extractor.llm_reference.provider, llm_extractor.llm_reference.model)
        )

        self._logger.info(
            "LLMsController starts execute extractor <%s> for document <%s> with provider <%s>, model <%s>.",
            llm_extractor.id(),
            document_id,
            provider,
            model,
        )

        filtered = self._capabilities_service.filter_to_supported_parameters(
            provider, model, llm_extractor.extraction_params.llm_params
        )

        retrieved_insights = self._providers.retrieve_insights(
            provider=provider,
            model=model,
            entity_id=document_id,
            elements=self._genai_queries_from_domain(llm_extractor.query_values()),
            custom_instructions=llm_extractor.extraction_params.effective_instruction,
            retrival_group_size=llm_extractor.extraction_params.grouping_factor,
            page_span=GenAIPageSpan(
                start=llm_extractor.extraction_params.page_span.start,
                end=llm_extractor.extraction_params.page_span.end,
            )
            if llm_extractor.extraction_params.page_span is not None
            else None,
            files=self._get_files_paths(
                document_id=document_id,
                context_attachments=llm_extractor.extraction_params.context_attachments,
                page_span=llm_extractor.extraction_params.page_span,
            )
            if llm_extractor.extraction_params.context_attachments is not None
            else None,
            raw_llm_params=filtered,
        )

        edata = self._transform_into_extracted_data(
            document_id=document_id,
            queries=llm_extractor.queries,
            retrieved_insights=retrieved_insights,
        )

        if llm_extractor.extraction_params.coordinates_enabled:
            edata = self._enrich_with_coordinates(
                llm_extractor=llm_extractor,
                document_id=document_id,
                extracted_data=edata,
            )
        else:
            self._logger.info(
                "Skipping coordinate enrichment for extractor <%s> document <%s>.",
                llm_extractor.id(),
                document_id,
            )

        self._extraction.save_extracted_data(edata)

    def llm_exists(self, provider: str, model: str) -> bool:
        provider_models = self._providers.models_of(provider)

        for provider_model in provider_models:
            if provider_model.code == model:
                return True

        return False

    def _enrich_with_coordinates(
        self,
        llm_extractor: LLMExtractor,
        document_id: str,
        extracted_data: ExtractedData,
    ) -> ExtractedData:
        self._logger.info(
            "Attempting coordinate enrichment for extractor <%s> document <%s>.",
            llm_extractor.id(),
            document_id,
        )
        try:
            edata_with_llm_coords = self._coordinates_processor.add_llm_coordinates(
                document_id,
                SerializedExtractedData.from_model(extracted_data),
            )
        except Exception as err:
            self._logger.warning(
                "Coordinate enrichment failed for extractor <%s> document <%s>: "
                "exception_type=%s stage=coordinate_enrichment. Saving original extracted data.",
                llm_extractor.id(),
                document_id,
                type(err).__name__,
                exc_info=True,
            )
            return extracted_data

        if edata_with_llm_coords is None:
            self._logger.info(
                "Coordinate enrichment returned no result for extractor <%s> document <%s>. "
                "Saving original extracted data.",
                llm_extractor.id(),
                document_id,
            )
            return extracted_data

        self._logger.info(
            "Coordinate enrichment succeeded for extractor <%s> document <%s>.",
            llm_extractor.id(),
            document_id,
        )
        return edata_with_llm_coords

    def _split_llm_type(self, llm_type: str) -> tuple[ProviderName, ModelName]:
        if "@" not in llm_type:
            return self._DEFAULT_PROVIDER, llm_type

        provider, model = llm_type.split("@", maxsplit=1)
        return provider, model

    def _transform_into_extracted_data(
        self,
        document_id: str,
        queries: dict[QueryCode, Query],
        retrieved_insights: RetrievedInsights,
    ) -> ExtractedData:
        edata = ExtractedDataFactory.make_extracted_data(int(document_id))

        for query_code, insight in retrieved_insights.insights.items():
            query = queries[query_code]
            self._insights_recorder.record_insights(edata, query, insight)

        return edata

    def _genai_queries_from_domain(self, queries: list[Query]) -> dict[str, GenAIQuery]:
        return {query.code: self._genai_query_factory.genai_query_from_domain(query) for query in queries}

    def _get_files_paths(  # type: ignore
        self, document_id: str, context_attachments: ContextAttachments, page_span: PageSpan | None
    ) -> list[str] | None:
        if context_attachments == ContextAttachments.ORIGINAL_DOCUMENT:
            self._logger.info("Getting original document as context attachment is not implementing yet.")
            return None
        if context_attachments == ContextAttachments.DOCUMENT_IMAGES:
            original_images = self._unifier.get_original_images(document_id)
            if page_span is None:
                return [image.path for image in original_images]
            else:
                return [
                    image.path for image in original_images if image.page in range(page_span.start, page_span.end + 1)
                ]
