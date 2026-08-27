from typing import Any

from deps_gen_ai.common import ElementCode, PageSpan, Query
from deps_gen_ai.providers import ProviderCode
from deps_gen_ai.providers_aggregate import ProvidersAggregate

from deps_ai_fusion.domain.model import IModelCapabilitiesService
from deps_ai_fusion.infrastructure.proxies import FileStorageProxy
from deps_ai_fusion.infrastructure.services.insight_processor import InsightProcessor
from deps_ai_fusion.infrastructure.services.processed_insight import ProcessedInsight

from .types import ModelName, ProviderModels, ProviderName, RawPageSpan

__all__ = ["AnalysisService"]


class AnalysisService:
    _DEFAULT_PROVIDER = ProviderCode.EPAM_DIAL.value

    def __init__(
        self,
        providers: ProvidersAggregate,
        storage: FileStorageProxy,
        capabilities_service: IModelCapabilitiesService,
        insight_processor: InsightProcessor,
    ) -> None:
        self._providers = providers
        self._storage = storage
        self._capabilities_service = capabilities_service
        self._insight_processor = insight_processor

    def retrieve_insights(
        self,
        llm_reference: str,
        document_id: str,
        requested_insights: dict[ElementCode, Query],
        raw_llm_params: dict[str, Any],
        custom_instructions: str | None,
        retrival_group_size: int | None,
        page_span: RawPageSpan | None = None,
        files: list[str] | None = None,
    ) -> dict[ElementCode, ProcessedInsight]:
        provider, model = self._split_llm_reference(llm_reference)
        filtered = self._capabilities_service.filter_to_supported_parameters(provider, model, raw_llm_params)

        retrieved = self._providers.retrieve_insights(
            provider=provider,
            model=model,
            entity_id=document_id,
            elements=requested_insights,
            custom_instructions=custom_instructions,
            raw_llm_params=filtered,
            retrival_group_size=retrival_group_size,
            page_span=PageSpan(start=page_span["start"], end=page_span["end"]) if page_span is not None else None,
            files=files,
        )
        return {
            code: self._insight_processor.process(response, str(code)) for code, response in retrieved.insights.items()
        }

    def retrieve_file_insights(
        self,
        llm_reference: str,
        filepath: str,
        requested_insights: dict[ElementCode, Query],
        raw_llm_params: dict[str, Any],
        custom_instructions: str | None,
        retrival_group_size: int | None,
        page_span: RawPageSpan | None = None,
        files: list[str] | None = None,
    ) -> dict[ElementCode, ProcessedInsight]:
        provider, model = self._split_llm_reference(llm_reference)
        raw_file: bytes = self._storage.download_content(filepath)
        filtered = self._capabilities_service.filter_to_supported_parameters(provider, model, raw_llm_params)

        retrieved = self._providers.retrieve_file_insights(
            provider=provider,
            model=model,
            filepath=filepath,
            file_blob=raw_file,
            elements=requested_insights,
            custom_instructions=custom_instructions,
            raw_llm_params=filtered,
            retrival_group_size=retrival_group_size,
            page_span=PageSpan(start=page_span["start"], end=page_span["end"]) if page_span is not None else None,
            files=files,
        )
        return {
            code: self._insight_processor.process(response, str(code)) for code, response in retrieved.insights.items()
        }

    def get_available_models(self) -> list[ProviderModels]:
        providers = self._providers.supported_providers()
        return [
            ProviderModels(
                provider=provider,
                models=self._providers.models_of(provider=provider.code, include_legacy=True),
            )
            for provider in providers
        ]

    def _split_llm_reference(self, llm_reference: str) -> tuple[ProviderName, ModelName]:
        if "@" not in llm_reference:
            return self._DEFAULT_PROVIDER, llm_reference

        provider, model = llm_reference.split("@", maxsplit=1)
        return provider, model
