from unittest.mock import Mock

import pytest
from deps_extracted_data.model import ExtractedData, ExtractedDataFactory

from deps_ai_fusion.domain.model import LLMExtractor
from deps_ai_fusion.infrastructure.proxies.exceptions import ParsingProxyError
from deps_ai_fusion.infrastructure.services.llms_controller import LLMsController
from tests.factories import ExtractionParamsFactory, LLMExtractorFactory, QueryFactory
from tests.fakes import FakeProvidersAggregate


@pytest.fixture
def mock_coordinates_processor():
    return Mock()


@pytest.fixture
def mock_capabilities_service():
    service = Mock()
    service.filter_to_supported_parameters.side_effect = lambda provider, model, params: params
    return service


@pytest.fixture
def llms_controller(
    fake_providers_aggregate: FakeProvidersAggregate,
    mock_extraction,
    mock_unifier,
    mock_coordinates_processor,
    mock_capabilities_service,
    unifier_proxy_return_value,
) -> LLMsController:
    fake_providers_aggregate.set_response("value")
    mock_unifier.get_original_images.return_value = unifier_proxy_return_value
    return LLMsController(
        providers=fake_providers_aggregate,
        extraction=mock_extraction,
        unifier=mock_unifier,
        coordinates_processor=mock_coordinates_processor,
        capabilities_service=mock_capabilities_service,
    )


@pytest.fixture
def extractor_with_query() -> LLMExtractor:
    query = QueryFactory.create()
    extractor = LLMExtractorFactory.create(
        extraction_params=ExtractionParamsFactory(
            coordinates_enabled=False,
            context_attachments=None,
            page_span=None,
        ),
        queries={query.code: query},
    )
    extractor.events.clear()
    return extractor


@pytest.fixture
def extractor_with_coordinates_enabled() -> LLMExtractor:
    query = QueryFactory.create()
    extractor = LLMExtractorFactory.create(
        extraction_params=ExtractionParamsFactory(
            coordinates_enabled=True,
            context_attachments=None,
            page_span=None,
        ),
        queries={query.code: query},
    )
    extractor.events.clear()
    return extractor


@pytest.fixture
def enriched_extracted_data() -> ExtractedData:
    return ExtractedDataFactory.make_extracted_data(222)


def test_execute_extractor__coordinates_disabled__coordinate_processor_not_called(
    llms_controller: LLMsController,
    mock_coordinates_processor: Mock,
    mock_extraction,
    extractor_with_query: LLMExtractor,
):
    llms_controller.execute_extractor(extractor_with_query, document_id="111")

    mock_coordinates_processor.add_llm_coordinates.assert_not_called()
    mock_extraction.save_extracted_data.assert_called_once()


def test_execute_extractor__coordinates_enabled__coordinate_processor_called_once(
    llms_controller: LLMsController,
    mock_coordinates_processor: Mock,
    mock_extraction,
    extractor_with_coordinates_enabled: LLMExtractor,
    enriched_extracted_data: ExtractedData,
):
    mock_coordinates_processor.add_llm_coordinates.return_value = enriched_extracted_data

    llms_controller.execute_extractor(extractor_with_coordinates_enabled, document_id="111")

    mock_coordinates_processor.add_llm_coordinates.assert_called_once()
    mock_extraction.save_extracted_data.assert_called_once_with(enriched_extracted_data)


def test_execute_extractor__coordinate_enrichment_succeeds__enriched_data_saved(
    llms_controller: LLMsController,
    mock_coordinates_processor: Mock,
    mock_extraction,
    extractor_with_coordinates_enabled: LLMExtractor,
    enriched_extracted_data: ExtractedData,
):
    mock_coordinates_processor.add_llm_coordinates.return_value = enriched_extracted_data

    llms_controller.execute_extractor(extractor_with_coordinates_enabled, document_id="111")

    saved = mock_extraction.save_extracted_data.call_args.args[0]
    assert saved is enriched_extracted_data


def test_execute_extractor__coordinate_enrichment_returns_none__original_data_saved(
    llms_controller: LLMsController,
    mock_coordinates_processor: Mock,
    mock_extraction,
    extractor_with_coordinates_enabled: LLMExtractor,
):
    mock_coordinates_processor.add_llm_coordinates.return_value = None

    llms_controller.execute_extractor(extractor_with_coordinates_enabled, document_id="111")

    mock_extraction.save_extracted_data.assert_called_once()
    saved = mock_extraction.save_extracted_data.call_args.args[0]
    assert saved is not None


def test_execute_extractor__coordinate_enrichment_fails__original_data_saved(
    llms_controller: LLMsController,
    mock_coordinates_processor: Mock,
    mock_extraction,
    extractor_with_coordinates_enabled: LLMExtractor,
):
    mock_coordinates_processor.add_llm_coordinates.side_effect = ParsingProxyError("layout unavailable")

    llms_controller.execute_extractor(extractor_with_coordinates_enabled, document_id="111")

    mock_extraction.save_extracted_data.assert_called_once()
    saved = mock_extraction.save_extracted_data.call_args.args[0]
    assert saved is not None


def test_execute_extractor__llm_execution_fails__error_propagated(
    llms_controller: LLMsController,
    fake_providers_aggregate: FakeProvidersAggregate,
    mock_coordinates_processor: Mock,
    mock_extraction,
    extractor_with_coordinates_enabled: LLMExtractor,
    mocker,
):
    mocker.patch.object(fake_providers_aggregate, "retrieve_insights", side_effect=RuntimeError("llm failed"))

    with pytest.raises(RuntimeError, match="llm failed"):
        llms_controller.execute_extractor(extractor_with_coordinates_enabled, document_id="111")

    mock_coordinates_processor.add_llm_coordinates.assert_not_called()
    mock_extraction.save_extracted_data.assert_not_called()


def test_execute_extractor__extracted_data_save_fails__infrastructure_error_propagated(
    llms_controller: LLMsController,
    mock_coordinates_processor: Mock,
    mock_extraction,
    extractor_with_query: LLMExtractor,
):
    mock_extraction.save_extracted_data.side_effect = RuntimeError("save failed")

    with pytest.raises(RuntimeError, match="save failed"):
        llms_controller.execute_extractor(extractor_with_query, document_id="111")

    mock_coordinates_processor.add_llm_coordinates.assert_not_called()
