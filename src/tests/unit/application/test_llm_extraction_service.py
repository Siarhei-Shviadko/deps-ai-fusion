import uuid

import pytest
from deps_extracted_data.model import ExtractedDataFactory
from deps_gen_ai.providers import ProviderCode

from deps_ai_fusion.application import LLMExtractionService
from deps_ai_fusion.domain.exceptions import (
    LLMExtractorNotFoundError,
    QueryNotFoundError,
)
from deps_ai_fusion.domain.model import LLMExtractor, RawDataShape, RawLLMWorkflow
from deps_ai_fusion.infrastructure.proxies.exceptions import ParsingProxyError
from deps_ai_fusion.messaging.sagas import (
    CreateLLMExtractorSagaData,
    CreateLLMExtractorSteps,
)
from tests.factories import ExtractionParamsFactory, LLMExtractorFactory, QueryFactory
from tests.fakes import (
    FakeExtractionProxy,
    FakeLLMExtractorRepository,
    FakeProvidersAggregate,
)


def test_add_query__success(
    llm_extractor: LLMExtractor,
    test_code: str,
    test_raw_llm_workflow: RawLLMWorkflow,
    test_raw_data_shape: RawDataShape,
    fake_llm_extractor_repository: FakeLLMExtractorRepository,
    llm_extraction_service: LLMExtractionService,
):

    fake_llm_extractor_repository.save(llm_extractor)

    added_query = llm_extraction_service.add_query(
        code=test_code,
        raw_workflow=test_raw_llm_workflow,
        raw_data_shape=test_raw_data_shape,
        extractor_id=llm_extractor.id(),
        document_type_id=llm_extractor.document_type_id(),
        tenant_id=llm_extractor.tenant_id(),
    )

    assert added_query.code == test_code

    saved_llm_extractor = fake_llm_extractor_repository.find_for_document_type(
        id_=llm_extractor.id(),
        document_type_id=llm_extractor.document_type_id(),
        tenant_id=llm_extractor.tenant_id(),
    )

    assert len(saved_llm_extractor.queries) == len(llm_extractor.queries)
    assert saved_llm_extractor.queries.get(test_code) == added_query


def test_add_query__llm_extractor_does_not_exist__error(
    test_code: str,
    llm_extractor_id: str,
    document_type_id: str,
    tenant_id: str,
    test_raw_llm_workflow: RawLLMWorkflow,
    test_raw_data_shape: RawDataShape,
    fake_llm_extractor_repository: FakeLLMExtractorRepository,
    llm_extraction_service: LLMExtractionService,
):
    with pytest.raises(LLMExtractorNotFoundError):
        llm_extraction_service.add_query(
            code=test_code,
            raw_workflow=test_raw_llm_workflow,
            raw_data_shape=test_raw_data_shape,
            extractor_id=llm_extractor_id,
            document_type_id=document_type_id,
            tenant_id=tenant_id,
        )


def test_update_query__success(
    llm_extractor_with_query: LLMExtractor,
    test_raw_llm_workflow: RawLLMWorkflow,
    fake_llm_extractor_repository: FakeLLMExtractorRepository,
    llm_extraction_service: LLMExtractionService,
):
    fake_llm_extractor_repository.save(llm_extractor_with_query)
    updated_prompt_value = uuid.uuid4().hex
    for node in test_raw_llm_workflow["nodes"]:
        node["prompt"] += updated_prompt_value
    query_for_update = llm_extractor_with_query.query_values()[0]

    llm_extraction_service.update_query(
        code=query_for_update.code,
        raw_workflow=test_raw_llm_workflow,
        extractor_id=llm_extractor_with_query.id(),
        document_type_id=llm_extractor_with_query.document_type_id(),
        tenant_id=llm_extractor_with_query.tenant_id(),
    )

    saved_llm_extractor = fake_llm_extractor_repository.find_for_document_type(
        id_=llm_extractor_with_query.id(),
        document_type_id=llm_extractor_with_query.document_type_id(),
        tenant_id=llm_extractor_with_query.tenant_id(),
    )

    for node in saved_llm_extractor.query_values()[0].workflow.nodes:
        assert updated_prompt_value in node.prompt


def test_update_query__llm_extractor_does_not_exist__error(
    test_code: str,
    test_raw_llm_workflow: RawLLMWorkflow,
    llm_extractor_id: str,
    document_type_id: str,
    tenant_id: str,
    fake_llm_extractor_repository: FakeLLMExtractorRepository,
    llm_extraction_service: LLMExtractionService,
):
    with pytest.raises(LLMExtractorNotFoundError):
        llm_extraction_service.update_query(
            code=test_code,
            raw_workflow=test_raw_llm_workflow,
            extractor_id=llm_extractor_id,
            document_type_id=document_type_id,
            tenant_id=tenant_id,
        )


def test_perform_extraction__ok(
    fake_providers_aggregate: FakeProvidersAggregate,
    mock_extraction,
    mock_unifier,
    unifier_proxy_return_value,
    llms_controller_with_fakes,
    fake_llm_extractor_repository,
    llm_extraction_service: LLMExtractionService,
    llm_extractor_with_query: LLMExtractor,
    tenant_id,
):
    fake_llm_extractor_repository.save(llm_extractor_with_query)
    fake_llm_response = "HelloWorld"
    fake_providers_aggregate.set_response(fake_llm_response)
    mock_extraction.save_extracted_data.return_value = None
    mock_unifier.get_original_images.return_value = unifier_proxy_return_value

    llm_extraction_service.perform_extraction(
        extractor_id=llm_extractor_with_query.id(),
        tenant_id=tenant_id,
        document_id="111",
        llm_type=None,
    )

    mock_extraction.save_extracted_data.assert_called()


def test_perform_extraction__extractor_without_prompts__not_called(
    fake_providers_aggregate: FakeProvidersAggregate,
    fake_llm_extractor_repository,
    mock_extraction,
    llms_controller_with_fakes,
    llm_extraction_service: LLMExtractionService,
    llm_extractor_factory,
    tenant_id,
):
    llm_extractor = llm_extractor_factory(queries={})
    fake_llm_extractor_repository.save(llm_extractor)
    fake_llm_response = "HelloWorld"
    fake_providers_aggregate.set_response(fake_llm_response)
    mock_extraction.save_extracted_data.return_value = None

    llm_extraction_service.perform_extraction(
        extractor_id=llm_extractor.id(),
        tenant_id=tenant_id,
        document_id="111",
        llm_type=None,
    )

    mock_extraction.save_extracted_data.assert_not_called()


def test_perform_extraction__no_extractor__error(
    fake_providers_aggregate: FakeProvidersAggregate,
    fake_llm_extractor_repository,
    mock_extraction,
    llm_extraction_service: LLMExtractionService,
    llm_extractor,
    tenant_id,
):
    fake_llm_extractor_repository.save(llm_extractor)
    fake_llm_response = "HelloWorld"
    fake_providers_aggregate.set_response(fake_llm_response)
    mock_extraction.save_extracted_data.return_value = None

    with pytest.raises(LLMExtractorNotFoundError):
        llm_extraction_service.perform_extraction(
            extractor_id="fake_id",
            tenant_id=tenant_id,
            document_id="111",
            llm_type=None,
        )


@pytest.mark.parametrize(
    "llm_type,expected_provider,expected_model",
    [
        ("dial@model", "dial", "model"),
        ("model", LLMExtractionService._DEFAULT_PROVIDER, "model"),
    ],
)
def test_perform_extraction__llm_type_provided__ok(
    llm_type: str,
    expected_provider: str,
    expected_model: str,
    fake_providers_aggregate: FakeProvidersAggregate,
    fake_llm_extractor_repository,
    mock_extraction,
    mock_unifier,
    unifier_proxy_return_value,
    llms_controller_with_fakes,
    llm_extraction_service: LLMExtractionService,
    llm_extractor_with_query,
    tenant_id,
):
    fake_llm_extractor_repository.save(llm_extractor_with_query)
    fake_llm_response = "HelloWorld"
    fake_providers_aggregate.set_response(fake_llm_response)
    mock_extraction.save_extracted_data.return_value = None
    mock_unifier.get_original_images.return_value = unifier_proxy_return_value

    llm_extraction_service.perform_extraction(
        extractor_id=llm_extractor_with_query.id(),
        tenant_id=tenant_id,
        document_id="111",
        llm_type=llm_type,
    )

    fake_providers_aggregate.assert_insights_retrival_request_made_with(
        provider=expected_provider,
        model=expected_model,
    )


def test_get_llm_extractors_for_document_type__success(
    llm_extractor: LLMExtractor,
    fake_llm_extractor_repository: FakeLLMExtractorRepository,
    llm_extraction_service: LLMExtractionService,
):
    fake_llm_extractor_repository.save(llm_extractor)

    llm_extractors = llm_extraction_service.get_llm_extractors_for_document_type(
        document_type_id=llm_extractor.document_type_id(),
        tenant_id=llm_extractor.tenant_id(),
    )

    assert len(llm_extractors) == 1
    assert llm_extractors[0] == llm_extractor


def test_delete_query__success(
    llm_extractor_with_query: LLMExtractor,
    fake_llm_extractor_repository: FakeLLMExtractorRepository,
    llm_extraction_service: LLMExtractionService,
):
    query_for_delete = llm_extractor_with_query.query_values()[0]
    fake_llm_extractor_repository.save(llm_extractor_with_query)
    expected_len_of_queries = len(llm_extractor_with_query.queries) - 1

    llm_extraction_service.delete_query(
        code=query_for_delete.code,
        extractor_id=llm_extractor_with_query.id(),
        document_type_id=llm_extractor_with_query.document_type_id(),
        tenant_id=llm_extractor_with_query.tenant_id(),
    )

    saved_llm_extractor = fake_llm_extractor_repository.find_for_document_type(
        id_=llm_extractor_with_query.id(),
        document_type_id=llm_extractor_with_query.document_type_id(),
        tenant_id=llm_extractor_with_query.tenant_id(),
    )

    assert len(saved_llm_extractor.queries) == expected_len_of_queries
    assert query_for_delete.code not in saved_llm_extractor.queries


def test_delete_query__query_not_found(
    llm_extractor_with_query: LLMExtractor,
    fake_llm_extractor_repository: FakeLLMExtractorRepository,
    llm_extraction_service: LLMExtractionService,
):
    fake_llm_extractor_repository.save(llm_extractor_with_query)

    with pytest.raises(QueryNotFoundError):
        llm_extraction_service.delete_query(
            code=str(uuid.uuid4()),
            extractor_id=llm_extractor_with_query.id(),
            document_type_id=llm_extractor_with_query.document_type_id(),
            tenant_id=llm_extractor_with_query.tenant_id(),
        )


def test_create_extractor_step__coordinates_enabled_true__aggregate_saved_with_setting(
    fake_llm_extractor_repository: FakeLLMExtractorRepository,
    mock_extraction_proxy: FakeExtractionProxy,
    tenant_id: str,
):
    steps = CreateLLMExtractorSteps(
        extraction_proxy=mock_extraction_proxy,
        extractor_repository=fake_llm_extractor_repository,
    )
    data = CreateLLMExtractorSagaData(
        extractor_name="extractor",
        document_type_name="doc-type",
        tenant_id=tenant_id,
        model="model",
        provider=ProviderCode.EPAM_DIAL,
        custom_instruction="ci",
        grouping_factor=3,
        temperature=0.0,
        top_p=1.0,
        page_span=None,
        context_attachments=None,
        coordinates_enabled=True,
    )

    steps.attach_extractor(data)
    steps.create_extractor(data)

    saved = fake_llm_extractor_repository.find_by_id(data.extractor_id, tenant_id)
    assert saved is not None
    assert saved.extraction_params.coordinates_enabled is True


def test_create_extractor_step__coordinates_enabled_false__aggregate_saved_disabled(
    fake_llm_extractor_repository: FakeLLMExtractorRepository,
    mock_extraction_proxy: FakeExtractionProxy,
    tenant_id: str,
):
    steps = CreateLLMExtractorSteps(
        extraction_proxy=mock_extraction_proxy,
        extractor_repository=fake_llm_extractor_repository,
    )
    data = CreateLLMExtractorSagaData(
        extractor_name="extractor",
        document_type_name="doc-type",
        tenant_id=tenant_id,
        model="model",
        provider=ProviderCode.EPAM_DIAL,
        custom_instruction="ci",
        grouping_factor=3,
        temperature=0.0,
        top_p=1.0,
        page_span=None,
        context_attachments=None,
        coordinates_enabled=False,
    )

    steps.attach_extractor(data)
    steps.create_extractor(data)

    saved = fake_llm_extractor_repository.find_by_id(data.extractor_id, tenant_id)
    assert saved is not None
    assert saved.extraction_params.coordinates_enabled is False


def test_update_extractor__coordinates_enabled_false__aggregate_saved_disabled(
    llm_extractor: LLMExtractor,
    fake_llm_extractor_repository: FakeLLMExtractorRepository,
    llm_extraction_service: LLMExtractionService,
):
    llm_extractor.extraction_params = ExtractionParamsFactory(coordinates_enabled=True)
    fake_llm_extractor_repository.save(llm_extractor)

    llm_extraction_service.update_extractor(
        extractor_id=llm_extractor.id(),
        document_type_id=llm_extractor.document_type_id(),
        tenant_id=llm_extractor.tenant_id(),
        name=llm_extractor.name,
        custom_instruction="updated",
        grouping_factor=3,
        temperature=0.0,
        top_p=1.0,
        coordinates_enabled=False,
    )

    saved = fake_llm_extractor_repository.find_for_document_type(
        id_=llm_extractor.id(),
        document_type_id=llm_extractor.document_type_id(),
        tenant_id=llm_extractor.tenant_id(),
    )
    assert saved.extraction_params.coordinates_enabled is False


def test_update_extractor__repository_failure__infrastructure_error_propagated(
    llm_extractor: LLMExtractor,
    fake_llm_extractor_repository: FakeLLMExtractorRepository,
    llm_extraction_service: LLMExtractionService,
    mocker,
):
    fake_llm_extractor_repository.save(llm_extractor)
    mocker.patch.object(fake_llm_extractor_repository, "save", side_effect=RuntimeError("db failure"))

    with pytest.raises(RuntimeError, match="db failure"):
        llm_extraction_service.update_extractor(
            extractor_id=llm_extractor.id(),
            document_type_id=llm_extractor.document_type_id(),
            tenant_id=llm_extractor.tenant_id(),
            name=llm_extractor.name,
            custom_instruction="updated",
            grouping_factor=3,
            temperature=0.0,
            top_p=1.0,
            coordinates_enabled=True,
        )


def test_perform_extraction__coordinates_disabled__processor_not_called(
    fake_providers_aggregate: FakeProvidersAggregate,
    mock_extraction,
    mock_unifier,
    unifier_proxy_return_value,
    coordinates_processor_mock,
    fake_llm_extractor_repository: FakeLLMExtractorRepository,
    llm_extraction_service_with_coordinates_mock: LLMExtractionService,
    llm_extractor_with_query: LLMExtractor,
    tenant_id: str,
):
    fake_llm_extractor_repository.save(llm_extractor_with_query)
    fake_providers_aggregate.set_response("value")
    mock_extraction.save_extracted_data.return_value = None
    mock_unifier.get_original_images.return_value = unifier_proxy_return_value

    llm_extraction_service_with_coordinates_mock.perform_extraction(
        extractor_id=llm_extractor_with_query.id(),
        tenant_id=tenant_id,
        document_id="111",
    )

    coordinates_processor_mock.add_llm_coordinates.assert_not_called()
    mock_extraction.save_extracted_data.assert_called_once()


def test_perform_extraction__coordinates_enabled__enriched_result_saved(
    fake_providers_aggregate: FakeProvidersAggregate,
    mock_extraction,
    coordinates_processor_mock,
    fake_llm_extractor_repository: FakeLLMExtractorRepository,
    llm_extraction_service_with_coordinates_mock: LLMExtractionService,
    tenant_id: str,
    document_type_id: str,
):
    query = QueryFactory.create()
    extractor = LLMExtractorFactory.create(
        tenant_id=tenant_id,
        document_type_id=document_type_id,
        extraction_params=ExtractionParamsFactory(
            coordinates_enabled=True,
            context_attachments=None,
            page_span=None,
        ),
        queries={query.code: query},
    )
    extractor.events.clear()
    fake_llm_extractor_repository.save(extractor)
    fake_providers_aggregate.set_response("value")
    enriched = ExtractedDataFactory.make_extracted_data(999)
    coordinates_processor_mock.add_llm_coordinates.return_value = enriched
    mock_extraction.save_extracted_data.return_value = None

    llm_extraction_service_with_coordinates_mock.perform_extraction(
        extractor_id=extractor.id(),
        tenant_id=tenant_id,
        document_id="111",
    )

    coordinates_processor_mock.add_llm_coordinates.assert_called_once()
    mock_extraction.save_extracted_data.assert_called_once_with(enriched)


def test_perform_extraction__coordinate_provider_fails__original_result_saved(
    fake_providers_aggregate: FakeProvidersAggregate,
    mock_extraction,
    coordinates_processor_mock,
    fake_llm_extractor_repository: FakeLLMExtractorRepository,
    llm_extraction_service_with_coordinates_mock: LLMExtractionService,
    tenant_id: str,
    document_type_id: str,
):
    query = QueryFactory.create()
    extractor = LLMExtractorFactory.create(
        tenant_id=tenant_id,
        document_type_id=document_type_id,
        extraction_params=ExtractionParamsFactory(
            coordinates_enabled=True,
            context_attachments=None,
            page_span=None,
        ),
        queries={query.code: query},
    )
    extractor.events.clear()
    fake_llm_extractor_repository.save(extractor)
    fake_providers_aggregate.set_response("value")
    coordinates_processor_mock.add_llm_coordinates.side_effect = ParsingProxyError("fail")
    mock_extraction.save_extracted_data.return_value = None

    llm_extraction_service_with_coordinates_mock.perform_extraction(
        extractor_id=extractor.id(),
        tenant_id=tenant_id,
        document_id="111",
    )

    mock_extraction.save_extracted_data.assert_called_once()
    saved = mock_extraction.save_extracted_data.call_args.args[0]
    assert saved is not None


def test_perform_extraction__extracted_data_save_fails__error_propagated(
    fake_providers_aggregate: FakeProvidersAggregate,
    mock_extraction,
    mock_unifier,
    unifier_proxy_return_value,
    coordinates_processor_mock,
    fake_llm_extractor_repository: FakeLLMExtractorRepository,
    llm_extraction_service_with_coordinates_mock: LLMExtractionService,
    llm_extractor_with_query: LLMExtractor,
    tenant_id: str,
):
    fake_llm_extractor_repository.save(llm_extractor_with_query)
    fake_providers_aggregate.set_response("value")
    mock_unifier.get_original_images.return_value = unifier_proxy_return_value
    mock_extraction.save_extracted_data.side_effect = RuntimeError("core save failed")

    with pytest.raises(RuntimeError, match="core save failed"):
        llm_extraction_service_with_coordinates_mock.perform_extraction(
            extractor_id=llm_extractor_with_query.id(),
            tenant_id=tenant_id,
            document_id="111",
        )

    coordinates_processor_mock.add_llm_coordinates.assert_not_called()
