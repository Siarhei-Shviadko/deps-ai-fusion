from uuid import uuid4

from deps_gen_ai.providers import ProviderCode
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from deps_ai_fusion.domain.model.llm_extractor import (
    LLMExtractorFactory as NativeLLMExtractorFactory,
)
from deps_ai_fusion.infrastructure.repositories import LLMExtractorRepository
from deps_ai_fusion.infrastructure.repositories.llm_extractor.mappers import (
    LLMExtractorMapper,
)
from deps_ai_fusion.infrastructure.tables import llm_extractor_table
from tests.factories import ExtractionParamsFactory, LLMExtractorFactory


def test_save_and_get__coordinates_enabled_true__value_round_trips(
    llm_extractor_repository: LLMExtractorRepository,
    llm_extractor_id: str,
    document_type_id: str,
    tenant_id: str,
) -> None:
    extractor = LLMExtractorFactory.create(
        id_=llm_extractor_id,
        document_type_id=document_type_id,
        tenant_id=tenant_id,
        extraction_params=ExtractionParamsFactory(coordinates_enabled=True),
    )
    extractor.events.clear()
    llm_extractor_repository.save(extractor)

    saved = llm_extractor_repository.get(id_=extractor.id(), tenant_id=extractor.tenant_id())

    assert saved is not None
    assert saved.extraction_params.coordinates_enabled is True


def test_save_and_get__coordinates_enabled_false__value_round_trips(
    llm_extractor_repository: LLMExtractorRepository,
    llm_extractor_id: str,
    document_type_id: str,
    tenant_id: str,
) -> None:
    extractor = LLMExtractorFactory.create(
        id_=llm_extractor_id,
        document_type_id=document_type_id,
        tenant_id=tenant_id,
        extraction_params=ExtractionParamsFactory(coordinates_enabled=False),
    )
    extractor.events.clear()
    llm_extractor_repository.save(extractor)

    saved = llm_extractor_repository.get(id_=extractor.id(), tenant_id=extractor.tenant_id())

    assert saved is not None
    assert saved.extraction_params.coordinates_enabled is False


def test_get__legacy_extraction_params_without_coordinates_enabled__defaults_to_false(
    llm_extractor_repository: LLMExtractorRepository,
    containers,
    llm_extractor_id: str,
    document_type_id: str,
    tenant_id: str,
) -> None:
    extractor = NativeLLMExtractorFactory.create(
        id_=llm_extractor_id,
        tenant_id=tenant_id,
        document_type_id=document_type_id,
        provider=ProviderCode.EPAM_DIAL,
        name=uuid4().hex,
        model=uuid4().hex,
    )
    raw = LLMExtractorMapper.to_dict(extractor)
    raw["extraction_params"].pop("coordinates_enabled", None)

    db = containers.datasources.postgres_datasource()
    with db.connection() as conn:
        conn.execute(
            insert(llm_extractor_table).values(
                id=raw["id"],
                tenant_id=raw["tenant_id"],
                document_type_id=raw["document_type_id"],
                llm_reference=raw["llm_reference"],
                name=raw["name"],
                queries=raw["queries"],
                extraction_params=raw["extraction_params"],
            )
        )

    loaded = llm_extractor_repository.get(id_=llm_extractor_id, tenant_id=tenant_id)

    assert loaded is not None
    assert loaded.extraction_params.coordinates_enabled is False


def test_save__loaded_legacy_extractor__coordinates_enabled_normalized(
    llm_extractor_repository: LLMExtractorRepository,
    containers,
    llm_extractor_id: str,
    document_type_id: str,
    tenant_id: str,
) -> None:
    extractor = NativeLLMExtractorFactory.create(
        id_=llm_extractor_id,
        tenant_id=tenant_id,
        document_type_id=document_type_id,
        provider=ProviderCode.EPAM_DIAL,
        name=uuid4().hex,
        model=uuid4().hex,
    )
    raw = LLMExtractorMapper.to_dict(extractor)
    raw["extraction_params"].pop("coordinates_enabled", None)

    db = containers.datasources.postgres_datasource()
    with db.connection() as conn:
        conn.execute(
            insert(llm_extractor_table).values(
                id=raw["id"],
                tenant_id=raw["tenant_id"],
                document_type_id=raw["document_type_id"],
                llm_reference=raw["llm_reference"],
                name=raw["name"],
                queries=raw["queries"],
                extraction_params=raw["extraction_params"],
            )
        )

    loaded = llm_extractor_repository.get(id_=llm_extractor_id, tenant_id=tenant_id)
    assert loaded is not None
    assert loaded.extraction_params.coordinates_enabled is False

    llm_extractor_repository.save(loaded)

    with db.connection() as conn:
        row = conn.execute(
            select(llm_extractor_table.c.extraction_params).where(
                llm_extractor_table.c.id == llm_extractor_id,
                llm_extractor_table.c.tenant_id == tenant_id,
            )
        ).fetchone()

    assert row is not None
    assert row.extraction_params["coordinates_enabled"] is False
