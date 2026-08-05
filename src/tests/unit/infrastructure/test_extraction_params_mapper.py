from deps_ai_fusion.domain.model import ExtractionParams
from deps_ai_fusion.infrastructure.repositories.llm_extractor.mappers.extraction_params import (
    ExtractionParamsMapper,
)


def test_extraction_params_mapper__roundtrip__preserves_extended_fields() -> None:
    original = ExtractionParams(
        custom_instruction="ci",
        grouping_factor=2,
        temperature=0.2,
        top_p=0.9,
        extra_llm_params={
            "max_tokens": 500,
            "stop": ["STOP"],
            "seed": 7,
            "frequency_penalty": -0.5,
            "presence_penalty": 0.25,
            "logprobs": True,
            "top_logprobs": 3,
            "response_format": {"type": "json_object"},
            "model_kwargs": {"x": 1},
        },
    )

    data = ExtractionParamsMapper.to_dict(original)
    restored = ExtractionParamsMapper.from_dict(data)

    assert restored.custom_instruction == original.custom_instruction
    assert restored.grouping_factor == original.grouping_factor
    assert restored.temperature == original.temperature
    assert restored.top_p == original.top_p
    assert restored.llm_params == original.llm_params


def test_extraction_params_mapper__legacy_flat_keys__loads_into_llm_params() -> None:
    legacy = {
        "custom_instruction": "c",
        "grouping_factor": 2,
        "temperature": 0.2,
        "top_p": 0.9,
        "extra_llm_params": {
            "max_tokens": 100,
            "seed": 1,
        },
        "page_span": None,
        "context_attachments": None,
    }
    restored = ExtractionParamsMapper.from_dict(legacy)
    assert restored.llm_params["max_tokens"] == 100
    assert restored.llm_params["seed"] == 1


def test_extraction_params_mapper__missing_extra_llm_params__defaults_to_empty_dict() -> None:
    data = {
        "custom_instruction": "ci",
        "grouping_factor": 2,
        "temperature": 0.2,
        "top_p": 0.9,
    }
    restored = ExtractionParamsMapper.from_dict(data)

    assert restored.extra_llm_params == {}
