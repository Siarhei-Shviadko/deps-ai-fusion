from deps_ai_fusion.domain.model import ParameterSupport


def test_get_capabilities__dial_model__returns_registry_capabilities(model_capabilities_service):
    caps = model_capabilities_service.get_capabilities("dial", "gpt-4o")
    assert caps.provider == "dial"
    assert caps.model_id == "gpt-4o"
    assert caps.support_of("temperature") == ParameterSupport.SUPPORTED


def test_get_capabilities__env_overrides_model_specific(model_capabilities_service):
    caps = model_capabilities_service.get_capabilities("dial", "gpt-5-mini-text")
    assert caps.support_of("logprobs") == ParameterSupport.NOT_SUPPORTED


def test_filter_to_supported_parameters__keeps_supported_params(model_capabilities_service):
    params = {"temperature": 0.5, "top_p": 0.9, "seed": 42}
    filtered = model_capabilities_service.filter_to_supported_parameters("dial", "gpt-4o", params)
    assert "temperature" in filtered
    assert "top_p" in filtered
    assert "seed" in filtered
    assert filtered["temperature"] == 0.5
    assert filtered["top_p"] == 0.9
    assert filtered["seed"] == 42


def test_filter_to_supported_parameters__removes_unsupported_params(model_capabilities_service):
    params = {"temperature": 0.5, "unsupported_param": "value", "logprobs": True}
    filtered = model_capabilities_service.filter_to_supported_parameters("dial", "gpt-5-mini-text", params)
    assert "temperature" in filtered
    assert "unsupported_param" not in filtered
    assert "logprobs" not in filtered


def test_filter_to_supported_parameters__model_kwargs_merged_into_result(model_capabilities_service):
    params = {"temperature": 0.5, "model_kwargs": {"logprobs": False, "custom_param": "value"}}
    filtered = model_capabilities_service.filter_to_supported_parameters("dial", "gpt-4o", params)
    assert "temperature" in filtered
    assert "model_kwargs" not in filtered
    assert "logprobs" in filtered
    assert "custom_param" in filtered
    assert filtered["logprobs"] is False
    assert filtered["custom_param"] == "value"


def test_filter_to_supported_parameters__model_kwargs_override_supported_params(model_capabilities_service):
    params = {"temperature": 0.5, "model_kwargs": {"temperature": 0.8}}
    filtered = model_capabilities_service.filter_to_supported_parameters("dial", "gpt-4o", params)
    assert filtered["temperature"] == 0.8
