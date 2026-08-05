from deps_ai_fusion.api.serializers.base_llm_params import BaseLLMParams


class TestBaseLLMParams:
    def test_base_llm_params__defaults__all_none_or_default_false(self) -> None:
        params = BaseLLMParams.model_validate({})

        assert params.temperature is None
        assert params.top_p is None
        assert params.max_tokens is None
        assert params.stop is None
        assert params.seed is None
        assert params.logprobs is False
        assert params.extra_model_params is None

    def test_base_llm_params__all_fields_populated(self) -> None:
        params = BaseLLMParams.model_validate(
            {
                "temperature": 0.7,
                "topP": 0.9,
                "maxTokens": 500,
                "stop": ["END", "\n"],
                "seed": 42,
                "logprobs": True,
                "extraModelParams": {"key": "value"},
            }
        )

        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.max_tokens == 500
        assert params.stop == ["END", "\n"]
        assert params.seed == 42
        assert params.logprobs is True
        assert params.extra_model_params == {"key": "value"}
