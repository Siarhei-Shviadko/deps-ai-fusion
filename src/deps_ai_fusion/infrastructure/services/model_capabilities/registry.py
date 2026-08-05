from types import MappingProxyType

from deps_ai_fusion.domain.model import (
    IModelCapabilitiesRegistry,
    ModelCapabilities,
    ParameterSupport,
)

__all__ = ["StaticModelCapabilitiesRegistry"]


_OPENAI_DEFAULTS = MappingProxyType(
    {
        "temperature": ParameterSupport.SUPPORTED,
        "top_p": ParameterSupport.SUPPORTED,
        "max_tokens": ParameterSupport.SUPPORTED,
        "stop": ParameterSupport.SUPPORTED,
        "seed": ParameterSupport.SUPPORTED,
        "logprobs": ParameterSupport.SUPPORTED,
    }
)

_AZURE_DEFAULTS = _OPENAI_DEFAULTS
_DIAL_DEFAULTS = _OPENAI_DEFAULTS

_GOOGLE_DEFAULTS = MappingProxyType(
    {
        "temperature": ParameterSupport.SUPPORTED,
        "top_p": ParameterSupport.SUPPORTED,
        "max_tokens": ParameterSupport.SUPPORTED,
        "stop": ParameterSupport.SUPPORTED,
        "seed": ParameterSupport.NOT_SUPPORTED,
        "logprobs": ParameterSupport.NOT_SUPPORTED,
    }
)

_AWS_BEDROCK_DEFAULTS = MappingProxyType(
    {
        "temperature": ParameterSupport.SUPPORTED,
        "top_p": ParameterSupport.SUPPORTED,
        "max_tokens": ParameterSupport.SUPPORTED,
        "stop": ParameterSupport.SUPPORTED,
        "seed": ParameterSupport.NOT_SUPPORTED,
        "logprobs": ParameterSupport.NOT_SUPPORTED,
    }
)

_PROVIDER_DEFAULTS = {  # noqa: WPS407
    "openai": _OPENAI_DEFAULTS,
    "azure": _AZURE_DEFAULTS,
    "google": _GOOGLE_DEFAULTS,
    "dial": _DIAL_DEFAULTS,
    "aws-bedrock": _AWS_BEDROCK_DEFAULTS,
}


class StaticModelCapabilitiesRegistry(IModelCapabilitiesRegistry):
    def __init__(self, env_overrides: dict[str, dict[str, str]] | None = None) -> None:
        """Initialize registry with optional environment-based overrides.

        Args:
            env_overrides: Dict mapping "provider" or "provider/model" keys to
                          parameter support dicts. Example:
                          {
                              "openai": {"temperature": "SUPPORTED"},
                              "openai.custom-model": {"max_tokens": "NOT_SUPPORTED"}
                          }
        """

        self._env_overrides = {key.casefold(): value for key, value in (env_overrides or {}).items()}

    def get_capabilities(self, provider: str, model: str) -> ModelCapabilities:
        merged: dict[str, str] = {}
        llm_reference = f"{provider}.{model}"

        merged.update(_PROVIDER_DEFAULTS[provider])

        merged.update(self._env_overrides.get(provider, {}))

        merged.update(self._env_overrides.get(llm_reference, {}))

        return ModelCapabilities(
            model_id=model,
            provider=provider,
            parameter_support={key: ParameterSupport(value) for key, value in merged.items()},
        )
