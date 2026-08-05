from typing import Any

from deps_ai_fusion.domain.model import IModelCapabilitiesRegistry, ModelCapabilities

__all__ = ["ModelCapabilitiesService"]


class ModelCapabilitiesService:
    def __init__(self, registry: IModelCapabilitiesRegistry) -> None:
        self._registry = registry

    def get_capabilities(self, provider: str, model: str) -> ModelCapabilities:
        return self._registry.get_capabilities(provider, model)

    def filter_to_supported_parameters(
        self,
        provider: str,
        model: str,
        raw_llm_params: dict[str, Any],
    ) -> dict[str, Any]:
        caps = self._registry.get_capabilities(provider, model)

        input_model_kwargs = raw_llm_params.pop("model_kwargs", {})

        result = {}
        for key, value in raw_llm_params.items():
            if caps.is_supported(key):
                result[key] = value

        result.update(input_model_kwargs)

        return result
