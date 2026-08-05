from typing import Any, Protocol

from .model_capabilities import ModelCapabilities

__all__ = ["IModelCapabilitiesService"]


class IModelCapabilitiesService(Protocol):
    def get_capabilities(self, provider: str, model: str) -> ModelCapabilities:
        ...

    def filter_to_supported_parameters(
        self, provider: str, model: str, raw_llm_params: dict[str, Any]
    ) -> dict[str, Any]:
        ...
