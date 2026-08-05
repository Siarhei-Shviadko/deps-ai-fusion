from typing import Protocol

from .model_capabilities import ModelCapabilities

__all__ = ["IModelCapabilitiesRegistry"]


class IModelCapabilitiesRegistry(Protocol):
    def get_capabilities(self, provider: str, model: str) -> ModelCapabilities:
        ...
