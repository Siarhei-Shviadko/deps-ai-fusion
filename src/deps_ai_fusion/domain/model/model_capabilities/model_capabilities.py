from ..shared import Guard, ImmutableCheck
from .parameter_support import ParameterSupport

__all__ = [
    "ModelCapabilities",
]


class ModelCapabilities:
    model_id = Guard[str](str, ImmutableCheck())
    provider = Guard[str](str, ImmutableCheck())
    parameter_support = Guard[dict](dict, ImmutableCheck())

    def __init__(self, model_id: str, provider: str, parameter_support: dict[str, ParameterSupport]) -> None:
        self.model_id = model_id
        self.provider = provider
        self.parameter_support = parameter_support

    def __repr__(self) -> str:
        return f"ModelCapabilities(model_id={self.model_id}, provider={self.provider}, parameter_support={self.parameter_support})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ModelCapabilities)
            and self.model_id == other.model_id
            and self.provider == other.provider
            and self.parameter_support == other.parameter_support
        )

    def support_of(self, parameter: str) -> ParameterSupport:
        return self.parameter_support.get(parameter, ParameterSupport.NOT_SUPPORTED)

    def is_supported(self, parameter: str) -> bool:
        return self.support_of(parameter) != ParameterSupport.NOT_SUPPORTED
