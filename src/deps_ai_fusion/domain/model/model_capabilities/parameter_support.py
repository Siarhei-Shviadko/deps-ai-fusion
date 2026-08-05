from enum import Enum

__all__ = ["ParameterSupport"]


class ParameterSupport(str, Enum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
