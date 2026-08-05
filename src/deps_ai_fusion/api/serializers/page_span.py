from typing import Any

from pydantic import Field, model_validator

from .base import ConfiguredBaseModel

__all__ = ["SerializedPageSpan"]


class SerializedPageSpan(ConfiguredBaseModel):
    start: int = Field(..., ge=1)
    end: int = Field(..., ge=1)

    @model_validator(mode="after")
    def check_order(self) -> "SerializedPageSpan":
        if self.start > self.end:
            raise ValueError("PageSpan.start must be less or equal to end")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end}
