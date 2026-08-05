from typing import Any

from pydantic import Field, field_validator

from .base import ConfiguredBaseModel

__all__ = ["BaseLLMParams"]

STOP_SEQUENCE_MAX_COUNT = 4


class BaseLLMParams(ConfiguredBaseModel):
    temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
        examples=[0.7],
        description="""Controls the randomness of text generation.
        Lower temperatures make the model more deterministic and repetitive, while higher temperatures make the model more creative and random.
        """,
    )
    top_p: float | None = Field(
        default=None,
        alias="topP",
        description="""Controls diversity via nucleus sampling.
        Only tokens with cumulative probability mass of top_p are considered. Value must be between 0 and 1.
        Lower values make output more focused and deterministic.
        """,
    )
    max_tokens: int | None = Field(
        None,
        alias="maxTokens",
        ge=1,
        examples=[4096],
        description="Maximum number of tokens to generate. Model-specific limits apply.",
    )
    stop: list[str] | None = Field(
        None,
        examples=[["END", "\n"]],
        description="Up to four sequences where the API will stop generating further tokens.",
    )
    seed: int | None = Field(
        None,
        examples=[42],
        ge=0,
        description="Seed for deterministic sampling. Not supported by all models.",
    )
    logprobs: bool = Field(
        False,
        examples=[False],
        description="Whether to return log probabilities of the output tokens. Not supported by all models.",
    )
    extra_model_params: dict[str, Any] | None = Field(
        None,
        alias="extraModelParams",
        examples=[{"n": 5}],
        description="Provider-specific parameters passed directly to the LLM API.",
    )

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float | None) -> float | None:
        if value is not None and (value < 0 or value > 2):
            raise ValueError("Temperature must be between 0 and 2")

        return value

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, value: float | None) -> float | None:
        if value is not None and (value < 0 or value > 1):
            raise ValueError("topP must be between 0 and 1")

        return value

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(value) > STOP_SEQUENCE_MAX_COUNT:
            raise ValueError("stop must contain at most 4 sequences")
        return value
