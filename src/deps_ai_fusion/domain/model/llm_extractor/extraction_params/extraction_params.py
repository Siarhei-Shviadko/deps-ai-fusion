from typing import Any

from ...shared import Guard, ImmutableCheck, RangeCheck
from ..raw_extraction_params import RawPageSpan
from .context_attachments import ContextAttachments
from .page_span import PageSpan

__all__ = [
    "ExtractionParams",
    "DEFAULT_CUSTOM_INSTRUCTION",
    "DEFAULT_GROUPING_FACTOR",
    "DEFAULT_TEMPERATURE",
]


DEFAULT_CUSTOM_INSTRUCTION: str = """\
For the provided document context and based on the specific user instructions, analyze and extract relevant insights.
Focus on identifying actionable, relevant, and concise information that fulfills the purpose of the instructions.
Ensure that the extracted insights are accurate, adhere to specified constraints and respond strictly in the format requested.\
"""  # noqa: E501
DEFAULT_GROUPING_FACTOR: int = 3
DEFAULT_TEMPERATURE: float = 0.0  # noqa: WPS358


class ExtractionParams:  # noqa: WPS230
    custom_instruction = Guard[str](str, ImmutableCheck())
    grouping_factor = Guard[int](int, ImmutableCheck(), RangeCheck(min_value=1))
    temperature = Guard[float](float, ImmutableCheck(), RangeCheck(min_value=0, max_value=2))
    top_p = Guard[float](float, ImmutableCheck(), RangeCheck(min_value=0, max_value=1))
    page_span = Guard[PageSpan](PageSpan, ImmutableCheck())
    context_attachments = Guard[ContextAttachments](ContextAttachments, ImmutableCheck())
    extra_llm_params = Guard[dict](dict, ImmutableCheck())

    def __init__(
        self,
        custom_instruction: str | None = None,
        grouping_factor: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_llm_params: dict[str, Any] | None = None,
        page_span: PageSpan | None = None,
        context_attachments: ContextAttachments | None = None,
    ) -> None:
        self.custom_instruction = DEFAULT_CUSTOM_INSTRUCTION if custom_instruction is None else custom_instruction
        self.grouping_factor = DEFAULT_GROUPING_FACTOR if grouping_factor is None else grouping_factor
        self.temperature = DEFAULT_TEMPERATURE if temperature is None else temperature

        if top_p is not None:
            self.top_p = top_p
        if page_span:
            self.page_span = page_span
        if context_attachments:
            self.context_attachments = context_attachments
        if extra_llm_params is not None:
            self.extra_llm_params = extra_llm_params

    @property
    def llm_params(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            **(self.extra_llm_params or {}),
        }

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ExtractionParams)
            and other.custom_instruction == self.custom_instruction
            and other.grouping_factor == self.grouping_factor
            and other.temperature == self.temperature
            and other.top_p == self.top_p
            and other.extra_llm_params == self.extra_llm_params
            and other.page_span == self.page_span
            and other.context_attachments == self.context_attachments
        )

    def __repr__(self) -> str:
        return "\n".join(
            (
                f"<class '{self.__class__.__name__}':",
                f"{self.custom_instruction = },",
                f"{self.grouping_factor = },",
                f"{self.temperature = },",
                f"{self.top_p = },",
                f"{self.extra_llm_params = },",
                f"{self.page_span = },",
                f"{self.context_attachments = }>",
            ),
        )

    def create_updated(
        self,
        custom_instruction: str,
        grouping_factor: int,
        temperature: float,
        top_p: float | None,
        extra_llm_params: dict[str, Any] | None = None,
        page_span: RawPageSpan | None = None,
        context_attachments: ContextAttachments | None = None,
    ) -> "ExtractionParams":
        return ExtractionParams(
            custom_instruction=custom_instruction,
            grouping_factor=grouping_factor,
            temperature=temperature,
            top_p=top_p,
            extra_llm_params=extra_llm_params,
            page_span=page_span and PageSpan.from_dict(page_span),
            context_attachments=context_attachments,
        )
