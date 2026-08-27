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
    "DEFAULT_COORDINATES_ENABLED",
]

CONFIDENCE_INSTRUCTIONS = """
CORE PRINCIPLES:
- Use ONLY the content of this document. Do not infer, recall, or supplement from general knowledge or training data.
- Never guess. If a field is not explicitly present in the document, return value: null.
- Never return a value without evidence.

EVIDENCE REQUIREMENTS:
For every extracted value you MUST provide evidence:
- 'text': an exact, verbatim quote from the document that directly supports the extracted value. Do not paraphrase, summarize, or reconstruct. Keep it as short and specific as possible — only the fragment that supports the value, not the surrounding paragraph.
- 'page': the 1-based page number where this evidence was found (first page = 1).
- If value is null, set evidence to null.
- Do not include any coordinate or position information (bounding boxes, offsets, etc.) — only 'text' and 'page'.

OCR ARTEFACTS:
If broken words or random symbols are visible in the source text, quote it exactly as it appears — do not clean or reconstruct. Adjust self_confidence accordingly (MEDIUM or LOW).

SELF_CONFIDENCE:
For every extracted value you MUST provide a self_confidence assessment: 'HIGH', 'MEDIUM', or 'LOW'.
- HIGH: value clearly matches a labeled field, appears exactly once in the document, format is unambiguous, evidence quote is short and specific.
- MEDIUM: value found but with some uncertainty — multiple similar candidates exist, no clear label nearby, evidence is long or broad, OCR noise is present, or disambiguation was required.
- LOW: value is inferred or guessed, conflicting candidates found, no reliable evidence located, or document quality prevents confident reading.

BOOLEAN / CHECKBOX FIELDS:
- evidence.text must quote the marker or status word together with its surrounding label — not just the label alone.
- If no explicit marker or status word is present in the document text, return value: null.

FORBIDDEN PATTERNS:
Never use the following in value or evidence.text: 'N/A', 'n/a', 'unknown', 'not found', 'not available', 'not applicable', 'unclear', 'I think', 'probably', 'cannot find'. If the field is genuinely absent, return value: null.

DISAMBIGUATION:
If multiple valid candidates exist for a field and you cannot determine which is correct after applying field-level rules provided in the user prompt, set self_confidence: 'MEDIUM'.

OUTPUT FORMAT:
- Return strictly valid JSON as specified in the user prompt.
- Do not add any fields, comments, markdown formatting, or explanations outside the JSON structure.
- Follow the exact schema requested by the user (scalar fields, list fields, key-value fields — as specified)."""

DEFAULT_CUSTOM_INSTRUCTION: str = """\
For the provided document context and based on the specific user instructions, analyze and extract relevant insights.
Focus on identifying actionable, relevant, and concise information that fulfills the purpose of the instructions.
Ensure that the extracted insights are accurate, adhere to specified constraints and respond strictly in the format requested."""
DEFAULT_GROUPING_FACTOR: int = 3
DEFAULT_TEMPERATURE: float = 0.0  # noqa: WPS358
DEFAULT_COORDINATES_ENABLED: bool = False


class ExtractionParams:  # noqa: WPS230
    custom_instruction = Guard[str](str, ImmutableCheck())
    grouping_factor = Guard[int](int, ImmutableCheck(), RangeCheck(min_value=1))
    temperature = Guard[float](float, ImmutableCheck(), RangeCheck(min_value=0, max_value=2))
    top_p = Guard[float](float, ImmutableCheck(), RangeCheck(min_value=0, max_value=1))
    page_span = Guard[PageSpan](PageSpan, ImmutableCheck())
    context_attachments = Guard[ContextAttachments](ContextAttachments, ImmutableCheck())
    extra_llm_params = Guard[dict](dict, ImmutableCheck())
    coordinates_enabled = Guard[bool](bool, ImmutableCheck())

    def __init__(
        self,
        custom_instruction: str | None = None,
        grouping_factor: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_llm_params: dict[str, Any] | None = None,
        page_span: PageSpan | None = None,
        context_attachments: ContextAttachments | None = None,
        coordinates_enabled: bool | None = None,
    ) -> None:
        self.custom_instruction = DEFAULT_CUSTOM_INSTRUCTION if custom_instruction is None else custom_instruction
        self.grouping_factor = DEFAULT_GROUPING_FACTOR if grouping_factor is None else grouping_factor
        self.temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
        self.coordinates_enabled = DEFAULT_COORDINATES_ENABLED if coordinates_enabled is None else coordinates_enabled

        if top_p is not None:
            self.top_p = top_p
        if page_span:
            self.page_span = page_span
        if context_attachments:
            self.context_attachments = context_attachments
        if extra_llm_params is not None:
            self.extra_llm_params = extra_llm_params

    @property
    def effective_instruction(self) -> str:
        return f"{self.custom_instruction}\n\n{CONFIDENCE_INSTRUCTIONS}"

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
            and other.coordinates_enabled == self.coordinates_enabled
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
                f"{self.context_attachments = },",
                f"{self.coordinates_enabled = }>",
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
        coordinates_enabled: bool = DEFAULT_COORDINATES_ENABLED,
    ) -> "ExtractionParams":
        return ExtractionParams(
            custom_instruction=custom_instruction,
            grouping_factor=grouping_factor,
            temperature=temperature,
            top_p=top_p,
            extra_llm_params=extra_llm_params,
            page_span=page_span and PageSpan.from_dict(page_span),
            context_attachments=context_attachments,
            coordinates_enabled=coordinates_enabled,
        )
