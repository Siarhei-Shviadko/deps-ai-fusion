from typing import Annotated

from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from deps_ai_fusion.domain.model import Cardinality, DataType

from ..state import AgentState

__all__ = ["ExecuteLLMExtractionRequest", "CreateGenAIFieldRequest", "DataShape", "ListDocumentTypeFieldsRequest"]


class DataShape(BaseModel):
    data_type: DataType = Field(
        description="Atomic type of the extracted value(s). Examples: STRING, BOOLEAN, KEY_VALUE_PAIR.",
    )
    cardinality: Cardinality = Field(
        description="Expected number of values: SCALAR for a single value, LIST for multiple values.",
    )
    include_aliases: bool = Field(
        False,
        description="Available only for list cardinality."
        " If True, then name/title for each element of list will be extracted.",
    )


class CreateGenAIFieldRequest(BaseModel):
    reasoning: str = Field(
        description="Brief rationale (<=20 words) for creating the field now, including user approval.",
    )
    name: str = Field(
        description="Human-readable, plain format field name unique within the Document Type.",
        examples=["Invoice Total", "Vendor Name", "Line Items"],
    )
    prompts_chain: list[str] = Field(
        description=(
            "Ordered prompts forming the extraction workflow. Each prompt must be a complete, standalone LLM instruction. "
            "Use a SINGLE prompt unless multi-step reasoning demonstrably improves accuracy.\n\n"
            "PROMPT FORMAT RULES:\n"
            "- KEY_VALUE_PAIR (SCALAR or LIST): ALWAYS use the structured template.\n"
            "- STRING / BOOLEAN with non-trivial extraction logic: use the structured template.\n"
            "- STRING / BOOLEAN simple/unambiguous: one concise sentence is sufficient.\n\n"
            "STRUCTURED TEMPLATE — mandatory sections (in order):\n"
            "  ## Objective — extraction role, field name, section, IMPORTANT RULE, return instruction\n"
            "  ## Field Instructions — per entry: Context, Field Labels, What to capture (key/value/alias), Extraction Instructions\n"
            "  ## Disambiguation Rules — label proximity, structured vs narrative preference, conflict resolution, empty-value handling\n"
            "  ## Normalization — whitespace trimming, field-specific formatting\n"
            "  ## Transformation Rules — OPTIONAL, include only if user explicitly requested transformation\n\n"
            "CRITICAL: Replace ALL <...> placeholders and [...] markers with actual content. "
            "Never leave template placeholders in the final prompt."
        ),
    )
    response_model: DataShape = Field(description="Desired response structure for the field.")

    state: Annotated[AgentState, InjectedState()]
    tool_call_id: Annotated[str, InjectedToolCallId()]


class ExecuteLLMExtractionRequest(BaseModel):
    reasoning: str = Field(description="Short hypothesis for this test (<=20 words).")
    prompts_chain: list[str] = Field(
        description=(
            "Ordered prompts to run for this test. Each prompt must follow the same format rules as in create-genai-field: "
            "KEY_VALUE_PAIR fields require the structured template (Objective / Field Instructions / "
            "Disambiguation Rules / Normalization); simple STRING/BOOLEAN fields may use a single sentence. "
            "Never leave <...> or [...] placeholders in the prompts."
        ),
    )
    response_model: DataShape = Field(description="Expected output shape for the test run.")

    state: Annotated[AgentState, InjectedState()]
    tool_call_id: Annotated[str, InjectedToolCallId()]


class DocumentTypeCreationRequest(BaseModel):
    reasoning: str = Field(description="Reason for creating the document type.")
    document_type_name: str = Field(description="Name of the document type to create.")

    tool_call_id: Annotated[str, InjectedToolCallId()]
    state: Annotated[AgentState, InjectedState()]


class ListDocumentTypeFieldsRequest(BaseModel):
    reasoning: str = Field(description="Why you need to inspect existing fields (<=20 words).")

    state: Annotated[AgentState, InjectedState()]
