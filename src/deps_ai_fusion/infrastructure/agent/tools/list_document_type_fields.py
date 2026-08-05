from typing import Annotated

from langchain_core.tools import ArgsSchema, BaseTool
from langgraph.prebuilt import InjectedState

from deps_ai_fusion.infrastructure.proxies import ExtractionProxy
from deps_ai_fusion.infrastructure.proxies.exceptions import RestClientError

from ..state import AgentState
from .schemas import ListDocumentTypeFieldsRequest

__all__ = ["ListDocumentTypeFieldsTool"]


class ListDocumentTypeFieldsTool(BaseTool):
    name: str = "list-document-type-fields"
    description: str = (
        "List all existing GenAI Fields for the current Document Type. "
        "Always call this before create-genai-field to verify no field with the same name or purpose already exists."
    )
    args_schema: ArgsSchema | None = ListDocumentTypeFieldsRequest

    extraction_proxy: ExtractionProxy

    def _run(
        self,
        reasoning: str,
        state: Annotated[AgentState, InjectedState()],
    ) -> str:
        if state.document_type_id is None:
            return "No Document Type is set yet — no fields to list."

        try:
            fields = self.extraction_proxy.list_extraction_fields(
                document_type_id=state.document_type_id,
            )
        except RestClientError as exc:
            return f"Could not retrieve fields: {exc}. Proceeding without field list."

        if not fields:
            return "No fields exist yet for this Document Type."

        lines = [f"- {f.get('name', '?')} (code: {f.get('pk', '?')}, type: {f.get('type', '?')})" for f in fields]
        return "\n".join(["Existing fields:", *lines])
