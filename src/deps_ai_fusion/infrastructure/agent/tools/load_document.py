from typing import Annotated

from deps_gen_ai.common import ContextReference
from deps_gen_ai.context_creators import PlainLayoutContextCreator
from langchain_core.tools import BaseTool
from langgraph.prebuilt import InjectedState

from ..state import AgentState
from ..text_sanitizer import sanitize_text

__all__ = ["DocumentLoadingTool"]


class DocumentLoadingTool(BaseTool):
    name: str = "load-document-layout"
    description: str = (
        "Load the full document text context into memory for grounding. "
        "Use before answering questions that depend on document content or before testing a prompts_chain. "
        "Do not call more than once per turn — layout is not persisted between conversation turns."
    )

    layout_cc: PlainLayoutContextCreator

    def _run(
        self,
        reasoning: str,
        state: Annotated[AgentState, InjectedState()],
    ) -> str:
        raw = self.layout_cc.context_of(
            ContextReference.from_raw_params(entity_id=state.document_id),
        )
        return sanitize_text(raw)
