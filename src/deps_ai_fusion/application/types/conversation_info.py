from dataclasses import dataclass

from deps_gen_ai.common import LLM, Provider
from deps_gen_ai.providers import ProviderCode

from deps_ai_fusion.domain.model.conversation import Conversation
from deps_ai_fusion.infrastructure.services.processed_insight import ProcessedInsight

__all__ = ["ConversationInfo"]


@dataclass(frozen=True, slots=True)
class ConversationInfo:
    conversation: Conversation
    providers: list[Provider]
    models: dict[ProviderCode, list[LLM]]
    processed_completions: dict[str, ProcessedInsight]
