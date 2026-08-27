from deps_gen_ai.providers import ProviderCode
from pydantic import Field

from deps_ai_fusion.application.types import ConversationInfo
from deps_ai_fusion.domain.model.conversation import Conversation
from deps_ai_fusion.infrastructure.services.processed_insight import ProcessedInsight

from ..base import ConfiguredBaseModel
from .completion import SerializedCompletion
from .provider import SerializedProvider

__all__ = ["SerializedConversation", "SerializedConversationInfo"]


class SerializedConversation(ConfiguredBaseModel):
    entity_id: str = Field(..., alias="entityId")
    tenant_id: str = Field(..., alias="tenantId")
    user_id: str = Field(..., alias="userId")
    completions: list[SerializedCompletion]

    @classmethod
    def from_domain(
        cls, conversation: Conversation, processed_completions: dict[str, ProcessedInsight]
    ) -> "SerializedConversation":
        return cls(
            entity_id=conversation.entity_id(),
            tenant_id=conversation.tenant_id(),
            user_id=conversation.user_id(),
            completions=[
                SerializedCompletion.from_stored(
                    completion=completion,
                    processed=processed_completions[code],
                )
                for code, completion in conversation.completions.items()
            ],
        )


class SerializedConversationInfo(ConfiguredBaseModel):
    conversation: SerializedConversation
    providers: list[SerializedProvider]

    @classmethod
    def from_dto(cls, info: ConversationInfo) -> "SerializedConversationInfo":
        return cls(
            conversation=SerializedConversation.from_domain(
                conversation=info.conversation,
                processed_completions=info.processed_completions,
            ),
            providers=[
                SerializedProvider.from_objects(
                    provider=provider,
                    models=info.models[ProviderCode(provider.code)],
                )
                for provider in info.providers
            ],
        )
