from deps_gen_ai.common import PageSpan, Query
from deps_gen_ai.providers import ProviderCode
from deps_gen_ai.providers_aggregate import ProvidersAggregate
from deps_message_flow.events.publisher import DomainEventPublisher

from deps_ai_fusion.constants import AI_FUSION_DESTINATION
from deps_ai_fusion.domain.exceptions import ConversationNotFoundError
from deps_ai_fusion.domain.model.conversation import (
    Conversation,
    ConversationFactory,
    IConversationRepository,
)
from deps_ai_fusion.infrastructure.services import InsightProcessor

from .structured_outputs import ReasoningResponse
from .types import CompletionWithInsight, ConversationInfo, RawPageSpan

__all__ = ["ConversationService"]


class ConversationService:
    def __init__(
        self,
        providers: ProvidersAggregate,
        conversation_repository: IConversationRepository,
        domain_event_publisher: DomainEventPublisher,
        insight_processor: InsightProcessor,
    ) -> None:
        self._providers = providers
        self._conversation_repository = conversation_repository
        self._domain_event_publisher = domain_event_publisher
        self._insight_processor = insight_processor

    def get_conversation(self, entity_id: str, user_id: str, tenant_id: str) -> ConversationInfo:
        conversation = self._get_or_create_conversation(entity_id=entity_id, user_id=user_id, tenant_id=tenant_id)

        providers = self._providers.supported_providers()
        models = {
            ProviderCode(provider.code): self._providers.models_of(provider=provider.code, include_legacy=False)
            for provider in providers
        }
        processed_completions = {
            code: self._insight_processor.process_stored(
                response=completion.response,
                field_code=code,
                fallback_confidence=completion.confidence,
            )
            for code, completion in conversation.completions.items()
        }

        return ConversationInfo(
            conversation=conversation,
            providers=providers,
            models=models,
            processed_completions=processed_completions,
        )

    def chat_request(
        self,
        entity_id: str,
        tenant_id: str,
        user_id: str,
        provider: str,
        model: str,
        question: str,
        page_span: RawPageSpan | None = None,
        files: list[str] | None = None,
    ) -> CompletionWithInsight:
        conversation = self._get_conversation(entity_id=entity_id, user_id=user_id, tenant_id=tenant_id)
        raw_history = conversation.form_history()

        response = self._providers.chat_request(
            provider=provider,
            model=model,
            entity_id=entity_id,
            query=Query.from_raw(prompts=[question], response_model=ReasoningResponse),
            page_span=PageSpan(start=page_span["start"], end=page_span["end"]) if page_span is not None else None,
            history=raw_history,
            files=files,
        )

        completion = conversation.add_completion(
            provider=provider,
            model=model,
            question=question,
            response=self._insight_processor.extract_content(response),
            confidence=response.confidence,
        )

        self._conversation_repository.save(conversation)
        self._publish_events(conversation)

        return CompletionWithInsight(
            completion=completion,
            processed=self._insight_processor.process_stored(
                response=completion.response,
                field_code=completion.code,
                fallback_confidence=completion.confidence,
            ),
        )

    def clear_conversation(
        self,
        entity_id: str,
        tenant_id: str,
        user_id: str,
    ) -> None:
        conversation = self._get_conversation(
            entity_id=entity_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )
        conversation.clear()

        self._conversation_repository.save(conversation)

    def remove_completions(self, entity_id: str, tenant_id: str, user_id: str, completion_codes: list[str]) -> None:
        conversation = self._get_conversation(entity_id=entity_id, user_id=user_id, tenant_id=tenant_id)
        conversation.remove_completions(completion_codes)

        self._conversation_repository.save(conversation)

    def _get_conversation(self, entity_id: str, user_id: str, tenant_id: str) -> Conversation:
        conversation = self._conversation_repository.get(
            entity_id=entity_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )

        if conversation is None:
            raise ConversationNotFoundError(
                entity_id=entity_id,
                user_id=user_id,
                tenant_id=tenant_id,
            )

        return conversation

    def _get_or_create_conversation(self, entity_id: str, user_id: str, tenant_id: str) -> Conversation:
        conversation = self._conversation_repository.get(
            entity_id=entity_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )

        if conversation is None:
            conversation = ConversationFactory.create(
                entity_id=entity_id,
                user_id=user_id,
                tenant_id=tenant_id,
            )
            self._conversation_repository.save(conversation)

        return conversation

    def _publish_events(self, conversation: Conversation) -> None:
        if conversation.events:
            self._domain_event_publisher.publish(
                aggregate_type=AI_FUSION_DESTINATION,
                aggregate_id=conversation.entity_id(),
                domain_events=conversation.drain_events(),
            )
