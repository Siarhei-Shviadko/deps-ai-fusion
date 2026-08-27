from dataclasses import dataclass

from deps_ai_fusion.domain.model.conversation import Completion
from deps_ai_fusion.infrastructure.services.processed_insight import ProcessedInsight

__all__ = ["CompletionWithInsight"]


@dataclass
class CompletionWithInsight:
    completion: Completion
    processed: ProcessedInsight
