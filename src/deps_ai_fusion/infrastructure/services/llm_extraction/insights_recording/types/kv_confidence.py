from deps_ai_fusion.infrastructure.services.llm_extraction.confidence import (
    ConfidenceEvaluator,
)

from ...genai_query_factory import KeyValuePairResponse

__all__: list[str] = []


class KeyValueConfidenceComputer:
    def __init__(self, evaluator: ConfidenceEvaluator, parsed: KeyValuePairResponse | None) -> None:
        self._evaluator = evaluator
        self._parsed = parsed

    def for_key(self, field_code: str, value: str, is_list: bool = False) -> float:
        return self._evaluator.evaluate(
            field_code=field_code,
            value=value,
            evidence=self._parsed.key_evidence if self._parsed else None,
            self_confidence=self._parsed.key_self_confidence if self._parsed else None,
            is_list=is_list,
        ).as_float()

    def for_value(self, field_code: str, value: str, is_list: bool = False) -> float:
        return self._evaluator.evaluate(
            field_code=field_code,
            value=value,
            evidence=self._parsed.value_evidence if self._parsed else None,
            self_confidence=self._parsed.value_self_confidence if self._parsed else None,
            is_list=is_list,
        ).as_float()
