from deps_extracted_data.model import ExtractedData
from deps_gen_ai.common import LLMResponse

from deps_ai_fusion.domain.model import Query

from ...genai_query_factory import KeyValuePairResponse, KeyValuePairsListResponse
from .abstract import AbstractInsightsRecorder
from .kv_confidence import KeyValueConfidenceComputer

__all__ = ["KeyValuePairListInsightsRecorder"]


class KeyValuePairListInsightsRecorder(AbstractInsightsRecorder[KeyValuePairsListResponse]):
    def record_insights(
        self,
        edata: ExtractedData,
        for_query: Query,
        insight: LLMResponse[KeyValuePairsListResponse],
    ) -> None:
        parsed = insight.parsed if isinstance(insight.parsed, KeyValuePairsListResponse) else None
        if parsed is not None:
            items: list[KeyValuePairResponse] = parsed.items
        elif isinstance(insight.parsed, dict):
            items = [
                KeyValuePairResponse(
                    reasoning=raw.get("reasoning", ""),
                    key=str(raw.get("key", "")),
                    value=str(raw.get("value", "")),
                    key_evidence=self._ev_from_raw(raw.get("key_evidence")),
                    key_self_confidence=self._sc_from_raw(raw.get("key_self_confidence")),
                    value_evidence=self._ev_from_raw(raw.get("value_evidence")),
                    value_self_confidence=self._sc_from_raw(raw.get("value_self_confidence")),
                )
                for raw in insight.parsed.get("items", [])
                if isinstance(raw, dict)
            ]
        else:
            items = [
                KeyValuePairResponse(reasoning="", key=raw["key"], value=raw["value"])
                for raw in KeyValuePairsListResponse.parse_llm_response(insight)
            ]

        elements = []
        for item in items:
            computer = KeyValueConfidenceComputer(self._confidence_evaluator, item)
            elements.append(
                self.field_data_factory.create_key_value_pair(
                    key=self.field_data_factory.create_string(
                        value=item.key,
                        coordinates=None,
                        confidence=computer.for_key(for_query.code, item.key, is_list=True),
                    ),
                    value=self.field_data_factory.create_string(
                        value=item.value,
                        coordinates=None,
                        confidence=computer.for_value(for_query.code, item.value, is_list=True),
                    ),
                )
            )

        edata.add_key_value_pair_list(
            self.extracted_field_factory.create_key_value_pair_list(
                field_code=for_query.code,
                elements=elements,
            ),
        )
