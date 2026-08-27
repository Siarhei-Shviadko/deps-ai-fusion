from deps_extracted_data.model import ExtractedData
from deps_gen_ai.common import LLMResponse

from deps_ai_fusion.domain.model import Query

from ...genai_query_factory import KeyValuePairResponse
from .abstract import AbstractInsightsRecorder
from .kv_confidence import KeyValueConfidenceComputer

__all__ = ["KeyValuePairInsightsRecorder"]


class KeyValuePairInsightsRecorder(AbstractInsightsRecorder[KeyValuePairResponse]):
    def record_insights(
        self,
        edata: ExtractedData,
        for_query: Query,
        insight: LLMResponse[KeyValuePairResponse],
    ) -> None:
        kv_pair = KeyValuePairResponse.parse_llm_response(insight)
        parsed = insight.parsed if isinstance(insight.parsed, KeyValuePairResponse) else None
        computer = KeyValueConfidenceComputer(self._confidence_evaluator, parsed)

        edata.add_key_value_pair(
            self.extracted_field_factory.create_key_value_pair(
                field_code=for_query.code,
                key=self.field_data_factory.create_string(
                    value=kv_pair["key"],
                    coordinates=None,
                    confidence=computer.for_key(for_query.code, kv_pair["key"]),
                ),
                value=self.field_data_factory.create_string(
                    value=kv_pair["value"],
                    coordinates=None,
                    confidence=computer.for_value(for_query.code, kv_pair["value"]),
                ),
            ),
        )
