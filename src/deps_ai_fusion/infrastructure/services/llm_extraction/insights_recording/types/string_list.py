from deps_extracted_data.model import ExtractedData
from deps_gen_ai.common import LLMResponse

from deps_ai_fusion.domain.model import Query

from ...genai_query_factory import StringsListResponse
from .abstract import AbstractInsightsRecorder

__all__ = ["StringListInsightsRecorder"]


class StringListInsightsRecorder(AbstractInsightsRecorder[StringsListResponse]):
    def record_insights(
        self,
        edata: ExtractedData,
        for_query: Query,
        insight: LLMResponse[StringsListResponse],
    ) -> None:
        parsed = insight.parsed if isinstance(insight.parsed, StringsListResponse) else None

        if parsed:
            elements = [
                self.field_data_factory.create_string(
                    value=item.value,
                    confidence=self._compute_confidence(
                        field_code=for_query.code,
                        value_for_check=item.value,
                        evidence=item.evidence,
                        self_confidence=item.self_confidence,
                        is_list=True,
                    ),
                    coordinates=None,
                )
                for item in parsed.values
            ]
        else:
            raw_items = insight.parsed.get("values", []) if isinstance(insight.parsed, dict) else []
            elements = [
                self.field_data_factory.create_string(
                    value=str(raw.get("value", "")),
                    confidence=self._compute_confidence(
                        field_code=for_query.code,
                        value_for_check=str(raw.get("value", "")),
                        evidence=self._ev_from_raw(raw.get("evidence")),
                        self_confidence=self._sc_from_raw(raw.get("self_confidence")),
                        is_list=True,
                    ),
                    coordinates=None,
                )
                for raw in raw_items
                if isinstance(raw, dict) and raw.get("value")
            ] or [
                self.field_data_factory.create_string(
                    value=v,
                    confidence=self._compute_confidence(
                        field_code=for_query.code, value_for_check=v, evidence=None, self_confidence=None, is_list=True
                    ),
                    coordinates=None,
                )
                for v in StringsListResponse.parse_llm_response(insight)
            ]

        edata.add_string_list(
            self.extracted_field_factory.create_string_list(
                field_code=for_query.code,
                elements=elements,
            ),
        )
