from deps_extracted_data.model import CheckboxValue, ExtractedData
from deps_gen_ai.common import LLMResponse

from deps_ai_fusion.domain.model import Query

from ...confidence.rule_engine import SchemaHint
from ...genai_query_factory import BooleanResponse, BooleansListResponse
from .abstract import AbstractInsightsRecorder

__all__ = ["BooleanListInsightsRecorder"]


class BooleanListInsightsRecorder(AbstractInsightsRecorder[BooleansListResponse]):
    def record_insights(
        self,
        edata: ExtractedData,
        for_query: Query,
        insight: LLMResponse[BooleansListResponse],
    ) -> None:
        parsed = insight.parsed if isinstance(insight.parsed, BooleansListResponse) else None
        items: list[BooleanResponse] = (
            parsed.values
            if parsed is not None
            else [BooleanResponse(reasoning="", value=v) for v in BooleansListResponse.parse_llm_response(insight)]
        )

        elements = [
            self.field_data_factory.create_checkbox(
                value=CheckboxValue.create(item.value),
                confidence=self._compute_confidence(
                    field_code=for_query.code,
                    value_for_check=item.value,
                    evidence=item.evidence,
                    self_confidence=item.self_confidence,
                    is_list=True,
                    schema_hint=SchemaHint.BOOLEAN,
                ),
                coordinates=None,
            )
            for item in items
        ]

        edata.add_checkbox_list(
            self.extracted_field_factory.create_checkbox_list(
                field_code=for_query.code,
                elements=elements,
            ),
        )
