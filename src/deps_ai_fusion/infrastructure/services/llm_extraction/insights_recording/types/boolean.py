from deps_extracted_data.model import CheckboxValue, ExtractedData
from deps_gen_ai.common import LLMResponse

from deps_ai_fusion.domain.model import Query

from ...confidence.rule_engine import SchemaHint
from ...genai_query_factory import BooleanResponse
from .abstract import AbstractInsightsRecorder

__all__ = ["BooleanInsightsRecorder"]


class BooleanInsightsRecorder(AbstractInsightsRecorder[BooleanResponse]):
    def record_insights(
        self,
        edata: ExtractedData,
        for_query: Query,
        insight: LLMResponse[BooleanResponse],
    ) -> None:
        value = BooleanResponse.parse_llm_response(insight)
        parsed = insight.parsed if isinstance(insight.parsed, BooleanResponse) else None
        if parsed is None and isinstance(insight.parsed, dict):
            evidence = self._ev_from_raw(insight.parsed.get("evidence"))
            self_confidence = self._sc_from_raw(insight.parsed.get("self_confidence"))
        else:
            evidence = parsed.evidence if parsed else None
            self_confidence = parsed.self_confidence if parsed else None
        confidence = self._compute_confidence(
            field_code=for_query.code,
            value_for_check=value,
            evidence=evidence,
            self_confidence=self_confidence,
            schema_hint=SchemaHint.BOOLEAN,
        )
        edata.add_checkbox(
            self.extracted_field_factory.create_checkbox(
                field_code=for_query.code,
                value=CheckboxValue.create(value),
                coordinates=None,
                confidence=confidence,
            ),
        )
