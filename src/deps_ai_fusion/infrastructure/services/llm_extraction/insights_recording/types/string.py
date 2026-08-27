from deps_extracted_data.model import ExtractedData
from deps_gen_ai.common import LLMResponse

from deps_ai_fusion.domain.model import Query

from ...genai_query_factory import StringResponse
from .abstract import AbstractInsightsRecorder

__all__ = ["StringInsightsRecorder"]


class StringInsightsRecorder(AbstractInsightsRecorder[StringResponse]):
    def record_insights(
        self,
        edata: ExtractedData,
        for_query: Query,
        insight: LLMResponse[StringResponse],
    ) -> None:
        parsed = insight.parsed if isinstance(insight.parsed, StringResponse) else None
        if parsed is None:
            parsed = StringResponse.from_json_content(insight.content)
        value = parsed.value if parsed else StringResponse.parse_llm_response(insight)
        confidence = self._compute_confidence(
            field_code=for_query.code,
            value_for_check=value,
            evidence=parsed.evidence if parsed else None,
            self_confidence=parsed.self_confidence if parsed else None,
        )
        edata.add_string(
            self.extracted_field_factory.create_string(
                field_code=for_query.code,
                value=value,
                coordinates=None,
                confidence=confidence,
            ),
        )
