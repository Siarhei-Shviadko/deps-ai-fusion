from deps_extracted_data.model import EntityId, ExtractedData, GenericData
from deps_gen_ai.common import LLMResponse

from deps_ai_fusion.domain.model import Query

from ...genai_query_factory import StringsListWithAliasesResponse
from .abstract import AbstractInsightsRecorder

__all__ = ["StringListWithAliasesInsightsRecorder"]

_StringElements = list[GenericData[str]]


class StringListWithAliasesInsightsRecorder(AbstractInsightsRecorder[StringsListWithAliasesResponse]):
    def record_insights(
        self,
        edata: ExtractedData,
        for_query: Query,
        insight: LLMResponse[StringsListWithAliasesResponse],
    ) -> None:
        parsed = insight.parsed if isinstance(insight.parsed, StringsListWithAliasesResponse) else None
        if parsed:
            elements, aliases = self._collect_from_parsed(for_query.code, parsed)
        else:
            elements, aliases = self._collect_from_raw(for_query.code, insight)
        edata.add_string_list(
            self.extracted_field_factory.create_string_list(
                field_code=for_query.code,
                elements=elements,
                aliases=aliases,
            ),
        )

    def _collect_from_parsed(
        self, field_code: str, parsed: StringsListWithAliasesResponse
    ) -> tuple[_StringElements, dict[EntityId, str]]:
        elements: list[GenericData[str]] = []
        aliases: dict[EntityId, str] = {}
        for item in parsed.items:
            element_data = self.field_data_factory.create_string(
                value=item.value,
                confidence=self._compute_confidence(
                    field_code=field_code,
                    value_for_check=item.value,
                    evidence=item.evidence,
                    self_confidence=item.self_confidence,
                    is_list=True,
                ),
                coordinates=None,
            )
            elements.append(element_data)
            if item.alias and item.alias.strip():
                aliases[element_data.id] = item.alias
        return elements, aliases

    def _collect_from_raw(
        self, field_code: str, insight: LLMResponse[StringsListWithAliasesResponse]
    ) -> tuple[_StringElements, dict[EntityId, str]]:
        if isinstance(insight.parsed, dict):
            return self._collect_from_raw_dict(field_code, insight.parsed.get("items", []))
        return self._collect_from_raw_fallback(field_code, insight)

    def _collect_from_raw_dict(self, field_code: str, raw_items: list) -> tuple[_StringElements, dict[EntityId, str]]:
        elements: list[GenericData[str]] = []
        aliases: dict[EntityId, str] = {}
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            value = str(raw.get("value", ""))
            element_data = self.field_data_factory.create_string(
                value=value,
                confidence=self._compute_confidence(
                    field_code=field_code,
                    value_for_check=value,
                    evidence=self._ev_from_raw(raw.get("evidence")),
                    self_confidence=self._sc_from_raw(raw.get("self_confidence")),
                    is_list=True,
                ),
                coordinates=None,
            )
            elements.append(element_data)
            alias = self._truncate_alias(str(raw.get("alias", "")))
            if alias and alias.strip():
                aliases[element_data.id] = alias
        return elements, aliases

    def _collect_from_raw_fallback(
        self, field_code: str, insight: LLMResponse[StringsListWithAliasesResponse]
    ) -> tuple[_StringElements, dict[EntityId, str]]:
        elements: list[GenericData[str]] = []
        aliases: dict[EntityId, str] = {}
        for element in StringsListWithAliasesResponse.parse_llm_response(insight):
            element_data = self.field_data_factory.create_string(
                value=element["value"],
                confidence=self._compute_confidence(
                    field_code=field_code,
                    value_for_check=element["value"],
                    evidence=None,
                    self_confidence=None,
                    is_list=True,
                ),
                coordinates=None,
            )
            elements.append(element_data)
            alias = self._truncate_alias(element["alias"])
            if alias and alias.strip():
                aliases[element_data.id] = alias
        return elements, aliases
