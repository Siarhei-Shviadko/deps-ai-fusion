from deps_extracted_data.model import EntityId, ExtractedData
from deps_gen_ai.common import LLMResponse

from deps_ai_fusion.domain.model import Query

from ...genai_query_factory import KeyValuePairsListWithAliasesResponse
from .abstract import AbstractInsightsRecorder

__all__ = ["KeyValuePairListWithAliasesInsightsRecorder"]


class KeyValuePairListWithAliasesInsightsRecorder(AbstractInsightsRecorder[KeyValuePairsListWithAliasesResponse]):
    def record_insights(
        self,
        edata: ExtractedData,
        for_query: Query,
        insight: LLMResponse[KeyValuePairsListWithAliasesResponse],
    ) -> None:
        parsed = insight.parsed if isinstance(insight.parsed, KeyValuePairsListWithAliasesResponse) else None
        if parsed:
            elements, aliases = self._collect_from_parsed(for_query.code, parsed)
        else:
            elements, aliases = self._collect_from_raw(for_query.code, insight)
        edata.add_key_value_pair_list(
            self.extracted_field_factory.create_key_value_pair_list(
                field_code=for_query.code,
                elements=elements,
                aliases=aliases,
            ),
        )

    def _collect_from_parsed(
        self, field_code: str, parsed: KeyValuePairsListWithAliasesResponse
    ) -> tuple[list, dict[EntityId, str]]:
        elements = []
        aliases: dict[EntityId, str] = {}
        for item in parsed.items:
            key_conf = self._compute_confidence(
                field_code=field_code,
                value_for_check=item.key,
                evidence=item.key_evidence,
                self_confidence=item.key_self_confidence,
                is_list=True,
            )
            value_conf = self._compute_confidence(
                field_code=field_code,
                value_for_check=item.value,
                evidence=item.value_evidence,
                self_confidence=item.value_self_confidence,
                is_list=True,
            )
            kvp = self.field_data_factory.create_key_value_pair(
                key=self.field_data_factory.create_string(value=item.key, coordinates=None, confidence=key_conf),
                value=self.field_data_factory.create_string(value=item.value, coordinates=None, confidence=value_conf),
            )
            elements.append(kvp)
            if item.alias and item.alias.strip():
                aliases[kvp.id] = item.alias
        return elements, aliases

    def _collect_from_raw(
        self, field_code: str, insight: LLMResponse[KeyValuePairsListWithAliasesResponse]
    ) -> tuple[list, dict[EntityId, str]]:
        if isinstance(insight.parsed, dict):
            return self._collect_from_raw_dict(field_code, insight.parsed.get("items", []))
        return self._collect_from_raw_fallback(field_code, insight)

    def _collect_from_raw_dict(self, field_code: str, raw_items: list) -> tuple[list, dict[EntityId, str]]:
        elements = []
        aliases: dict[EntityId, str] = {}
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            kvp = self.field_data_factory.create_key_value_pair(
                key=self.field_data_factory.create_string(
                    value=str(raw.get("key", "")),
                    coordinates=None,
                    confidence=self._compute_confidence(
                        field_code=field_code,
                        value_for_check=str(raw.get("key", "")),
                        evidence=self._ev_from_raw(raw.get("key_evidence")),
                        self_confidence=self._sc_from_raw(raw.get("key_self_confidence")),
                        is_list=True,
                    ),
                ),
                value=self.field_data_factory.create_string(
                    value=str(raw.get("value", "")),
                    coordinates=None,
                    confidence=self._compute_confidence(
                        field_code=field_code,
                        value_for_check=str(raw.get("value", "")),
                        evidence=self._ev_from_raw(raw.get("value_evidence")),
                        self_confidence=self._sc_from_raw(raw.get("value_self_confidence")),
                        is_list=True,
                    ),
                ),
            )
            elements.append(kvp)
            alias = str(raw.get("alias", ""))
            if alias and alias.strip():
                aliases[kvp.id] = alias
        return elements, aliases

    def _collect_from_raw_fallback(
        self, field_code: str, insight: LLMResponse[KeyValuePairsListWithAliasesResponse]
    ) -> tuple[list, dict[EntityId, str]]:
        elements = []
        aliases: dict[EntityId, str] = {}
        for item in KeyValuePairsListWithAliasesResponse.parse_llm_response(insight):
            kvp = self.field_data_factory.create_key_value_pair(
                key=self.field_data_factory.create_string(
                    value=item["key"],
                    coordinates=None,
                    confidence=self._compute_confidence(
                        field_code=field_code,
                        value_for_check=item["key"],
                        evidence=None,
                        self_confidence=None,
                        is_list=True,
                    ),
                ),
                value=self.field_data_factory.create_string(
                    value=item["value"],
                    coordinates=None,
                    confidence=self._compute_confidence(
                        field_code=field_code,
                        value_for_check=item["value"],
                        evidence=None,
                        self_confidence=None,
                        is_list=True,
                    ),
                ),
            )
            elements.append(kvp)
            alias = item["alias"]
            if alias and alias.strip():
                aliases[kvp.id] = alias
        return elements, aliases
