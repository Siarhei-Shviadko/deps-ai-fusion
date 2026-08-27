from deps_extracted_data.model import (
    CheckboxValue,
    EntityId,
    ExtractedData,
    GenericData,
)
from deps_gen_ai.common import LLMResponse

from deps_ai_fusion.domain.model import Query

from ...confidence.rule_engine import SchemaHint
from ...genai_query_factory import BooleansListWithAliasesResponse
from .abstract import AbstractInsightsRecorder

__all__ = ["BooleanListWithAliasesInsightsRecorder"]

_BoolElements = list[GenericData]


class BooleanListWithAliasesInsightsRecorder(AbstractInsightsRecorder[BooleansListWithAliasesResponse]):
    def record_insights(
        self,
        edata: ExtractedData,
        for_query: Query,
        insight: LLMResponse[BooleansListWithAliasesResponse],
    ) -> None:
        parsed = insight.parsed if isinstance(insight.parsed, BooleansListWithAliasesResponse) else None
        if parsed:
            elements, aliases = self._collect_from_parsed(for_query.code, parsed)
        else:
            elements, aliases = self._collect_from_raw(for_query.code, insight)
        edata.add_checkbox_list(
            self.extracted_field_factory.create_checkbox_list(
                field_code=for_query.code,
                elements=elements,
                aliases=aliases,
            ),
        )

    def _collect_from_parsed(
        self,
        field_code: str,
        parsed: BooleansListWithAliasesResponse,
    ) -> tuple[_BoolElements, dict[EntityId, str]]:
        elements: _BoolElements = []
        aliases: dict[EntityId, str] = {}
        for item in parsed.items:
            element_data = self.field_data_factory.create_checkbox(
                value=CheckboxValue.create(item.value),
                confidence=self._compute_confidence(
                    field_code=field_code,
                    value_for_check=item.value,
                    evidence=item.evidence,
                    self_confidence=item.self_confidence,
                    is_list=True,
                    schema_hint=SchemaHint.BOOLEAN,
                    label=item.alias if item.alias and item.alias.strip() else None,
                ),
                coordinates=None,
            )
            elements.append(element_data)
            if item.alias and item.alias.strip():
                aliases[element_data.id] = item.alias
        return elements, aliases

    def _collect_from_raw(
        self,
        field_code: str,
        insight: LLMResponse[BooleansListWithAliasesResponse],
    ) -> tuple[_BoolElements, dict[EntityId, str]]:
        if isinstance(insight.parsed, dict):
            return self._collect_from_raw_dict(field_code, insight.parsed.get("items", []))
        return self._collect_from_raw_fallback(field_code, insight)

    def _collect_from_raw_dict(
        self,
        field_code: str,
        raw_items: list,
    ) -> tuple[_BoolElements, dict[EntityId, str]]:
        elements: _BoolElements = []
        aliases: dict[EntityId, str] = {}
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            element_data, alias = self._build_checkbox_from_raw(field_code, raw)
            elements.append(element_data)
            if alias:
                aliases[element_data.id] = alias
        return elements, aliases

    def _build_checkbox_from_raw(self, field_code: str, raw: dict) -> tuple[GenericData, str]:
        value = raw.get("value")
        alias = self._truncate_alias(str(raw.get("alias", "")))
        element_data = self.field_data_factory.create_checkbox(
            value=CheckboxValue.create(value),
            confidence=self._compute_confidence(
                field_code=field_code,
                value_for_check=value,
                evidence=self._ev_from_raw(raw.get("evidence")),
                self_confidence=self._sc_from_raw(raw.get("self_confidence")),
                is_list=True,
                schema_hint=SchemaHint.BOOLEAN,
                label=alias.strip() or None,
            ),
            coordinates=None,
        )
        return element_data, alias.strip()

    def _collect_from_raw_fallback(
        self,
        field_code: str,
        insight: LLMResponse[BooleansListWithAliasesResponse],
    ) -> tuple[_BoolElements, dict[EntityId, str]]:
        elements: _BoolElements = []
        aliases: dict[EntityId, str] = {}
        for element in BooleansListWithAliasesResponse.parse_llm_response(insight):
            value = element["value"]
            alias = self._truncate_alias(element["alias"])
            element_data = self.field_data_factory.create_checkbox(
                value=CheckboxValue.create(value),
                confidence=self._compute_confidence(
                    field_code=field_code,
                    value_for_check=value,
                    evidence=None,
                    self_confidence=None,
                    is_list=True,
                ),
                coordinates=None,
            )
            elements.append(element_data)
            if alias and alias.strip():
                aliases[element_data.id] = alias
        return elements, aliases
