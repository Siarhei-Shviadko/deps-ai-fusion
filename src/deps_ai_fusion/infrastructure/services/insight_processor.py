from typing import Any

from deps_gen_ai.common import LLMResponse

from deps_ai_fusion.infrastructure.services.llm_extraction.confidence import (
    ConfidenceEvaluator,
    ConfidenceLevel,
    ConfidenceTrace,
    Evidence,
)

from .insight_parser import InsightParser
from .processed_insight import ProcessedInsight, ProcessedValueItem

__all__ = ["InsightProcessor"]


class InsightProcessor:
    def __init__(self) -> None:
        self._evaluator = ConfidenceEvaluator()
        self._parser = InsightParser()

    def extract_content(self, response: LLMResponse[Any]) -> str:
        return self._parser.extract_content(response)

    def process(self, response: LLMResponse[Any], field_code: str) -> ProcessedInsight:
        return self._process_content(
            content=self._parser.extract_content(response),
            field_code=field_code,
            error_occurred=not response.success,
            fallback_confidence=response.confidence,
        )

    def process_stored(self, response: str, field_code: str, fallback_confidence: float | None) -> ProcessedInsight:
        return self._process_content(
            content=response,
            field_code=field_code,
            error_occurred=False,
            fallback_confidence=fallback_confidence,
        )

    def _process_content(
        self, content: str, field_code: str, error_occurred: bool, fallback_confidence: float | None
    ) -> ProcessedInsight:
        item = self._parser.parse_item(content)
        display_content = self._parser.content_from(item, content)
        if item is None:
            return ProcessedInsight(
                content=display_content,
                error_occurred=error_occurred,
                confidence=fallback_confidence,
                evidence=None,
                confidence_trace=None,
            )
        values = item.get("values")
        if isinstance(values, list):
            return self._build_list(values, field_code, display_content, error_occurred)
        if "key" in item:
            return self._build_kv(item, field_code, display_content, error_occurred)
        return self._build_scalar(item, field_code, display_content, error_occurred)

    def _evaluate(
        self,
        field_code: str,
        value: str,
        evidence_data: dict | None,
        sc_raw: str | None,
        is_list: bool = False,
    ) -> tuple[float, Evidence | None, ConfidenceTrace]:
        evidence = Evidence(**evidence_data) if isinstance(evidence_data, dict) else None
        sc = ConfidenceLevel(sc_raw) if sc_raw in {c.value for c in ConfidenceLevel} else None
        trace = self._evaluator.evaluate(
            field_code=field_code, value=value, evidence=evidence, self_confidence=sc, is_list=is_list
        )
        return trace.as_float(), evidence, trace

    def _build_scalar(self, item: dict, field_code: str, content: str, error_occurred: bool) -> ProcessedInsight:
        conf, ev, trace = self._evaluate(
            field_code=field_code,
            value=str(item.get("value", "")),
            evidence_data=item.get("evidence"),
            sc_raw=item.get("self_confidence"),
        )
        return ProcessedInsight(
            content=content,
            error_occurred=error_occurred,
            confidence=conf,
            evidence=ev,
            confidence_trace=trace,
        )

    def _build_kv(self, item: dict, field_code: str, content: str, error_occurred: bool) -> ProcessedInsight:
        k_conf, k_ev, k_trace = self._evaluate(
            field_code=field_code,
            value=str(item.get("key", "")),
            evidence_data=item.get("key_evidence"),
            sc_raw=item.get("key_self_confidence"),
        )
        v_conf, v_ev, v_trace = self._evaluate(
            field_code=field_code,
            value=str(item.get("value", "")),
            evidence_data=item.get("value_evidence"),
            sc_raw=item.get("value_self_confidence"),
        )
        confs = [c for c in (k_conf, v_conf) if c is not None]
        return ProcessedInsight(
            content=content,
            error_occurred=error_occurred,
            confidence=min(confs) if confs else None,
            evidence=None,
            confidence_trace=None,
            key_confidence=k_conf,
            key_evidence=k_ev,
            key_confidence_trace=k_trace,
            value_confidence=v_conf,
            value_evidence=v_ev,
            value_confidence_trace=v_trace,
        )

    def _build_value_item(self, raw: dict, field_code: str) -> ProcessedValueItem:
        if "key" in raw:
            k_conf, k_ev, k_trace = self._evaluate(
                field_code=field_code,
                value=str(raw.get("key", "")),
                evidence_data=raw.get("key_evidence"),
                sc_raw=raw.get("key_self_confidence"),
                is_list=True,
            )
            v_conf, v_ev, v_trace = self._evaluate(
                field_code=field_code,
                value=str(raw.get("value", "")),
                evidence_data=raw.get("value_evidence"),
                sc_raw=raw.get("value_self_confidence"),
                is_list=True,
            )
            confs = [c for c in (k_conf, v_conf) if c is not None]
            return ProcessedValueItem(
                value=str(raw.get("value", "")),
                confidence=min(confs) if confs else None,
                evidence=None,
                confidence_trace=None,
                key=str(raw.get("key", "")),
                key_confidence=k_conf,
                key_evidence=k_ev,
                key_confidence_trace=k_trace,
                value_confidence=v_conf,
                value_evidence=v_ev,
                value_confidence_trace=v_trace,
            )
        conf, ev, trace = self._evaluate(
            field_code=field_code,
            value=str(raw.get("value", "")),
            evidence_data=raw.get("evidence"),
            sc_raw=raw.get("self_confidence"),
            is_list=True,
        )
        return ProcessedValueItem(
            value=str(raw.get("value", "")),
            confidence=conf,
            evidence=ev,
            confidence_trace=trace,
        )

    def _build_list(self, raw_values: list, field_code: str, content: str, error_occurred: bool) -> ProcessedInsight:
        items: list[ProcessedValueItem] = []
        confidences: list[float] = []
        for raw in raw_values:
            if not isinstance(raw, dict):
                continue
            item = self._build_value_item(raw, field_code)
            items.append(item)
            if item.confidence is not None:
                confidences.append(item.confidence)
        return ProcessedInsight(
            content=content,
            error_occurred=error_occurred,
            confidence=min(confidences) if confidences else None,
            evidence=None,
            confidence_trace=None,
            values=items or None,
        )
