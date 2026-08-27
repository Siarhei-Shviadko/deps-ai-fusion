import dataclasses
import logging

from .aggregator import ConfidenceAggregator
from .models import ConfidenceLevel, ConfidenceTrace, Evidence
from .rule_engine import RuleEngine, SchemaHint

__all__ = ["ConfidenceEvaluator"]

VALUE_PREVIEW_LENGTH = 100


class ConfidenceEvaluator:
    _logger = logging.getLogger(__qualname__)

    def __init__(self) -> None:
        self._rule_engine = RuleEngine()
        self._aggregator = ConfidenceAggregator()

    def evaluate(
        self,
        field_code: str,
        value: str | bool | None,
        evidence: Evidence | None,
        self_confidence: ConfidenceLevel | None,
        is_list: bool = False,
        schema_hint: SchemaHint | None = None,
        label: str | None = None,
    ) -> ConfidenceTrace:
        rule_result = self._rule_engine.evaluate(value, evidence, self_confidence, is_list, schema_hint, label)
        c_final, rule_id = self._aggregator.aggregate(rule_result.c_rules, self_confidence)
        trace = ConfidenceTrace(
            field_code=field_code,
            value_preview=str(value)[:VALUE_PREVIEW_LENGTH],
            evidence_text_preview=evidence.text[:VALUE_PREVIEW_LENGTH] if evidence else "",
            self_confidence=self_confidence,
            c_rules=rule_result.c_rules,
            reason_codes=rule_result.reason_codes,
            c_final=c_final,
            aggregation_rule_id=rule_id,
        )
        self._logger.debug("confidence_trace", extra={"confidence": dataclasses.asdict(trace)})
        self._logger.debug(
            "reason_codes field=%s codes=%s",
            field_code,
            [rc.value for rc in trace.reason_codes],
        )
        return trace
