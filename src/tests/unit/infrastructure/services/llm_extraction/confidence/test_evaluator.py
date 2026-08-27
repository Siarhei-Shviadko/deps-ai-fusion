import logging
from unittest.mock import MagicMock, patch

import pytest

from deps_ai_fusion.infrastructure.services.llm_extraction.confidence.evaluator import (
    ConfidenceEvaluator,
)
from deps_ai_fusion.infrastructure.services.llm_extraction.confidence.models import (
    ConfidenceLevel,
    ConfidenceTrace,
    Evidence,
    ReasonCode,
)
from deps_ai_fusion.infrastructure.services.llm_extraction.confidence.rule_engine import (
    RuleEngineResult,
)


@pytest.fixture()
def evaluator() -> ConfidenceEvaluator:
    return ConfidenceEvaluator()


class TestEvaluateReturnTrace:
    def test_returns_confidence_trace(self, evaluator: ConfidenceEvaluator) -> None:
        evidence = Evidence(text="Total: 1234 USD", page=1)
        trace = evaluator.evaluate("total", "1234", evidence, ConfidenceLevel.HIGH)
        assert isinstance(trace, ConfidenceTrace)
        assert trace.field_code == "total"

    def test_high_evidence_high_self_confidence_gives_high(self, evaluator: ConfidenceEvaluator) -> None:
        evidence = Evidence(text="Total: 1234 USD", page=1)
        trace = evaluator.evaluate("total", "1234", evidence, ConfidenceLevel.HIGH)
        assert trace.c_final == ConfidenceLevel.HIGH

    def test_missing_evidence_gives_low(self, evaluator: ConfidenceEvaluator) -> None:
        trace = evaluator.evaluate("total", "1234", None, ConfidenceLevel.HIGH)
        assert trace.c_final == ConfidenceLevel.LOW
        assert ReasonCode.EVIDENCE_MISSING in trace.reason_codes

    def test_value_not_in_evidence_gives_low(self, evaluator: ConfidenceEvaluator) -> None:
        evidence = Evidence(text="completely different text", page=1)
        trace = evaluator.evaluate("total", "9999.99", evidence, ConfidenceLevel.HIGH)
        assert trace.c_final == ConfidenceLevel.LOW
        assert ReasonCode.VALUE_NOT_IN_EVIDENCE in trace.reason_codes

    def test_value_preview_truncated_to_120(self, evaluator: ConfidenceEvaluator) -> None:
        long_value = "x" * 200
        evidence = Evidence(text="x " * 100, page=1)
        trace = evaluator.evaluate("f", long_value, evidence, None)
        assert len(trace.value_preview) <= 120

    def test_evidence_text_preview_truncated_to_120(self, evaluator: ConfidenceEvaluator) -> None:
        long_text = "word " * 100
        evidence = Evidence(text=long_text, page=1)
        trace = evaluator.evaluate("f", "word", evidence, None)
        assert len(trace.evidence_text_preview) <= 120

    def test_no_evidence_gives_empty_evidence_preview(self, evaluator: ConfidenceEvaluator) -> None:
        trace = evaluator.evaluate("f", "val", None, None)
        assert trace.evidence_text_preview == ""

    def test_aggregation_rule_id_set(self, evaluator: ConfidenceEvaluator) -> None:
        evidence = Evidence(text="val here", page=1)
        trace = evaluator.evaluate("f", "val", evidence, ConfidenceLevel.HIGH)
        assert trace.aggregation_rule_id != ""

    def test_as_float_high(self, evaluator: ConfidenceEvaluator) -> None:
        evidence = Evidence(text="Total: 1234 USD", page=1)
        trace = evaluator.evaluate("total", "1234", evidence, ConfidenceLevel.HIGH)
        if trace.c_final == ConfidenceLevel.HIGH:
            assert trace.as_float() == 0.95

    def test_as_float_low(self, evaluator: ConfidenceEvaluator) -> None:
        trace = evaluator.evaluate("total", "1234", None, None)
        assert trace.c_final == ConfidenceLevel.LOW
        assert trace.as_float() == 0.1


class TestLogging:
    def test_debug_log_emitted(self, evaluator: ConfidenceEvaluator) -> None:
        with patch.object(ConfidenceEvaluator, "_logger") as mock_logger:
            evidence = Evidence(text="Total: 1234", page=1)
            evaluator.evaluate("total", "1234", evidence, ConfidenceLevel.HIGH)
            assert mock_logger.debug.call_count == 2
            first_call = mock_logger.debug.call_args_list[0]
            assert "confidence_trace" in first_call[0] or "confidence_trace" in str(first_call)

    def test_log_contains_field_code(self, evaluator: ConfidenceEvaluator) -> None:
        with patch.object(ConfidenceEvaluator, "_logger") as mock_logger:
            evidence = Evidence(text="Total: 1234", page=1)
            evaluator.evaluate("my_field", "1234", evidence, ConfidenceLevel.HIGH)
            extra = mock_logger.debug.call_args_list[0][1]["extra"]
            assert extra["confidence"]["field_code"] == "my_field"
