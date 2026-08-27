import pytest

from deps_ai_fusion.infrastructure.services.llm_extraction.confidence.models import (
    ConfidenceLevel,
    Evidence,
    ReasonCode,
)
from deps_ai_fusion.infrastructure.services.llm_extraction.confidence.rule_engine import (
    RuleEngine,
    SchemaHint,
)


@pytest.fixture()
def engine() -> RuleEngine:
    return RuleEngine()


def _evidence(text: str = "Invoice total: 1234.56 USD", page: int = 1) -> Evidence:
    return Evidence(text=text, page=page)


class TestR0EvidenceValid:
    def test_missing_evidence_is_hard_fail(self, engine: RuleEngine) -> None:
        result = engine.evaluate("1234.56", None, None)
        assert result.c_rules == ConfidenceLevel.LOW
        assert ReasonCode.EVIDENCE_MISSING in result.reason_codes
        assert result.c_rules == ConfidenceLevel.LOW

    def test_empty_text_is_hard_fail(self, engine: RuleEngine) -> None:
        result = engine.evaluate("1234.56", Evidence(text="", page=1), None)
        assert result.c_rules == ConfidenceLevel.LOW
        assert ReasonCode.EVIDENCE_MISSING in result.reason_codes

    def test_invalid_page_is_hard_fail(self, engine: RuleEngine) -> None:
        result = engine.evaluate("1234.56", Evidence(text="some text", page=0), None)
        assert result.c_rules == ConfidenceLevel.LOW
        assert ReasonCode.INVALID_EVIDENCE in result.reason_codes

    def test_whitespace_only_text_is_hard_fail(self, engine: RuleEngine) -> None:
        result = engine.evaluate("1234.56", Evidence(text="   ", page=1), None)
        assert result.c_rules == ConfidenceLevel.LOW


class TestR1ValueInEvidence:
    def test_exact_match_passes(self, engine: RuleEngine) -> None:
        result = engine.evaluate("1234", _evidence("Invoice total: 1234 USD"), None)
        assert result.c_rules != ConfidenceLevel.LOW
        assert ReasonCode.VALUE_NOT_IN_EVIDENCE not in result.reason_codes

    def test_normalized_match_passes(self, engine: RuleEngine) -> None:
        result = engine.evaluate("Oak-Park", _evidence("Property: Oak Park, CA"), None)
        assert result.c_rules != ConfidenceLevel.LOW
        assert ReasonCode.VALUE_NOT_IN_EVIDENCE not in result.reason_codes

    def test_case_insensitive_match_passes(self, engine: RuleEngine) -> None:
        result = engine.evaluate("TOTAL", _evidence("Total: 500.00"), None)
        assert result.c_rules != ConfidenceLevel.LOW

    def test_value_absent_is_hard_fail(self, engine: RuleEngine) -> None:
        result = engine.evaluate("9999.99", _evidence("Invoice total: 1234.56 USD"), None)
        assert result.c_rules == ConfidenceLevel.LOW
        assert ReasonCode.VALUE_NOT_IN_EVIDENCE in result.reason_codes

    def test_none_value_passes_r1(self, engine: RuleEngine) -> None:
        result = engine.evaluate(None, _evidence("some text"), None)
        assert ReasonCode.VALUE_NOT_IN_EVIDENCE not in result.reason_codes

    def test_r1_skipped_for_lists(self, engine: RuleEngine) -> None:
        result = engine.evaluate("absent_value", _evidence("unrelated text"), None, is_list=True)
        assert ReasonCode.VALUE_NOT_IN_EVIDENCE not in result.reason_codes


class TestR2EvidenceNotTooBroad:
    def test_short_evidence_passes(self, engine: RuleEngine) -> None:
        result = engine.evaluate("123", _evidence("Total: 123"), None)
        assert ReasonCode.EVIDENCE_TOO_BROAD not in result.reason_codes

    def test_long_evidence_is_soft_fail(self, engine: RuleEngine) -> None:
        long_text = "x " * 300
        result = engine.evaluate("x", _evidence(long_text), None)
        assert ReasonCode.EVIDENCE_TOO_BROAD in result.reason_codes
        assert result.c_rules != ConfidenceLevel.LOW
        assert result.c_rules == ConfidenceLevel.MEDIUM


class TestR5ModelAmbiguity:
    def test_medium_self_confidence_adds_ambiguity_code(self, engine: RuleEngine) -> None:
        result = engine.evaluate("val", _evidence("val present"), ConfidenceLevel.MEDIUM)
        assert ReasonCode.MODEL_REPORTS_AMBIGUITY in result.reason_codes
        assert result.c_rules != ConfidenceLevel.LOW

    def test_high_self_confidence_no_ambiguity(self, engine: RuleEngine) -> None:
        result = engine.evaluate("val", _evidence("val present"), ConfidenceLevel.HIGH)
        assert ReasonCode.MODEL_REPORTS_AMBIGUITY not in result.reason_codes

    def test_low_self_confidence_no_ambiguity_code(self, engine: RuleEngine) -> None:
        result = engine.evaluate("val", _evidence("val present"), ConfidenceLevel.LOW)
        assert ReasonCode.MODEL_REPORTS_AMBIGUITY not in result.reason_codes


class TestR8ForbiddenArtifacts:
    @pytest.mark.parametrize("bad_value", ["N/A", "n/a", "unknown", "not found", "cannot find", "unclear"])
    def test_forbidden_values_are_hard_fail(self, engine: RuleEngine, bad_value: str) -> None:
        result = engine.evaluate(bad_value, _evidence("some text"), None)
        assert result.c_rules == ConfidenceLevel.LOW
        assert ReasonCode.FORBIDDEN_ARTIFACT in result.reason_codes

    def test_none_value_passes_r8(self, engine: RuleEngine) -> None:
        result = engine.evaluate(None, _evidence("some text"), None)
        assert ReasonCode.FORBIDDEN_ARTIFACT not in result.reason_codes

    def test_valid_value_passes(self, engine: RuleEngine) -> None:
        result = engine.evaluate("1234.56", _evidence("Total 1234.56"), None)
        assert ReasonCode.FORBIDDEN_ARTIFACT not in result.reason_codes


class TestR9OcrNoise:
    def test_clean_text_passes(self, engine: RuleEngine) -> None:
        result = engine.evaluate("val", _evidence("Clean readable text, page 2."), None)
        assert ReasonCode.OCR_NOISE_DETECTED not in result.reason_codes

    def test_noisy_text_is_soft_fail(self, engine: RuleEngine) -> None:
        result = engine.evaluate("val", _evidence("val ###@@@~~~ noise"), None)
        assert ReasonCode.OCR_NOISE_DETECTED in result.reason_codes
        assert result.c_rules != ConfidenceLevel.LOW


class TestR8ForbiddenInEvidence:
    def test_forbidden_in_evidence_is_soft_fail(self, engine: RuleEngine) -> None:
        result = engine.evaluate("valid", _evidence("valid amount is unknown"), None)
        assert ReasonCode.FORBIDDEN_ARTIFACT in result.reason_codes
        assert result.c_rules != ConfidenceLevel.LOW

    def test_forbidden_in_value_is_hard_fail(self, engine: RuleEngine) -> None:
        result = engine.evaluate("n/a", _evidence("N/A amount"), None)
        assert result.c_rules == ConfidenceLevel.LOW
        assert ReasonCode.FORBIDDEN_ARTIFACT in result.reason_codes

    def test_forbidden_in_value_suppresses_evidence_check(self, engine: RuleEngine) -> None:
        result = engine.evaluate("unknown", _evidence("unknown value unknown"), None)
        assert result.c_rules == ConfidenceLevel.LOW
        assert result.reason_codes.count(ReasonCode.FORBIDDEN_ARTIFACT) == 1

    def test_clean_evidence_passes(self, engine: RuleEngine) -> None:
        result = engine.evaluate("valid", _evidence("Total: 500.00"), None)
        assert ReasonCode.FORBIDDEN_ARTIFACT not in result.reason_codes

    def test_no_evidence_skips_evidence_check(self, engine: RuleEngine) -> None:
        result = engine.evaluate("valid", None, None)
        assert ReasonCode.FORBIDDEN_ARTIFACT not in result.reason_codes


class TestSchemaValidation:
    def test_valid_iso_date_passes(self, engine: RuleEngine) -> None:
        result = engine.evaluate("2024-01-15", _evidence("Invoice date: 2024-01-15"), None, schema_hint=SchemaHint.DATE)
        assert ReasonCode.SCHEMA_FAIL not in result.reason_codes
        assert ReasonCode.FORMAT_FAIL not in result.reason_codes

    def test_non_iso_date_is_format_fail(self, engine: RuleEngine) -> None:
        result = engine.evaluate("15/01/2024", _evidence("Invoice date 15/01/2024"), None, schema_hint=SchemaHint.DATE)
        assert ReasonCode.FORMAT_FAIL in result.reason_codes
        assert result.c_rules != ConfidenceLevel.LOW

    def test_non_date_value_is_schema_fail(self, engine: RuleEngine) -> None:
        result = engine.evaluate("not a date", _evidence("Invoice date: not a date"), None, schema_hint=SchemaHint.DATE)
        assert result.c_rules == ConfidenceLevel.LOW
        assert ReasonCode.SCHEMA_FAIL in result.reason_codes

    def test_valid_amount_passes(self, engine: RuleEngine) -> None:
        result = engine.evaluate("1234.56", _evidence("Total: 1234.56"), None, schema_hint=SchemaHint.AMOUNT)
        assert ReasonCode.SCHEMA_FAIL not in result.reason_codes
        assert ReasonCode.FORMAT_FAIL not in result.reason_codes

    def test_non_numeric_amount_is_schema_fail(self, engine: RuleEngine) -> None:
        result = engine.evaluate("abc", _evidence("amount abc"), None, schema_hint=SchemaHint.AMOUNT)
        assert result.c_rules == ConfidenceLevel.LOW
        assert ReasonCode.SCHEMA_FAIL in result.reason_codes

    def test_valid_id_passes(self, engine: RuleEngine) -> None:
        result = engine.evaluate("INV-2024-001", _evidence("Invoice INV-2024-001"), None, schema_hint=SchemaHint.ID)
        assert ReasonCode.SCHEMA_FAIL not in result.reason_codes
        assert ReasonCode.FORMAT_FAIL not in result.reason_codes

    def test_no_schema_hint_skips_validation(self, engine: RuleEngine) -> None:
        result = engine.evaluate("anything goes", _evidence("anything goes"), None)
        assert ReasonCode.SCHEMA_FAIL not in result.reason_codes
        assert ReasonCode.FORMAT_FAIL not in result.reason_codes

    def test_none_value_passes_schema_validation(self, engine: RuleEngine) -> None:
        result = engine.evaluate(None, _evidence("some text"), None, schema_hint=SchemaHint.DATE)
        assert ReasonCode.SCHEMA_FAIL not in result.reason_codes
        assert ReasonCode.FORMAT_FAIL not in result.reason_codes


class TestCombinedOutcomes:
    def test_all_pass_gives_high(self, engine: RuleEngine) -> None:
        result = engine.evaluate("1234", _evidence("Total: 1234 USD"), ConfidenceLevel.HIGH)
        assert result.c_rules == ConfidenceLevel.HIGH
        assert result.c_rules != ConfidenceLevel.LOW
        assert result.reason_codes == []

    def test_hard_fail_overrides_soft_signals(self, engine: RuleEngine) -> None:
        result = engine.evaluate("missing_val", _evidence("x " * 300), ConfidenceLevel.MEDIUM)
        assert result.c_rules == ConfidenceLevel.LOW
        assert result.c_rules == ConfidenceLevel.LOW

    def test_only_soft_fails_give_medium(self, engine: RuleEngine) -> None:
        result = engine.evaluate("x", _evidence("x " * 300), ConfidenceLevel.MEDIUM)
        assert result.c_rules != ConfidenceLevel.LOW
        assert result.c_rules == ConfidenceLevel.MEDIUM
