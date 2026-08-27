import re
from typing import ClassVar

from ..models import ConfidenceLevel, Evidence, ReasonCode
from .boolean_markers import BooleanMarkers
from .result import RuleEngineResult
from .schema import _SCHEMA_VALIDATORS, SchemaHint

__all__ = ["RuleChecks"]

_STRIP_RE: re.Pattern = re.compile(r"[^a-zA-Z0-9\s]")

_FORBIDDEN: frozenset[str] = frozenset(
    (
        "n/a",
        "na",
        "unknown",
        "i think",
        "probably",
        "cannot find",
        "not found",
        "unclear",
        "not available",
        "not applicable",
    )
)


def _normalize(value: str) -> str:
    return " ".join(_STRIP_RE.sub("", value.lower().replace("-", " ")).split())


_FORBIDDEN_NORMALIZED: frozenset[str] = frozenset(_normalize(f) for f in _FORBIDDEN)
_FORBIDDEN_SINGLE_WORDS: frozenset[str] = frozenset(f for f in _FORBIDDEN_NORMALIZED if " " not in f)
_FORBIDDEN_PHRASES: frozenset[str] = frozenset(f for f in _FORBIDDEN_NORMALIZED if " " in f)


class RuleChecks:
    _OCR_NOISE_RE: ClassVar[re.Pattern] = re.compile(r"[^\w\s,.!?;:()\-/]{3,}")

    @staticmethod
    def evidence_valid(evidence: Evidence | None) -> bool:
        return evidence is not None and bool(evidence.text.strip()) and evidence.page >= 1

    @staticmethod
    def value_in_evidence(value: str | bool | None, evidence: Evidence) -> bool:
        if value is None:
            return True
        normalized_value = _normalize(str(value))
        if not normalized_value:
            return True
        normalized_evidence = _normalize(evidence.text)
        literal_match = (
            normalized_value in set(normalized_evidence.split())
            if " " not in normalized_value
            else normalized_value in normalized_evidence
        )
        if normalized_value == "true":
            boolean_match = bool(BooleanMarkers.find_true_markers(evidence.text))
        elif normalized_value == "false":
            boolean_match = bool(BooleanMarkers.find_false_markers(evidence.text))
        else:
            boolean_match = False
        return literal_match or boolean_match

    @staticmethod
    def model_not_ambiguous(self_confidence: ConfidenceLevel | None) -> bool:
        return self_confidence != ConfidenceLevel.MEDIUM

    @staticmethod
    def value_clean(value: str | bool | None) -> bool:
        if value is None:
            return True
        normalized = _normalize(str(value))
        words = set(normalized.split())
        return (
            normalized not in _FORBIDDEN_NORMALIZED
            and not words.intersection(_FORBIDDEN_SINGLE_WORDS)
            and not any(f in normalized for f in _FORBIDDEN_PHRASES)
        )

    @staticmethod
    def evidence_clean(evidence: Evidence) -> bool:
        normalized = _normalize(evidence.text)
        words = set(normalized.split())
        return not words.intersection(_FORBIDDEN_SINGLE_WORDS) and not any(f in normalized for f in _FORBIDDEN_PHRASES)

    @staticmethod
    def no_ocr_noise(evidence: Evidence) -> bool:
        return not bool(RuleChecks._OCR_NOISE_RE.search(evidence.text))

    @staticmethod
    def schema_check(value: str | bool | None, schema_hint: SchemaHint) -> str | None:
        if value is None:
            return None
        str_value = str(value).strip()
        if not str_value:
            return "hard"
        strict_re, loose_re = _SCHEMA_VALIDATORS[schema_hint]
        if strict_re.search(str_value):
            return None
        if loose_re.search(str_value):
            return "soft"
        return "hard"

    @staticmethod
    def check_artifact_rules(
        value: str | bool | None,
        evidence: Evidence | None,
    ) -> tuple[list[ReasonCode], list[ReasonCode]]:
        hard_fails: list[ReasonCode] = []
        soft_fails: list[ReasonCode] = []
        if not RuleChecks.value_clean(value):
            hard_fails.append(ReasonCode.FORBIDDEN_ARTIFACT)
        elif evidence is not None and RuleChecks.evidence_valid(evidence) and not RuleChecks.evidence_clean(evidence):
            soft_fails.append(ReasonCode.FORBIDDEN_ARTIFACT)
        return hard_fails, soft_fails

    @staticmethod
    def check_schema_rule(
        value: str | bool | None,
        schema_hint: SchemaHint,
    ) -> tuple[list[ReasonCode], list[ReasonCode]]:
        hard_fails: list[ReasonCode] = []
        soft_fails: list[ReasonCode] = []
        outcome = RuleChecks.schema_check(value, schema_hint)
        if outcome == "hard":
            hard_fails.append(ReasonCode.SCHEMA_FAIL)
        elif outcome == "soft":
            soft_fails.append(ReasonCode.FORMAT_FAIL)
        return hard_fails, soft_fails

    @staticmethod
    def build_result(
        hard_fails: list[ReasonCode],
        soft_fails: list[ReasonCode],
    ) -> RuleEngineResult:
        reason_codes = hard_fails + soft_fails
        if hard_fails:
            return RuleEngineResult(c_rules=ConfidenceLevel.LOW, reason_codes=reason_codes)
        c_rules = ConfidenceLevel.MEDIUM if soft_fails else ConfidenceLevel.HIGH
        return RuleEngineResult(c_rules=c_rules, reason_codes=reason_codes)
