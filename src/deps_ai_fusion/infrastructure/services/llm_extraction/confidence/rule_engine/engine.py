from ..models import ConfidenceLevel, Evidence, ReasonCode
from .boolean_checks import BooleanRuleChecks
from .checks import RuleChecks
from .result import RuleEngineResult
from .schema import SchemaHint

__all__ = ["RuleEngine"]


class RuleEngine:
    EVIDENCE_MAX_LEN = 500

    def evaluate(
        self,
        value: str | bool | None,
        evidence: Evidence | None,
        self_confidence: ConfidenceLevel | None,
        is_list: bool = False,
        schema_hint: SchemaHint | None = None,
        label: str | None = None,
    ) -> RuleEngineResult:
        if schema_hint == SchemaHint.BOOLEAN:
            e_hard, e_soft = self._check_boolean_evidence_rules(value, evidence, label)
            s_hard: list[ReasonCode] = []
            s_soft: list[ReasonCode] = []
        else:
            e_hard, e_soft = self._check_evidence_rules(value, evidence, is_list)
            s_hard, s_soft = RuleChecks.check_schema_rule(value, schema_hint) if schema_hint is not None else ([], [])

        a_hard, a_soft = RuleChecks.check_artifact_rules(value, evidence)

        amb_soft: list[ReasonCode] = []
        if not RuleChecks.model_not_ambiguous(self_confidence):
            amb_soft.append(ReasonCode.MODEL_REPORTS_AMBIGUITY)

        hard_fails = e_hard + a_hard + s_hard
        soft_fails = e_soft + a_soft + amb_soft + s_soft
        return RuleChecks.build_result(hard_fails, soft_fails)

    def _check_boolean_evidence_rules(
        self,
        value: str | bool | None,
        evidence: Evidence | None,
        label: str | None = None,
    ) -> tuple[list[ReasonCode], list[ReasonCode]]:
        hard_fails: list[ReasonCode] = []
        soft_fails: list[ReasonCode] = []
        if evidence is None or not evidence.text.strip():
            hard_fails.append(ReasonCode.EVIDENCE_MISSING)
            return hard_fails, soft_fails
        if evidence.page < 1:
            hard_fails.append(ReasonCode.INVALID_EVIDENCE)
            return hard_fails, soft_fails
        b_hard, b_soft = BooleanRuleChecks.evaluate(value, evidence, label)
        hard_fails.extend(b_hard)
        soft_fails.extend(b_soft)
        return hard_fails, soft_fails

    def _check_evidence_rules(
        self,
        value: str | bool | None,
        evidence: Evidence | None,
        is_list: bool,
    ) -> tuple[list[ReasonCode], list[ReasonCode]]:
        hard_fails: list[ReasonCode] = []
        soft_fails: list[ReasonCode] = []
        if evidence is None or not evidence.text.strip():
            hard_fails.append(ReasonCode.EVIDENCE_MISSING)
            return hard_fails, soft_fails
        if evidence.page < 1:
            hard_fails.append(ReasonCode.INVALID_EVIDENCE)
            return hard_fails, soft_fails
        if not is_list and not RuleChecks.value_in_evidence(value, evidence):
            hard_fails.append(ReasonCode.VALUE_NOT_IN_EVIDENCE)
        if len(evidence.text) >= self.EVIDENCE_MAX_LEN:
            soft_fails.append(ReasonCode.EVIDENCE_TOO_BROAD)
        if not RuleChecks.no_ocr_noise(evidence):
            soft_fails.append(ReasonCode.OCR_NOISE_DETECTED)
        return hard_fails, soft_fails
