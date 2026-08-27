from ..models import Evidence, ReasonCode
from .boolean_markers import BooleanMarkers

__all__ = ["BooleanRuleChecks"]


class BooleanRuleChecks:
    @staticmethod
    def evaluate(
        value: str | bool | None,
        evidence: Evidence | None,
        label: str | None = None,
    ) -> tuple[list[ReasonCode], list[ReasonCode]]:
        hard_fails: list[ReasonCode] = []
        soft_fails: list[ReasonCode] = []
        if evidence is None:
            return hard_fails, soft_fails
        true_markers = BooleanMarkers.find_true_markers(evidence.text)
        false_markers = BooleanMarkers.find_false_markers(evidence.text)
        consistency_fail = BooleanRuleChecks._consistency_fail(value, true_markers, false_markers)
        if consistency_fail is not None:
            hard_fails.append(consistency_fail)
        if not true_markers and not false_markers:
            return hard_fails, soft_fails
        if len(true_markers) + len(false_markers) > 1:
            soft_fails.append(ReasonCode.MULTIPLE_CANDIDATES)
        if label is not None and label.strip().lower() not in evidence.text.lower():
            soft_fails.append(ReasonCode.LABEL_NOT_FOUND_NEAR_MARKER)
        return hard_fails, soft_fails

    @staticmethod
    def _consistency_fail(
        value: str | bool | None,
        true_markers: list[str],
        false_markers: list[str],
    ) -> ReasonCode | None:
        if not true_markers and not false_markers:
            return ReasonCode.BOOLEAN_MARKER_CONFLICT if value is True else None
        if true_markers and false_markers:
            return ReasonCode.BOOLEAN_MARKER_AMBIGUOUS
        if isinstance(value, bool) and bool(true_markers) != value:
            return ReasonCode.BOOLEAN_MARKER_CONFLICT
        return None
