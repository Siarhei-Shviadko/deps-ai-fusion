import dataclasses

from ..models import ConfidenceLevel, ReasonCode

__all__ = ["RuleEngineResult"]


@dataclasses.dataclass
class RuleEngineResult:
    c_rules: ConfidenceLevel
    reason_codes: list[ReasonCode]
