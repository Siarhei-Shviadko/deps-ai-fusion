import dataclasses
import types
from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

__all__ = ["ConfidenceLevel", "Evidence", "ReasonCode", "ConfidenceTrace"]


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    page: int


class ReasonCode(str, Enum):
    EVIDENCE_MISSING = "EVIDENCE_MISSING"
    INVALID_EVIDENCE = "INVALID_EVIDENCE"
    VALUE_NOT_IN_EVIDENCE = "VALUE_NOT_IN_EVIDENCE"
    EVIDENCE_TOO_BROAD = "EVIDENCE_TOO_BROAD"
    FORBIDDEN_ARTIFACT = "FORBIDDEN_ARTIFACT"
    MODEL_REPORTS_AMBIGUITY = "MODEL_REPORTS_AMBIGUITY"
    OCR_NOISE_DETECTED = "OCR_NOISE_DETECTED"
    SCHEMA_FAIL = "SCHEMA_FAIL"
    FORMAT_FAIL = "FORMAT_FAIL"
    MULTIPLE_CANDIDATES = "MULTIPLE_CANDIDATES"
    CROSS_FIELD_CONFLICT = "CROSS_FIELD_CONFLICT"
    GRADER_LOW_CONFIDENCE = "GRADER_LOW_CONFIDENCE"
    BOOLEAN_MARKER_NOT_FOUND = "BOOLEAN_MARKER_NOT_FOUND"
    BOOLEAN_MARKER_CONFLICT = "BOOLEAN_MARKER_CONFLICT"
    BOOLEAN_MARKER_AMBIGUOUS = "BOOLEAN_MARKER_AMBIGUOUS"
    LABEL_NOT_FOUND_NEAR_MARKER = "LABEL_NOT_FOUND_NEAR_MARKER"


@dataclasses.dataclass
class ConfidenceTrace:
    _FLOAT_MAP: ClassVar = types.MappingProxyType(
        {
            "HIGH": 0.95,
            "MEDIUM": 0.85,
            "LOW": 0.1,
        }
    )
    field_code: str
    value_preview: str
    evidence_text_preview: str
    self_confidence: ConfidenceLevel | None
    c_rules: ConfidenceLevel
    reason_codes: list[ReasonCode]
    c_final: ConfidenceLevel
    aggregation_rule_id: str

    def as_float(self) -> float:
        return self._FLOAT_MAP[self.c_final.value]
