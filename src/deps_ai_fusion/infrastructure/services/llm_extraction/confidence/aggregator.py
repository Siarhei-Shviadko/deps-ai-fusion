import types

from .models import ConfidenceLevel

__all__ = ["ConfidenceAggregator"]


class ConfidenceAggregator:
    _H = ConfidenceLevel.HIGH
    _M = ConfidenceLevel.MEDIUM
    _L = ConfidenceLevel.LOW
    _AGGREGATION_TABLE = types.MappingProxyType(
        {
            (_H, _H): (_H, "HIGH_x_HIGH"),
            (_H, _M): (_H, "HIGH_x_MED"),
            (_H, _L): (_M, "HIGH_x_LOW_disagree"),
            (_M, _H): (_H, "MED_x_HIGH"),
            (_M, _M): (_M, "MED_x_MED"),
            (_M, _L): (_L, "MED_x_LOW"),
            (_L, _H): (_L, "LOW_x_HIGH_overconfident"),
            (_L, _M): (_L, "LOW_x_MED"),
            (_L, _L): (_L, "LOW_x_LOW"),
        }
    )

    def aggregate(
        self,
        c_rules: ConfidenceLevel,
        self_confidence: ConfidenceLevel | None,
    ) -> tuple[ConfidenceLevel, str]:
        effective_self = self_confidence if self_confidence is not None else self._M
        return self._AGGREGATION_TABLE[(c_rules, effective_self)]
