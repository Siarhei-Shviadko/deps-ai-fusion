import pytest

from deps_ai_fusion.infrastructure.services.llm_extraction.confidence.aggregator import (
    ConfidenceAggregator,
)
from deps_ai_fusion.infrastructure.services.llm_extraction.confidence.models import (
    ConfidenceLevel,
)

H = ConfidenceLevel.HIGH
M = ConfidenceLevel.MEDIUM
L = ConfidenceLevel.LOW


@pytest.fixture()
def aggregator() -> ConfidenceAggregator:
    return ConfidenceAggregator()


class TestLowCRulesAlwaysLow:
    def test_low_c_rules_always_returns_low(self, aggregator: ConfidenceAggregator) -> None:
        for self_conf in (H, M, L, None):
            c_final, _ = aggregator.aggregate(L, self_conf)
            assert c_final == L


class TestAggregationTable:
    @pytest.mark.parametrize(
        "c_rules,self_conf,expected_final,expected_rule",
        [
            (H, H, H, "HIGH_x_HIGH"),
            (H, M, H, "HIGH_x_MED"),
            (H, L, M, "HIGH_x_LOW_disagree"),
            (M, H, H, "MED_x_HIGH"),
            (M, M, M, "MED_x_MED"),
            (M, L, L, "MED_x_LOW"),
            (L, H, L, "LOW_x_HIGH_overconfident"),
            (L, M, L, "LOW_x_MED"),
            (L, L, L, "LOW_x_LOW"),
        ],
    )
    def test_full_matrix(
        self,
        aggregator: ConfidenceAggregator,
        c_rules: ConfidenceLevel,
        self_conf: ConfidenceLevel,
        expected_final: ConfidenceLevel,
        expected_rule: str,
    ) -> None:
        c_final, rule_id = aggregator.aggregate(c_rules, self_conf)
        assert c_final == expected_final
        assert rule_id == expected_rule


class TestAbsentSelfConfidence:
    def test_none_self_confidence_treated_as_medium(self, aggregator: ConfidenceAggregator) -> None:
        c_final_none, _ = aggregator.aggregate(H, None)
        c_final_med, _ = aggregator.aggregate(H, M)
        assert c_final_none == c_final_med

    def test_none_with_low_c_rules_treated_as_medium(self, aggregator: ConfidenceAggregator) -> None:
        c_final, _ = aggregator.aggregate(L, None)
        assert c_final == L
