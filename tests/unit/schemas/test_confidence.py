"""
schemas/confidence.py 단위 테스트.

검증 항목:
  evaluate_source_credibility():
    - RAG 출처: recency + source_tier + rag_relevance 기준 평가
    - Web 출처: rag_relevance 자동 1
    - 오래된 출처: recency=0 → score=0
    - cosine 경계값 (0.75)

  calculate_confidence_scores():
    - 6개 축 모두 채워진 경우 overall=1.0
    - 일부 축 누락 시 해당 축 score=0
    - 교차 검증 보너스 (2개 이상 출처)
    - finding은 있으나 source_ids 빈 경우
"""

import pytest
from datetime import datetime, timezone, timedelta

from schemas.agent_io import SourceRecord, SourceType, FindingWithSource
from schemas.confidence import (
    evaluate_source_credibility,
    calculate_confidence_scores,
    AXIS_FIELDS,
    RECENCY_MONTHS,
    RAG_RELEVANCE_FLOOR,
)


# ─────────────────────────────────────────────
# 픽스처 헬퍼
# ─────────────────────────────────────────────

def _make_source(
    source_id="src_001",
    source_type=SourceType.RAG_FAISS,
    months_ago=3,
) -> SourceRecord:
    retrieved_at = (
        datetime.now(timezone.utc) - timedelta(days=months_ago * 30)
    ).isoformat()
    return SourceRecord(
        source_id=source_id,
        url="https://example.com",
        title="테스트 출처",
        retrieved_at=retrieved_at,
        source_type=source_type,
        credibility_score=0,
        credibility_flags={},
    )


def _make_finding(source_ids: list[str], axis="캐즘 대응") -> FindingWithSource:
    return FindingWithSource(
        content="원문 컨텍스트",
        source_ids=source_ids,
        analysis_axis=axis,
    )


# ─────────────────────────────────────────────
# evaluate_source_credibility
# ─────────────────────────────────────────────

class TestEvaluateSourceCredibility:

    def test_rag_source_all_pass_returns_score_1(self):
        source = _make_source(source_type=SourceType.RAG_FAISS, months_ago=3)
        result = evaluate_source_credibility(source, rag_cosine_score=0.85)
        assert result["credibility_score"] == 1
        assert result["credibility_flags"]["recency"] == 1
        assert result["credibility_flags"]["source_tier"] == 1
        assert result["credibility_flags"]["rag_relevance"] == 1

    def test_rag_source_low_cosine_returns_score_0(self):
        source = _make_source(source_type=SourceType.RAG_FAISS, months_ago=3)
        result = evaluate_source_credibility(source, rag_cosine_score=0.50)
        assert result["credibility_flags"]["rag_relevance"] == 0
        assert result["credibility_score"] == 0

    def test_rag_cosine_at_floor_passes(self):
        """경계값 0.75는 통과해야 함"""
        source = _make_source(source_type=SourceType.RAG_FAISS, months_ago=1)
        result = evaluate_source_credibility(source, rag_cosine_score=RAG_RELEVANCE_FLOOR)
        assert result["credibility_flags"]["rag_relevance"] == 1

    def test_web_source_rag_relevance_auto_1(self):
        """Web 출처는 rag_relevance 자동 1"""
        source = _make_source(source_type=SourceType.WEB, months_ago=2)
        result = evaluate_source_credibility(source, rag_cosine_score=None)
        assert result["credibility_flags"]["rag_relevance"] == 1

    def test_old_source_fails_recency(self):
        """12개월 초과 → recency=0 → score=0"""
        source = _make_source(months_ago=RECENCY_MONTHS + 1)
        result = evaluate_source_credibility(source, rag_cosine_score=0.90)
        assert result["credibility_flags"]["recency"] == 0
        assert result["credibility_score"] == 0

    def test_invalid_retrieved_at_fails_recency(self):
        source = _make_source()
        source["retrieved_at"] = "not-a-date"
        result = evaluate_source_credibility(source, rag_cosine_score=0.80)
        assert result["credibility_flags"]["recency"] == 0

    def test_cross_verified_initialized_to_zero(self):
        """단일 출처 평가 시 cross_verified는 항상 0으로 초기화"""
        source = _make_source()
        result = evaluate_source_credibility(source, rag_cosine_score=0.90)
        assert result["credibility_flags"]["cross_verified"] == 0

    def test_cross_verified_excluded_from_single_source_score(self):
        """cross_verified=0이어도 나머지 3가지가 1이면 score=1"""
        source = _make_source(source_type=SourceType.RAG_FAISS, months_ago=1)
        result = evaluate_source_credibility(source, rag_cosine_score=0.85)
        # cross_verified=0이지만 score=1이어야 함
        assert result["credibility_flags"]["cross_verified"] == 0
        assert result["credibility_score"] == 1

    def test_rag_rewritten_source_treated_as_industry_report(self):
        source = _make_source(source_type=SourceType.RAG_REWRITTEN, months_ago=2)
        result = evaluate_source_credibility(source, rag_cosine_score=0.80)
        assert result["credibility_flags"]["source_tier"] == 1


# ─────────────────────────────────────────────
# calculate_confidence_scores
# ─────────────────────────────────────────────

class TestCalculateConfidenceScores:

    def _make_evaluated_source(self, source_id, months_ago=2, cosine=0.90):
        src = _make_source(source_id=source_id, months_ago=months_ago)
        return evaluate_source_credibility(src, rag_cosine_score=cosine)

    def test_all_axes_filled_overall_1(self):
        sources = [self._make_evaluated_source(f"src_{i:03d}") for i in range(6)]
        findings = {
            ax: _make_finding([f"src_{i:03d}"])
            for i, ax in enumerate(AXIS_FIELDS)
        }
        scores = calculate_confidence_scores(findings, sources)
        assert scores["overall"] == 1.0
        for ax in AXIS_FIELDS:
            assert scores[ax] == 1

    def test_missing_axis_scores_zero(self):
        sources = [self._make_evaluated_source("src_000")]
        findings = {"ev_response": _make_finding(["src_000"])}
        scores = calculate_confidence_scores(findings, sources)
        assert scores["ev_response"] == 1
        assert scores["market_position"] == 0
        assert scores["overall"] == pytest.approx(1 / 6, rel=0.01)

    def test_empty_findings_overall_zero(self):
        scores = calculate_confidence_scores({}, [])
        assert scores["overall"] == 0.0
        for ax in AXIS_FIELDS:
            assert scores[ax] == 0

    def test_finding_with_no_source_ids(self):
        finding = FindingWithSource(
            content="내용", source_ids=[], analysis_axis="테스트"
        )
        scores = calculate_confidence_scores({"ev_response": finding}, [])
        assert scores["ev_response"] == 0

    def test_cross_verified_bonus_with_two_sources(self):
        s1 = self._make_evaluated_source("src_001")
        s2 = self._make_evaluated_source("src_002")
        finding = _make_finding(["src_001", "src_002"])
        scores = calculate_confidence_scores({"ev_response": finding}, [s1, s2])
        assert scores["ev_response"] == 1

    def test_overall_is_average_of_axis_scores(self):
        sources = [self._make_evaluated_source(f"src_{i:03d}") for i in range(3)]
        findings = {
            "ev_response":     _make_finding(["src_000"]),
            "market_position": _make_finding(["src_001"]),
            "tech_portfolio":  _make_finding(["src_002"]),
            # 나머지 3개 없음
        }
        scores = calculate_confidence_scores(findings, sources)
        expected_overall = 3 / 6
        assert scores["overall"] == pytest.approx(expected_overall, rel=0.01)

    def test_axis_fields_constant_has_six_items(self):
        assert len(AXIS_FIELDS) == 6
        expected = {"ev_response", "market_position", "tech_portfolio",
                    "ess_strategy", "regulatory_risk", "cost_structure"}
        assert set(AXIS_FIELDS) == expected
