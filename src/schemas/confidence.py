"""
출처 신뢰도 평가 — 바이너리 (1 or 0).

평가 기준 4가지 모두 1이어야 해당 출처 credibility_score = 1.
  recency        : 수집 자료가 12개월 이내
  source_tier    : 출처 유형이 공식 공시/산업 리포트/주요 언론
  rag_relevance  : FAISS cosine similarity >= 0.75 (RAG 출처에만 적용)
  cross_verified : 동일 분석 축에서 2개 이상 출처가 같은 사실을 지지
"""

from __future__ import annotations
from datetime import datetime, timezone

from schemas.agent_io import (
    ConfidenceScores,
    FindingWithSource,
    SourceRecord,
    SourceType,
)


# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────

RECENCY_MONTHS      = 12     # 이내면 recency = 1
RAG_RELEVANCE_FLOOR = 0.75   # cosine similarity 하한

HIGH_TIER_SOURCES = {
    "official_filing",   # SEC, DART, 공식 IR
    "industry_report",   # BloombergNEF, SNE Research, Wood Mackenzie
    "major_news",        # Reuters, Bloomberg, FT, WSJ
}

# SourceType → 내부 tier 문자열 매핑
SOURCE_TYPE_TO_TIER: dict[SourceType, str] = {
    SourceType.WEB:           "major_news",        # 보수적으로 평가
    SourceType.RAG_FAISS:     "industry_report",   # PDF 문서 기반
    SourceType.RAG_REWRITTEN: "industry_report",
}


# ─────────────────────────────────────────────
# 단일 출처 신뢰도 평가
# ─────────────────────────────────────────────

def evaluate_source_credibility(
    source: SourceRecord,
    rag_cosine_score: float | None = None,
) -> SourceRecord:
    """
    SourceRecord에 credibility_score와 credibility_flags를 채워 반환.

    Args:
        source           : 평가할 출처 레코드
        rag_cosine_score : RAG 출처일 때 FAISS cosine similarity 값.
                           web 출처는 None.
    """
    flags: dict[str, int] = {}

    # 1. 최신성
    try:
        retrieved = datetime.fromisoformat(source["retrieved_at"])
        months_ago = (datetime.now(timezone.utc) - retrieved).days / 30
        flags["recency"] = 1 if months_ago <= RECENCY_MONTHS else 0
    except (ValueError, KeyError):
        flags["recency"] = 0

    # 2. 출처 유형 등급
    tier = SOURCE_TYPE_TO_TIER.get(source.get("source_type"), "unknown")
    flags["source_tier"] = 1 if tier in HIGH_TIER_SOURCES else 0

    # 3. RAG 관련성 (RAG 출처에만 적용, web은 자동 1)
    if source.get("source_type") == SourceType.WEB:
        flags["rag_relevance"] = 1
    elif rag_cosine_score is not None:
        flags["rag_relevance"] = 1 if rag_cosine_score >= RAG_RELEVANCE_FLOOR else 0
    else:
        flags["rag_relevance"] = 0

    # 4. 교차 검증은 축 레벨에서 평가 (여기선 임시 0, 아래에서 갱신)
    #    단일 출처 score 산정에는 포함하지 않음 — 축 레벨에서만 의미 있음
    flags["cross_verified"] = 0

    # recency, source_tier, rag_relevance 3가지만으로 score 결정
    # cross_verified는 calculate_confidence_scores에서 갱신 후 재산정
    score_flags = {k: v for k, v in flags.items() if k != "cross_verified"}
    score = 1 if all(v == 1 for v in score_flags.values()) else 0

    return SourceRecord(
        **{k: v for k, v in source.items()
           if k not in ("credibility_score", "credibility_flags")},
        credibility_score=score,
        credibility_flags=flags,
    )


# ─────────────────────────────────────────────
# 축별 Confidence Score 산정
# ─────────────────────────────────────────────

AXIS_FIELDS = [
    "ev_response",
    "market_position",
    "tech_portfolio",
    "ess_strategy",
    "regulatory_risk",
    "cost_structure",
]


def calculate_confidence_scores(
    findings: dict[str, FindingWithSource | None],
    sources: list[SourceRecord],
) -> ConfidenceScores:
    """
    Args:
        findings : axis_name → FindingWithSource (없으면 None)
        sources  : 전체 SourceRecord 목록 (credibility_score 이미 평가된 상태)

    Returns:
        ConfidenceScores (바이너리 축별 + overall 평균)
    """
    source_map = {s["source_id"]: s for s in sources}
    scores: dict[str, int] = {}

    for axis in AXIS_FIELDS:
        finding = findings.get(axis)
        if not finding:
            scores[axis] = 0
            continue

        supporting_ids = finding.get("source_ids", [])
        supporting = [source_map[sid] for sid in supporting_ids if sid in source_map]

        if not supporting:
            scores[axis] = 0
            continue

        # 교차 검증: 지지 출처 2개 이상이면 보너스
        cross_verified = 1 if len(supporting) >= 2 else 0
        for s in supporting:
            s["credibility_flags"]["cross_verified"] = cross_verified

        # 축 confidence 결정:
        #   cross_verified=1: 4가지 flags 모두 1이어야 score=1
        #   cross_verified=0: recency, source_tier, rag_relevance 3가지만으로 판단
        def _axis_score(s: dict) -> int:
            flags = s["credibility_flags"]
            base_flags = {k: v for k, v in flags.items() if k != "cross_verified"}
            if cross_verified:
                return 1 if all(v == 1 for v in flags.values()) else 0
            return 1 if all(v == 1 for v in base_flags.values()) else 0

        scores[axis] = 1 if any(_axis_score(s) == 1 for s in supporting) else 0

    overall = sum(scores.values()) / len(AXIS_FIELDS)

    return ConfidenceScores(
        ev_response=scores.get("ev_response", 0),
        market_position=scores.get("market_position", 0),
        tech_portfolio=scores.get("tech_portfolio", 0),
        ess_strategy=scores.get("ess_strategy", 0),
        regulatory_risk=scores.get("regulatory_risk", 0),
        cost_structure=scores.get("cost_structure", 0),
        overall=round(overall, 3),
    )
