"""
Strategy Agent 입출력 스키마.

SKON/CATL 에이전트가 공유하는 타입 정의.
SCHEMA_VERSION을 올리면 fan_in_node에서 버전 불일치를 감지한다.
"""

from __future__ import annotations
from enum import Enum
from typing import Literal, TypedDict

from schemas.market_context import MarketContext


SCHEMA_VERSION = "1.0.0"


# ─────────────────────────────────────────────
# 열거형
# ─────────────────────────────────────────────

class AgentStatus(str, Enum):
    SUCCESS         = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED          = "failed"


class AgentFailureType(str, Enum):
    LLM_ERROR    = "llm_error"
    TOOL_ERROR   = "tool_error"
    TIMEOUT      = "timeout"
    MAX_ITER     = "max_iterations"
    SCHEMA_ERROR = "schema_error"


class SourceType(str, Enum):
    WEB           = "web"
    RAG_FAISS     = "rag_faiss"
    RAG_REWRITTEN = "rag_rewritten"


# ─────────────────────────────────────────────
# 출처 및 발견 타입
# ─────────────────────────────────────────────

class SourceRecord(TypedDict):
    source_id:         str    # "src_001" 형태 고유 ID
    url:               str
    title:             str
    retrieved_at:      str    # ISO 8601
    published_date: str | None # 콘텐츠 작성 날짜 (신규)
    source_type:       SourceType
    credibility_score: int    # 바이너리: 1 (신뢰) or 0 (미달)
    credibility_flags: dict   # {"recency": 1, "source_tier": 1, ...}


class FindingWithSource(TypedDict):
    """원문 컨텍스트를 그대로 보존. 요약 금지."""
    content:       str
    source_ids:    list[str]
    analysis_axis: str


# ─────────────────────────────────────────────
# Confidence Score
# ─────────────────────────────────────────────

class ConfidenceScores(TypedDict):
    """
    각 분석 축의 신뢰도. 바이너리: 1 or 0.

    단일 출처: recency + source_tier + rag_relevance 모두 1이면 score=1
    복수 출처: 위 3가지 + cross_verified 모두 1이면 score=1
    """
    ev_response:     int
    market_position: int
    tech_portfolio:  int
    ess_strategy:    int
    regulatory_risk: int
    cost_structure:  int
    overall:         float


# ─────────────────────────────────────────────
# Agent 입력 State — Send() payload이자 노드 input_schema
# ─────────────────────────────────────────────

class StrategyAgentInput(TypedDict):
    """
    Orchestrator가 Send()로 전달하는 payload.
    skon_agent_node / catl_agent_node의 input_schema로 등록되어
    OrchestratorState와 격리된 독립 State로 동작한다.
    """
    company:         Literal["SKON", "CATL"]
    market_context:  MarketContext
    review_feedback: str
    retry_count:     int


# ─────────────────────────────────────────────
# Agent 출력
# ─────────────────────────────────────────────

class StrategyAgentOutput(TypedDict, total=False):
    """
    total=False: 실패 시 일부 필드만 채워도 유효.

    status 결정:
      6개 축 모두 없음 → failed
      일부만 채워짐    → partial_success
      6개 모두 채워짐  → success
    """
    schema_version:     str
    company:            Literal["SKON", "CATL"]
    status:             AgentStatus
    failure_type:       AgentFailureType | None
    analysis_timestamp: str
    llm_call_count:     int
    tool_call_log:      list[dict]

    ev_response:     FindingWithSource
    market_position: FindingWithSource
    tech_portfolio:  FindingWithSource
    ess_strategy:    FindingWithSource
    regulatory_risk: FindingWithSource
    cost_structure:  FindingWithSource

    sources:           list[SourceRecord]
    confidence_scores: ConfidenceScores


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────

def validate_schema_version(output: StrategyAgentOutput) -> bool:
    return output.get("schema_version") == SCHEMA_VERSION
