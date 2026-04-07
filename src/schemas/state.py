"""
Orchestrator State 정의.

외부 인터페이스:
  OrchestratorInput  — 진입점에서 주입하는 초기값
                        (Market Agent 결과 + 사용자 요청)
  OrchestratorOutput — 종료 후 다음 단계(Phase 3)로 넘기는 값
                        (SKON/CATL 분석 결과)

내부 State:
  OrchestratorState  — 그래프 실행 중 관리되는 전체 State
                        Input/Output 필드를 포함하며,
                        Fan-out/Fan-in 제어 및 HITL 관련 필드 추가

병합을 위한 설계 원칙:
  - OrchestratorInput  필드명 = 상위 파이프라인 State 필드명과 일치
  - OrchestratorOutput 필드명 = 하위 파이프라인 State 필드명과 일치
  이렇게 하면 LangGraph 서브그래프로 편입 시 State 공유 키를 통해
  데이터가 자동으로 전달된다.
"""

from __future__ import annotations
import operator
from typing import Annotated, Literal, TypedDict

from schemas.agent_io import StrategyAgentOutput
from schemas.market_context import MarketContext


# ─────────────────────────────────────────────
# Reducer
# ─────────────────────────────────────────────

def _last_write(existing, new):
    """병렬 write 시 나중 값 우선."""
    return new if new is not None else existing


def _append_errors(existing: list, new: list) -> list:
    """에러 로그 누적."""
    return (existing or []) + (new or [])


# ─────────────────────────────────────────────
# 외부 인터페이스
# ─────────────────────────────────────────────

class OrchestratorInput(TypedDict):
    """
    Orchestrator 진입점에서 주입하는 초기값.
    상위 파이프라인(Market Agent)에서 전달받는다.

    병합 시 주의:
      상위 파이프라인 State에 동일한 키(user_request, market_context)가
      있어야 LangGraph 서브그래프 편입 시 자동 전달된다.
    """
    user_request:   str
    market_context: MarketContext


class OrchestratorOutput(TypedDict):
    """
    Orchestrator 종료 후 하위 파이프라인(Phase 3)으로 넘기는 값.

    병합 시 주의:
      하위 파이프라인 State에 동일한 키(skon_result, catl_result)가
      있어야 LangGraph 서브그래프 편입 시 자동 전달된다.
    """
    skon_result: StrategyAgentOutput
    catl_result: StrategyAgentOutput


# ─────────────────────────────────────────────
# 내부 State
# ─────────────────────────────────────────────

class OrchestratorState(TypedDict, total=False):
    """
    Orchestrator 그래프 전체 State.

    total=False: 모든 필드가 선택적.
    초기 진입 시 OrchestratorInput 필드만 채워진 상태로 시작한다.
    """
    # ── 입력 (OrchestratorInput) ──────────────
    # 불변: 노드에서 반환 금지 (assert_immutable_fields로 보호)
    user_request:   str
    market_context: MarketContext

    # ── 출력 (OrchestratorOutput) ─────────────
    # Fan-in 시 병렬 write → _last_write reducer
    skon_result: Annotated[StrategyAgentOutput, _last_write]
    catl_result: Annotated[StrategyAgentOutput, _last_write]

    # ── Fan-out/Fan-in 제어 ───────────────────
    skon_retry_count: int        # 현재까지 재조사 횟수
    catl_retry_count: int
    redo_targets:     list[str]  # ["skon"] | ["catl"] | ["skon","catl"]

    # ── HITL #2 ───────────────────────────────
    review_2_decision: Literal["approve", "redo_skon", "redo_catl", "redo_both"]
    review_2_feedback: str

    # ── 내부 제어 ─────────────────────────────
    fan_in_status: dict
    error_log:     Annotated[list[dict], _append_errors]
    collected_sources: Annotated[list[dict], operator.add]


# ─────────────────────────────────────────────
# 불변 필드 보호
# ─────────────────────────────────────────────

_IMMUTABLE = frozenset(OrchestratorInput.__annotations__.keys())


def assert_immutable_fields(update: dict, node_name: str) -> None:
    """노드 반환값에 입력 필드가 포함되면 RuntimeError."""
    violations = _IMMUTABLE & update.keys()
    if violations:
        raise RuntimeError(
            f"[{node_name}] 입력 필드 write 금지: {violations}"
        )
