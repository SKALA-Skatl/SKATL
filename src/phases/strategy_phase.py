"""
Orchestrator 그래프.

역할:
  Market Agent(Phase 1)로부터 market_context를 받아
  SKON/CATL Strategy Agent를 병렬 실행하고,
  Human Review #2를 거쳐 결과를 Comparative SWOT Agent(Phase 3)로 넘긴다.

독립 실행:
  이 모듈은 독립적으로 실행 가능하다.
  run(input, config) 함수가 외부 진입점이며,
  상위 파이프라인에서 호출하거나 단독으로 테스트할 수 있다.

상위 파이프라인 편입 시:
  build_graph()가 반환하는 StateGraph를 컴파일해
  상위 파이프라인의 add_node()에 전달한다.
  OrchestratorState와 상위 파이프라인 State의 공통 키
  (user_request, market_context, skon_result, catl_result)를 통해
  데이터가 자동으로 전달된다.

그래프 흐름:
  START
    → [orchestrator_fanout]  Send() 라우터
    → [skon_agent_node]      input_schema=StrategyAgentInput
    → [catl_agent_node]      input_schema=StrategyAgentInput
    → [fan_in_node]          결과 병합 + 버전 검증
    → [hitl_2_node]          interrupt() — Human Review #2
    → END  (approve 시)
    또는
    → [orchestrator_fanout]  redo 시 재조사 루프
    또는
    → [error_handler] → END  (양쪽 모두 실패 시)
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, interrupt

from agents.strategy_agent import run_strategy_agent
from schemas.agent_io import (
    AgentFailureType,
    AgentStatus,
    ConfidenceScores,
    SCHEMA_VERSION,
    StrategyAgentInput,
    StrategyAgentOutput,
    validate_schema_version,
)
from schemas.state import (
    OrchestratorInput,
    OrchestratorOutput,
    OrchestratorState,
    assert_immutable_fields,
)
from logging_utils import get_logger


logger = get_logger("orchestrator")

MAX_RETRIES = 2

# 실패 시 기본 ConfidenceScores
_ZERO_CONFIDENCE = ConfidenceScores(
    ev_response=0, market_position=0, tech_portfolio=0,
    ess_strategy=0, regulatory_risk=0, cost_structure=0, overall=0.0,
)


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────

def _make_failed(
    company: Literal["SKON", "CATL"],
    failure_type: AgentFailureType,
) -> StrategyAgentOutput:
    return StrategyAgentOutput(
        schema_version=SCHEMA_VERSION,
        company=company,
        status=AgentStatus.FAILED,
        failure_type=failure_type,
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        llm_call_count=0,
        tool_call_log=[],
        sources=[],
        confidence_scores=_ZERO_CONFIDENCE,
    )


# ─────────────────────────────────────────────
# 라우터 (Send() 반환 — add_node 금지)
# ─────────────────────────────────────────────

def orchestrator_fanout(state: OrchestratorState) -> list[Send]:
    """
    최초 실행 또는 redo 시 호출.
    redo_targets에 따라 필요한 Agent만 Send.
    retry_count >= MAX_RETRIES인 Agent는 Skip.
    """
    redo_targets = state.get("redo_targets") or []
    is_initial   = not redo_targets
    sends: list[Send] = []

    skon_retry = state.get("skon_retry_count", 0)
    if (is_initial or "skon" in redo_targets) and skon_retry < MAX_RETRIES:
        sends.append(Send(
            "skon_agent_node",
            StrategyAgentInput(
                company="SKON",
                user_request=state["user_request"],
                market_context=state["market_context"],
                review_feedback=state.get("review_2_feedback", ""),
                retry_count=skon_retry,
            ),
        ))

    catl_retry = state.get("catl_retry_count", 0)
    if (is_initial or "catl" in redo_targets) and catl_retry < MAX_RETRIES:
        sends.append(Send(
            "catl_agent_node",
            StrategyAgentInput(
                company="CATL",
                user_request=state["user_request"],
                market_context=state["market_context"],
                review_feedback=state.get("review_2_feedback", ""),
                retry_count=catl_retry,
            ),
        ))

    return sends


# ─────────────────────────────────────────────
# Agent 노드 (input_schema=StrategyAgentInput)
# ─────────────────────────────────────────────

async def skon_agent_node(state: StrategyAgentInput) -> dict:
    """
    SKON Strategy Agent 실행.
    input_schema=StrategyAgentInput으로 등록되어 OrchestratorState와 격리.
    반환값은 OrchestratorState에 병합된다.
    """
    try:
        result = await run_strategy_agent(state)
    except Exception as e:
        logger.error("skon_agent_node", e)
        result = _make_failed("SKON", AgentFailureType.LLM_ERROR)

    update = {
        "skon_result":        result,
        "skon_retry_count":   state["retry_count"] + 1,
        "collected_sources":  list(result.get("sources") or []),
    }
    assert_immutable_fields(update, "skon_agent_node")
    return update


async def catl_agent_node(state: StrategyAgentInput) -> dict:
    """
    CATL Strategy Agent 실행.
    input_schema=StrategyAgentInput으로 등록되어 OrchestratorState와 격리.
    """
    try:
        result = await run_strategy_agent(state)
    except Exception as e:
        logger.error("catl_agent_node", e)
        result = _make_failed("CATL", AgentFailureType.LLM_ERROR)

    update = {
        "catl_result":        result,
        "catl_retry_count":   state["retry_count"] + 1,
        "collected_sources":  list(result.get("sources") or []),
    }
    assert_immutable_fields(update, "catl_agent_node")
    return update


# ─────────────────────────────────────────────
# Fan-in 노드
# ─────────────────────────────────────────────

def fan_in_node(state: OrchestratorState) -> dict:
    """
    SKON/CATL 결과 수집.
    스키마 버전 검증 + 양쪽 실패 여부 기록.
    """
    skon = state.get("skon_result") or {}
    catl = state.get("catl_result") or {}

    version_ok = {
        "skon": validate_schema_version(skon),
        "catl": validate_schema_version(catl),
    }
    if not all(version_ok.values()):
        logger.node_enter("fan_in_node", {
            "warning": "schema_version_mismatch", **version_ok,
        })

    skon_status  = skon.get("status", AgentStatus.FAILED)
    catl_status  = catl.get("status", AgentStatus.FAILED)
    both_failed  = (skon_status == AgentStatus.FAILED and
                    catl_status == AgentStatus.FAILED)

    update: dict = {
        "fan_in_status": {
            "skon_status":       skon_status,
            "catl_status":       catl_status,
            "both_failed":       both_failed,
            "schema_version_ok": version_ok,
        },
        "redo_targets": [],  # 재조사 대상 초기화
    }

    if both_failed:
        update["error_log"] = [{
            "event":        "both_agents_failed",
            "skon_failure": skon.get("failure_type"),
            "catl_failure": catl.get("failure_type"),
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        }]

    assert_immutable_fields(update, "fan_in_node")
    return update


# ─────────────────────────────────────────────
# HITL #2 노드
# ─────────────────────────────────────────────

def hitl_2_node(state: OrchestratorState) -> dict:
    """
    Human Review #2.
    interrupt()로 실행을 일시 정지하고 Human 피드백을 기다린다.

    interrupt payload:
      SKON/CATL 분석 결과 원문 그대로 전달 (요약 없음).
      Human이 충분성·비교 축 일치 여부를 판단할 수 있도록
      신뢰도 스코어와 출처 목록을 함께 전달.

    resume 형식:
      {"decision": "approve"|"redo_skon"|"redo_catl"|"redo_both",
       "feedback": "<재조사 지시>"}  # redo 시에만 필요
    """
    skon_retry = state.get("skon_retry_count", 0)
    catl_retry = state.get("catl_retry_count", 0)

    # SKON/CATL 모두 재시도 한도 도달 시 interrupt 없이 자동 승인
    if skon_retry >= MAX_RETRIES and catl_retry >= MAX_RETRIES:
        print(f"[hitl_2_node] SKON/CATL 모두 재시도 한도({MAX_RETRIES}) 도달 — 자동 승인")
        return {
            "review_2_decision": "approve",
            "review_2_feedback": "",
            "redo_targets":      [],
        }

    skon = state.get("skon_result") or {}
    catl = state.get("catl_result") or {}

    allowed = _allowed_decisions(state)
    resume_value = interrupt({
        "phase":    "review_2",
        "skon":     _build_review_context(skon, skon_retry),
        "catl":     _build_review_context(catl, catl_retry),
        "allowed_decisions":   allowed,
        "retry_limits_reached": {
            "skon": skon_retry >= MAX_RETRIES,
            "catl": catl_retry >= MAX_RETRIES,
        },
    })

    validated = _validate_resume(resume_value, state)
    if not validated["valid"]:
        logger.error("hitl_2_node",
                     ValueError(f"Invalid resume: {validated['reason']}"))

    decision = resume_value.get("decision", "approve")
    feedback = resume_value.get("feedback", "")

    update = {
        "review_2_decision": decision,
        "review_2_feedback": feedback,
        "redo_targets":      _decision_to_targets(decision),
    }
    assert_immutable_fields(update, "hitl_2_node")
    return update


# ─────────────────────────────────────────────
# 에러 핸들러 노드
# ─────────────────────────────────────────────

def error_handler_node(state: OrchestratorState) -> dict:
    logger.node_enter("error_handler", {
        "skon_status": (state.get("skon_result") or {}).get("status"),
        "catl_status": (state.get("catl_result") or {}).get("status"),
    })
    update = {"error_log": [{
        "event":     "orchestrator_terminated",
        "reason":    "both_agents_failed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }]}
    assert_immutable_fields(update, "error_handler_node")
    return update


# ─────────────────────────────────────────────
# 조건부 엣지 라우터
# ─────────────────────────────────────────────

def _route_after_fan_in(state: OrchestratorState) -> str:
    fan_in_status = state.get("fan_in_status") or {}
    return "error_handler" if fan_in_status.get("both_failed") else "hitl_2_node"


def _route_after_hitl_2(state: OrchestratorState) -> str | list[Send]:
    """
    approve / retry 한계 소진 → END
    redo                      → orchestrator_fanout (Send() 리스트)
    """
    decision     = state.get("review_2_decision", "approve")
    redo_targets = state.get("redo_targets") or []

    if decision == "approve":
        return END

    all_exhausted = redo_targets and all(
        (t == "skon" and state.get("skon_retry_count", 0) >= MAX_RETRIES) or
        (t == "catl" and state.get("catl_retry_count", 0) >= MAX_RETRIES)
        for t in redo_targets
    )
    if all_exhausted:
        return END

    return orchestrator_fanout(state)


def _route_error_handler(state: OrchestratorState) -> str:
    return "__end__"


# ─────────────────────────────────────────────
# 그래프 빌더
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Orchestrator StateGraph 반환.
    compile()은 호출하지 않음.

    독립 실행:
        app = build_graph().compile(checkpointer=make_checkpointer())

    서브그래프로 편입:
        parent.add_node("orchestrator", build_graph().compile())
    """
    builder = StateGraph(OrchestratorState)

    # Agent 노드: input_schema로 OrchestratorState와 격리
    builder.add_node("skon_agent_node", skon_agent_node,
                     input_schema=StrategyAgentInput)
    builder.add_node("catl_agent_node", catl_agent_node,
                     input_schema=StrategyAgentInput)
    builder.add_node("fan_in_node",      fan_in_node)
    builder.add_node("hitl_2_node",      hitl_2_node)
    builder.add_node("error_handler",    error_handler_node)

    # START → orchestrator_fanout (라우터, Send() 반환)
    builder.add_conditional_edges(
        START,
        orchestrator_fanout,
        ["skon_agent_node", "catl_agent_node"],
    )

    # Fan-in
    builder.add_edge("skon_agent_node", "fan_in_node")
    builder.add_edge("catl_agent_node", "fan_in_node")

    # Fan-in → HITL #2 or 에러
    builder.add_conditional_edges(
        "fan_in_node",
        _route_after_fan_in,
        {"hitl_2_node": "hitl_2_node", "error_handler": "error_handler"},
    )

    # HITL #2 → approve(END) or redo(Send 리스트)
    builder.add_conditional_edges(
        "hitl_2_node",
        _route_after_hitl_2,
        ["skon_agent_node", "catl_agent_node", END],
    )

    # 에러 → END
    builder.add_conditional_edges(
        "error_handler",
        _route_error_handler,
        {"__end__": END},
    )

    return builder


# ─────────────────────────────────────────────
# Checkpointer 팩토리
# ─────────────────────────────────────────────

def make_checkpointer() -> MemorySaver:
    """
    OrchestratorState의 Enum 타입을 허용 목록에 등록한 MemorySaver 반환.
    독립 실행 및 서브그래프 편입 시 모두 사용.
    """
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("schemas.agent_io", "AgentStatus"),
            ("schemas.agent_io", "AgentFailureType"),
            ("schemas.agent_io", "SourceType"),
        ]
    )
    return MemorySaver(serde=serde)


# ─────────────────────────────────────────────
# 독립 실행 진입점
# ─────────────────────────────────────────────

def compile_standalone():
    """
    독립 실행용 컴파일 앱.
    Phase 1/3 없이 Orchestrator만 단독 실행할 때 사용.

    사용 예:
        app = compile_standalone()
        result = await app.ainvoke(
            OrchestratorInput(
                user_request="배터리 전략 분석",
                market_context=MOCK_MARKET_CONTEXT,
            ),
            config={"configurable": {"thread_id": "test_001"}},
        )
    """
    return build_graph().compile(checkpointer=make_checkpointer())


async def run(
    input: OrchestratorInput,
    config: RunnableConfig,
    app=None,
) -> OrchestratorOutput:
    """
    Orchestrator 단일 실행 (interrupt 없이 완료까지).
    상위 파이프라인에서 호출하는 방식.

    interrupt가 발생하면 GraphInterrupt 예외가 발생한다.
    streaming이 필요한 경우 app.astream()을 직접 사용할 것.

    Args:
        input  : OrchestratorInput (user_request, market_context)
        config : LangGraph RunnableConfig (thread_id 포함)
        app    : 컴파일된 앱. None이면 compile_standalone() 사용.

    Returns:
        OrchestratorOutput (skon_result, catl_result)
    """
    if app is None:
        app = compile_standalone()

    initial: dict = {
        **input,
        "skon_retry_count": 0,
        "catl_retry_count": 0,
        "redo_targets":     [],
        "error_log":        [],
    }

    await app.ainvoke(initial, config)

    final = app.get_state(config).values
    return OrchestratorOutput(
        skon_result=final.get("skon_result", {}),
        catl_result=final.get("catl_result", {}),
    )


# ─────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────

def _build_review_context(result: StrategyAgentOutput, retry_count: int) -> dict:
    """HITL #2 payload — 원문 그대로, 출처 직접 매핑."""
    return {
        "company":           result.get("company"),
        "status":            result.get("status"),
        "failure_type":      result.get("failure_type"),
        "retry_count":       retry_count,
        "max_retries":       MAX_RETRIES,
        "ev_response":       result.get("ev_response"),
        "market_position":   result.get("market_position"),
        "tech_portfolio":    result.get("tech_portfolio"),
        "ess_strategy":      result.get("ess_strategy"),
        "regulatory_risk":   result.get("regulatory_risk"),
        "cost_structure":    result.get("cost_structure"),
        "sources":           result.get("sources", []),
        "confidence_scores": result.get("confidence_scores"),
    }


def _allowed_decisions(state: OrchestratorState) -> list[str]:
    decisions = ["approve"]
    if state.get("skon_retry_count", 0) < MAX_RETRIES:
        decisions += ["redo_skon", "redo_both"]
    if state.get("catl_retry_count", 0) < MAX_RETRIES:
        if "redo_both" not in decisions:
            decisions.append("redo_both")
        decisions.append("redo_catl")
    return decisions


def _validate_resume(resume_value: dict, state: OrchestratorState) -> dict:
    valid = {"approve", "redo_skon", "redo_catl", "redo_both"}
    if not isinstance(resume_value, dict):
        return {"valid": False, "reason": "dict 이어야 함"}
    decision = resume_value.get("decision")
    if decision not in valid:
        return {"valid": False, "reason": f"유효하지 않은 decision: {decision}"}
    if decision in ("redo_skon", "redo_both"):
        if state.get("skon_retry_count", 0) >= MAX_RETRIES:
            return {"valid": False, "reason": "SKON retry 한계 도달"}
    if decision in ("redo_catl", "redo_both"):
        if state.get("catl_retry_count", 0) >= MAX_RETRIES:
            return {"valid": False, "reason": "CATL retry 한계 도달"}
    return {"valid": True, "reason": None}


def _decision_to_targets(decision: str) -> list[str]:
    return {
        "approve":   [],
        "redo_skon": ["skon"],
        "redo_catl": ["catl"],
        "redo_both": ["skon", "catl"],
    }.get(decision, [])
