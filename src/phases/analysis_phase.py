"""
Phase 3 — Comparative SWOT + Report with HITL #3.

그래프 흐름:
  START
    → [swot_agent_node]    run_comparative_swot() 호출
    → [report_agent_node]  run_report_agent() 호출
    → [hitl_3_node]        interrupt() — Human Review #3
    → END  (approve 시 Word 저장)
    또는
    → [swot_agent_node]    redo 시 SWOT부터 재실행 (최대 2회)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agents.comparative_swot import run_comparative_swot
from agents.report_agent import build_word_report, run_report_agent
from schemas.agent_io import AgentFailureType, AgentStatus, StrategyAgentOutput
from schemas.market_context import MarketContext
from logging_utils import get_logger


logger = get_logger("phase3_analysis")

MAX_RETRIES = 2
DEFAULT_OUTPUT_PATH = Path("data/reports/report.docx")


# ─────────────────────────────────────────────
# State 정의
# ─────────────────────────────────────────────

def _last_write(existing, new):
    return new if new is not None else existing

def _append_errors(existing: list, new: list) -> list:
    return (existing or []) + (new or [])


class Phase3Input(TypedDict):
    user_request:     str
    market_context:   MarketContext
    skon_result:      StrategyAgentOutput
    catl_result:      StrategyAgentOutput
    output_path:      str  # Word 저장 경로


class Phase3Output(TypedDict):
    comparative_swot: dict
    report:           dict
    output_path:      str


class Phase3State(TypedDict, total=False):
    # 입력 (불변)
    user_request:   str
    market_context: MarketContext
    skon_result:    StrategyAgentOutput
    catl_result:    StrategyAgentOutput
    output_path:    str

    # 실행 결과
    swot_result:    Annotated[dict, _last_write]
    report_result:  Annotated[dict, _last_write]

    # HITL #3 제어
    retry_count:       int
    review_3_decision: Literal["approve", "redo"]
    review_3_feedback: str

    error_log: Annotated[list[dict], _append_errors]
    collected_sources: Annotated[list[dict], operator.add]


_IMMUTABLE = frozenset(Phase3Input.__annotations__.keys())


def _assert_immutable(update: dict, node_name: str) -> None:
    violations = _IMMUTABLE & update.keys()
    if violations:
        raise RuntimeError(f"[{node_name}] 입력 필드 write 금지: {violations}")


# ─────────────────────────────────────────────
# 노드
# ─────────────────────────────────────────────

async def swot_agent_node(state: Phase3State) -> dict:
    """Comparative SWOT 에이전트 실행."""
    retry = state.get("retry_count", 0)
    try:
        result = await run_comparative_swot(
            user_request=state["user_request"],
            market_context=state["market_context"],
            skon_result=state["skon_result"],
            catl_result=state["catl_result"],
            human_feedback=state.get("review_3_feedback", ""),
            final_revision_mode=(retry >= MAX_RETRIES),
        )
    except Exception as e:
        logger.error("swot_agent_node", e)
        result = {"error": str(e), "confidence": 0.0}

    update = {"swot_result": result}
    _assert_immutable(update, "swot_agent_node")
    return update


async def report_agent_node(state: Phase3State) -> dict:
    """Report 에이전트 실행 (SWOT 완료 후 항상 재생성)."""
    try:
        result = await run_report_agent(
            user_request=state["user_request"],
            market_context=state["market_context"],
            skon_result=state["skon_result"],
            catl_result=state["catl_result"],
            comparative_swot=state["swot_result"],
            human_feedback=state.get("review_3_feedback", ""),
            final_revision_mode=(state.get("retry_count", 0) >= MAX_RETRIES),
            collected_sources=state.get("collected_sources") or [],
        )
    except Exception as e:
        logger.error("report_agent_node", e)
        result = {"error": str(e)}

    update = {"report_result": result}
    _assert_immutable(update, "report_agent_node")
    return update


def hitl_3_node(state: Phase3State) -> dict:
    """HITL #3 — 보고서 검토 후 승인 or SWOT부터 재실행."""
    retry = state.get("retry_count", 0)
    report = state.get("report_result", {})
    swot   = state.get("swot_result", {})

    resume_value = interrupt({
        "phase":               "review_3",
        "report":              report,
        "swot_confidence":     swot.get("confidence"),
        "comparison_axes":     swot.get("comparison_axes", []),
        "decision_takeaways":  swot.get("decision_takeaways", []),
        "allowed_decisions":   _allowed_decisions(state),
        "retry_count":         retry,
        "retry_limit_reached": retry >= MAX_RETRIES,
    })

    decision = resume_value.get("decision", "approve")
    feedback = resume_value.get("feedback", "")

    update = {
        "review_3_decision": decision,
        "review_3_feedback": feedback,
        "retry_count":       retry + 1,
    }
    _assert_immutable(update, "hitl_3_node")
    return update


# ─────────────────────────────────────────────
# 라우터
# ─────────────────────────────────────────────

def _route_after_hitl_3(state: Phase3State) -> str:
    decision    = state.get("review_3_decision", "approve")
    retry_count = state.get("retry_count", 0)

    if decision == "approve":
        return "save_report_node"
    if retry_count >= MAX_RETRIES:
        return "save_report_node"  # 한계 도달 시 현재 결과로 저장
    return "swot_agent_node"


# ─────────────────────────────────────────────
# Word 저장 노드
# ─────────────────────────────────────────────

def save_report_node(state: Phase3State) -> dict:
    """보고서 Word 저장."""
    report = state.get("report_result", {})
    swot   = state.get("swot_result", {})
    path   = Path(state.get("output_path") or str(DEFAULT_OUTPUT_PATH))

    try:
        build_word_report(report=report, swot=swot, output_path=path)
        logger.node_exit("save_report", duration_sec=0, status="ok",
                         metadata={"path": str(path)})
    except Exception as e:
        logger.error("save_report_node", e)

    return {}


# ─────────────────────────────────────────────
# 그래프 빌더
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(Phase3State)
    builder.add_node("swot_agent_node",   swot_agent_node)
    builder.add_node("report_agent_node", report_agent_node)
    builder.add_node("hitl_3_node",       hitl_3_node)
    builder.add_node("save_report_node",  save_report_node)

    builder.add_edge(START,               "swot_agent_node")
    builder.add_edge("swot_agent_node",   "report_agent_node")
    builder.add_edge("report_agent_node", "hitl_3_node")
    builder.add_conditional_edges(
        "hitl_3_node",
        _route_after_hitl_3,
        ["swot_agent_node", "save_report_node"],
    )
    builder.add_edge("save_report_node", END)

    return builder


def make_checkpointer() -> MemorySaver:
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("schemas.agent_io", "AgentStatus"),
            ("schemas.agent_io", "AgentFailureType"),
            ("schemas.agent_io", "SourceType"),
        ]
    )
    return MemorySaver(serde=serde)


def compile_standalone():
    return build_graph().compile(checkpointer=make_checkpointer())


async def run(input: Phase3Input, config: RunnableConfig, app=None) -> Phase3Output:
    """Phase 3 실행."""
    if app is None:
        app = compile_standalone()

    initial: dict = {
        **input,
        "retry_count":       0,
        "review_3_feedback": "",
        "error_log":         [],
    }
    await app.ainvoke(initial, config)

    final = app.get_state(config).values
    return Phase3Output(
        comparative_swot=final.get("swot_result", {}),
        report=final.get("report_result", {}),
        output_path=final.get("output_path", str(DEFAULT_OUTPUT_PATH)),
    )


# ─────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────

def _allowed_decisions(state: Phase3State) -> list[str]:
    decisions = ["approve"]
    if state.get("retry_count", 0) < MAX_RETRIES:
        decisions.append("redo")
    return decisions
    return decisions
