"""
Phase 1 market workflow with HITL #1.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.market_agent import run_market_agent
from logging_utils import get_logger
from schemas.agent_io import AgentFailureType, AgentStatus, SourceType
from schemas.market_agent_io import MarketAgentInput
from schemas.phase1_state import (
    Phase1Input,
    Phase1Output,
    Phase1State,
    assert_phase1_immutable_fields,
)


logger = get_logger("phase1_market")

MAX_RETRIES = 2


async def market_agent_node(state: Phase1State) -> dict:
    """Run the market agent."""

    agent_input: MarketAgentInput = {
        "user_request": state["user_request"],
        "review_feedback": state.get("review_1_feedback", ""),
        "retry_count": state.get("retry_count", 0),
    }
    result = await run_market_agent(agent_input)
    update = {
        "market_result": result,
        "retry_count": state.get("retry_count", 0) + 1,
    }
    if result.get("market_context"):
        update["market_context"] = result["market_context"]
    assert_phase1_immutable_fields(update, "market_agent_node")
    return update


def hitl_1_node(state: Phase1State) -> dict:
    """Pause for Human Review #1."""

    market_result = state.get("market_result", {})
    resume_value = interrupt(
        {
            "phase": "review_1",
            "market_result": _build_review_context(market_result, state.get("retry_count", 0)),
            "allowed_decisions": _allowed_decisions(state),
            "retry_limit_reached": state.get("retry_count", 0) >= MAX_RETRIES,
        }
    )

    decision = resume_value.get("decision", "approve")
    feedback = resume_value.get("feedback", "")
    update = {
        "review_1_decision": decision,
        "review_1_feedback": feedback,
    }
    assert_phase1_immutable_fields(update, "hitl_1_node")
    return update


def error_handler_node(state: Phase1State) -> dict:
    """Record workflow termination errors."""

    logger.node_enter("phase1_error_handler", {"retry_count": state.get("retry_count", 0)})
    update = {
        "error_log": [
            {
                "event": "phase1_terminated",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
    }
    assert_phase1_immutable_fields(update, "error_handler_node")
    return update


def _route_after_market(state: Phase1State) -> str:
    return "hitl_1_node"


def _route_after_hitl_1(state: Phase1State) -> str:
    decision = state.get("review_1_decision", "approve")
    if decision == "approve":
        return END
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return END
    return "market_agent_node"


def build_graph() -> StateGraph:
    """Build the Phase 1 graph."""

    builder = StateGraph(Phase1State)
    builder.add_node("market_agent_node", market_agent_node)
    builder.add_node("hitl_1_node", hitl_1_node)
    builder.add_node("error_handler", error_handler_node)

    builder.add_edge(START, "market_agent_node")
    builder.add_conditional_edges(
        "market_agent_node",
        _route_after_market,
        {"hitl_1_node": "hitl_1_node"},
    )
    builder.add_conditional_edges(
        "hitl_1_node",
        _route_after_hitl_1,
        ["market_agent_node", END],
    )
    return builder


def make_checkpointer() -> MemorySaver:
    """Create a checkpointer for Phase 1."""

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("schemas.agent_io", "AgentStatus"),
            ("schemas.agent_io", "AgentFailureType"),
            ("schemas.agent_io", "SourceType"),
        ]
    )
    return MemorySaver(serde=serde)


def compile_standalone():
    """Compile a standalone Phase 1 app."""

    return build_graph().compile(checkpointer=make_checkpointer())


async def run(input: Phase1Input, config: RunnableConfig, app=None) -> Phase1Output:
    """Run Phase 1 and return orchestrator-ready output."""

    if app is None:
        app = compile_standalone()

    initial: dict = {
        **input,
        "retry_count": 0,
        "review_1_feedback": "",
        "error_log": [],
    }
    await app.ainvoke(initial, config)

    final = app.get_state(config).values
    market_context = final.get("market_context")
    if not market_context:
        raise RuntimeError("Market Agent가 유효한 market_context를 만들지 못했습니다.")

    return Phase1Output(
        user_request=final["user_request"],
        market_context=market_context,
    )


def _build_review_context(market_result: dict, retry_count: int) -> dict:
    """Build the HITL payload for review."""

    market_context = market_result.get("market_context", {})
    tool_log = market_result.get("tool_call_log", [])
    rag_calls = sum(1 for t in tool_log if "rag" in str(t.get("tool", "")).lower())
    web_calls = sum(1 for t in tool_log if "web" in str(t.get("tool", "")).lower())

    section_quality = {}
    for key in ["ev_growth_slowdown", "market_share_ranking", "lfp_ncm_trend",
                "ess_hev_growth", "regulatory_status", "cost_competitiveness"]:
        section = market_context.get(key, {})
        char_count = len(" ".join(str(v) for v in section.values())) if isinstance(section, dict) else 0
        section_quality[key] = {
            "char_count": char_count,
            "sufficient": char_count >= 600,
        }

    return {
        "status": market_result.get("status"),
        "failure_type": market_result.get("failure_type"),
        "retry_count": retry_count,
        "max_retries": MAX_RETRIES,
        "tool_usage": {"rag_calls": rag_calls, "web_calls": web_calls},
        "section_quality": section_quality,
        "thin_sections": [k for k, v in section_quality.items() if not v["sufficient"]],
        "market_context": market_context,
        "sources": market_result.get("sources", []),
    }


def _allowed_decisions(state: Phase1State) -> list[str]:
    decisions = ["approve"]
    if state.get("retry_count", 0) < MAX_RETRIES:
        decisions.append("redo")
    return decisions


async def debug_interrupt_once(user_request: str) -> dict:
    """Run Phase 1 until the first interrupt and return its payload."""

    app = compile_standalone()
    config = {"configurable": {"thread_id": "phase1_debug"}}
    await app.ainvoke(
        {
            "user_request": user_request,
            "retry_count": 0,
            "review_1_feedback": "",
            "error_log": [],
        },
        config=config,
    )
    state = app.get_state(config)
    interrupts = getattr(state, "interrupts", ()) or ()
    if not interrupts:
        return {"interrupted": False, "payload": None}
    payload = getattr(interrupts[0], "value", None)
    return {"interrupted": True, "payload": payload}


async def debug_resume_once(user_request: str, decision: str, feedback: str = "") -> dict:
    """Resume one HITL decision and return the resulting state snapshot."""

    app = compile_standalone()
    config = {"configurable": {"thread_id": "phase1_debug_resume"}}
    await app.ainvoke(
        {
            "user_request": user_request,
            "retry_count": 0,
            "review_1_feedback": "",
            "error_log": [],
        },
        config=config,
    )
    await app.ainvoke(Command(resume={"decision": decision, "feedback": feedback}), config=config)
    return app.get_state(config).values
