"""
Strategy Agent 통합 템플릿.

SKON/CATL을 단일 함수로 처리.
state["company"] 값으로 프롬프트와 RAG tool이 자동 선택됨.

프롬프트 수정: prompts/strategy_prompt.py
회사 추가:    prompts/strategy_prompt.py의 COMPANY_PROMPTS
              + _RAG_TOOL_FACTORY에 항목 추가
"""

from __future__ import annotations
import asyncio
import json
from datetime import datetime, timezone
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from prompts import build_system_prompt, COMPANY_PROMPTS
from schemas.agent_io import (
    AgentFailureType,
    AgentStatus,
    ConfidenceScores,
    SCHEMA_VERSION,
    StrategyAgentInput,
    StrategyAgentOutput,
)
from schemas.confidence import calculate_confidence_scores, AXIS_FIELDS
from tools.rag_tool import make_skon_rag_tool, make_catl_rag_tool
from tools.web_search_tool import web_search
from logging_utils import get_logger


logger = get_logger("strategy_agent")

MAX_ITERATIONS = 10

_RAG_TOOL_FACTORY = {
    "SKON": make_skon_rag_tool,
    "CATL": make_catl_rag_tool,
}


# ─────────────────────────────────────────────
# 실행 로직
# ─────────────────────────────────────────────

async def run_strategy_agent(state: StrategyAgentInput) -> StrategyAgentOutput:
    """
    SKON/CATL 전략 에이전트 공통 실행 함수.
    state["company"]로 프롬프트와 RAG tool이 자동 선택된다.
    """
    company   = state["company"]
    node_name = f"{company.lower()}_strategy_agent"

    with logger.node_span(node_name, {"retry": state.get("retry_count", 0)}) as span:
        span["llm_calls"]  = 0
        span["tool_calls"] = []

        try:
            rag_tool = _RAG_TOOL_FACTORY[company]()
            llm      = ChatOpenAI(model="gpt-4o", temperature=0)
            agent    = create_agent(
                model=llm,
                tools=[rag_tool, web_search],
                system_prompt=build_system_prompt(state),
            )

            result = await agent.ainvoke(
                {"messages": [{"role": "user",
                               "content": f"{COMPANY_PROMPTS[company].company_name} 전략 분석을 시작하세요."}]},
                config={
                    "recursion_limit": MAX_ITERATIONS + 5,
                    "run_name":  f"{node_name}_retry{state.get('retry_count', 0)}",
                    "tags":      ["battery_strategy", company.lower(), "strategy_agent"],
                    "metadata":  {"company": company, "retry_count": state.get("retry_count", 0)},
                },
            )

            last_msg = result["messages"][-1].content
            parsed   = _parse_output(last_msg)
            if parsed is None:
                return _make_failed(company, AgentFailureType.SCHEMA_ERROR)

            findings = {ax: parsed.get(ax) for ax in AXIS_FIELDS}
            sources  = parsed.get("sources", [])

            filled = [ax for ax in AXIS_FIELDS if findings.get(ax)]
            if len(filled) == 0:
                status = AgentStatus.FAILED
            elif len(filled) == len(AXIS_FIELDS):
                status = AgentStatus.SUCCESS
            else:
                status = AgentStatus.PARTIAL_SUCCESS

            span["status"]      = status.value
            span["filled_axes"] = len(filled)

            return StrategyAgentOutput(
                schema_version=SCHEMA_VERSION,
                company=company,
                status=status,
                failure_type=None,
                analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                llm_call_count=span["llm_calls"],
                tool_call_log=span["tool_calls"],
                **{ax: findings[ax] for ax in AXIS_FIELDS if findings.get(ax)},
                sources=sources,
                confidence_scores=calculate_confidence_scores(findings, sources),
            )

        except asyncio.TimeoutError:
            return _make_failed(company, AgentFailureType.TIMEOUT)
        except RecursionError:
            return _make_failed(company, AgentFailureType.MAX_ITER)
        except Exception as e:
            logger.error(node_name, e)
            return _make_failed(company, AgentFailureType.LLM_ERROR)


# ─────────────────────────────────────────────
# 내부 유틸
# ─────────────────────────────────────────────

def _parse_output(content: str) -> dict | None:
    try:
        start = content.find("{")
        end   = content.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        return json.loads(content[start:end])
    except json.JSONDecodeError:
        return None


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
        confidence_scores=ConfidenceScores(
            ev_response=0, market_position=0, tech_portfolio=0,
            ess_strategy=0, regulatory_risk=0, cost_structure=0, overall=0.0,
        ),
    )
