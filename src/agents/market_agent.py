"""
Market agent runner.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from prompts.market_prompt import build_market_system_prompt
from rag.source_metadata import resolve_source_metadata
from schemas.agent_io import AgentFailureType, AgentStatus
from schemas.market_agent_io import MarketAgentInput, MarketAgentOutput
from tools.rag_tool import make_market_rag_tool
from tools.web_search_tool import web_search
from logging_utils import get_logger


logger = get_logger("market_agent")

MAX_ITERATIONS = 20  # RAG 6회 + web_search 6회 + 여유
REQUIRED_MARKET_KEYS = [
    "ev_growth_slowdown",
    "market_share_ranking",
    "lfp_ncm_trend",
    "ess_hev_growth",
    "regulatory_status",
    "cost_competitiveness",
]

MIN_SECTION_CHARS = 600  # 각 축 최소 글자 수


async def run_market_agent(state: MarketAgentInput) -> MarketAgentOutput:
    """Run the market agent and return structured market context."""

    with logger.node_span("market_agent", {"retry": state.get("retry_count", 0)}) as span:
        span["llm_calls"] = 0
        span["tool_calls"] = []

        try:
            rag_tool = make_market_rag_tool()
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            agent = create_agent(
                model=llm,
                tools=[rag_tool, web_search],
                system_prompt=build_market_system_prompt(state),
            )

            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "시장 분석을 시작하세요."}]},
                config={
                    "recursion_limit": MAX_ITERATIONS + 5,
                    "run_name": f"market_agent_retry{state.get('retry_count', 0)}",
                    "tags": ["battery_strategy", "market_agent"],
                    "metadata": {"retry_count": state.get("retry_count", 0)},
                },
            )

            parsed = _parse_output(result["messages"][-1].content)
            if parsed is None:
                return _make_failed(AgentFailureType.SCHEMA_ERROR)

            market_context = {
                key: parsed.get(key, {})
                for key in REQUIRED_MARKET_KEYS
            }
            sources = _normalize_source_records(parsed.get("source_records", []) or parsed.get("sources", []))
            references = _normalize_references(parsed.get("references", []), sources)
            market_context["source_records"] = sources
            market_context["references"] = references

            for key in REQUIRED_MARKET_KEYS:
                section = market_context.get(key)
                if isinstance(section, dict):
                    section.setdefault("source_ids", [])

            filled = [key for key in REQUIRED_MARKET_KEYS if market_context.get(key)]
            thin = _find_thin_sections(market_context)

            if len(filled) == 0:
                status = AgentStatus.FAILED
            elif len(filled) == len(REQUIRED_MARKET_KEYS) and not thin:
                status = AgentStatus.SUCCESS
            else:
                status = AgentStatus.PARTIAL_SUCCESS

            span["status"] = status.value
            span["filled_sections"] = len(filled)
            span["thin_sections"] = thin

            return MarketAgentOutput(
                status=status,
                failure_type=None,
                analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                llm_call_count=span["llm_calls"],
                tool_call_log=span["tool_calls"],
                market_context=market_context,
                sources=sources,
            )

        except asyncio.TimeoutError:
            return _make_failed(AgentFailureType.TIMEOUT)
        except RecursionError:
            return _make_failed(AgentFailureType.MAX_ITER)
        except Exception as e:
            logger.error("market_agent", e)
            return _make_failed(AgentFailureType.LLM_ERROR)


def _find_thin_sections(market_context: dict) -> list[str]:
    """600자 미만인 축 목록 반환 — HITL 품질 지표로 사용."""
    thin = []
    for key in REQUIRED_MARKET_KEYS:
        section = market_context.get(key)
        if not isinstance(section, dict):
            thin.append(key)
            continue
        text = " ".join(str(v) for v in section.values())
        if len(text) < MIN_SECTION_CHARS:
            thin.append(key)
    return thin


def _parse_output(content: str) -> dict | None:
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        return json.loads(content[start:end])
    except json.JSONDecodeError:
        return None


def _normalize_references(references: list[dict] | list[str], sources: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    source_map = {
        str(source.get("source_id", "")): source
        for source in sources
        if source.get("source_id")
    }

    for item in references:
        if isinstance(item, str):
            normalized.append({"source_id": "", "formatted_reference": item})
            continue
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id", ""))
        source = source_map.get(source_id, {})
        resolved = resolve_source_metadata(
            source_id=source_id,
            source=str(source.get("source_name", "")),
            title=str(source.get("title", "")),
        )
        formatted_reference = str(item.get("formatted_reference", "")).strip()
        if _should_replace_reference(formatted_reference):
            formatted_reference = resolved["reference_text"] or formatted_reference
        normalized.append(
            {
                "source_id": source_id,
                "formatted_reference": formatted_reference,
            }
        )

    if normalized:
        return _deduplicate_references(normalized)

    fallback: list[dict] = []
    for source_id, source in source_map.items():
        resolved = resolve_source_metadata(
            source_id=source_id,
            source=str(source.get("source_name", "")),
            title=str(source.get("title", "")),
        )
        title = str(source.get("title", "")).strip() or resolved["title"]
        url = str(source.get("url", "")).strip() or resolved["url"]
        retrieved_at = str(source.get("retrieved_at", "")).strip()
        source_type = str(source.get("source_type", "")).strip()
        reference_text = resolved["reference_text"]
        if reference_text:
            fallback.append(
                {
                    "source_id": source_id,
                    "formatted_reference": reference_text,
                }
            )
            continue
        if source_type == "web" and retrieved_at:
            fallback.append(
                {
                    "source_id": source_id,
                    "formatted_reference": f"{title or 'Web Source'}({retrieved_at[:10]}). *{title or 'Web Source'}*. Web, {url}",
                }
            )
        else:
            fallback.append(
                {
                    "source_id": source_id,
                    "formatted_reference": f"{title or 'Report'}({retrieved_at[:4] if retrieved_at else 'n.d.'}). *{title or 'Report'}*. {url}",
                }
            )
    return _deduplicate_references(fallback)


def _normalize_source_records(raw_sources: list[dict] | list[str]) -> list[dict]:
    normalized: list[dict] = []
    for item in raw_sources:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id", "")).strip()
        raw_title = str(item.get("title", "")).strip()
        resolved = resolve_source_metadata(
            source_id=source_id,
            source=str(item.get("source_name", "")).strip(),
            title=raw_title,
        )
        title = _normalize_title(raw_title, resolved["source_name"], resolved["title"])
        normalized.append(
            {
                "source_id": source_id,
                "url": str(item.get("url", "")).strip() or resolved["url"],
                "title": title,
                "retrieved_at": str(item.get("retrieved_at", "")).strip() or datetime.now(timezone.utc).isoformat(),
                "source_type": item.get("source_type", ""),
                "credibility_score": item.get("credibility_score", 0),
                "credibility_flags": item.get("credibility_flags", {}),
                "source_name": resolved["source_name"],
            }
        )
    return [item for item in normalized if item["source_id"]]


def _normalize_title(raw_title: str, source_name: str, canonical_title: str) -> str:
    """Choose a human-readable title instead of a raw filename."""

    if not raw_title:
        return canonical_title
    normalized_title = raw_title.strip().lower()
    if normalized_title.endswith(".pdf"):
        return canonical_title
    if source_name and normalized_title == source_name.strip().lower():
        return canonical_title
    return raw_title


def _should_replace_reference(formatted_reference: str) -> bool:
    """Detect placeholder-like references and replace them with canonical metadata."""

    if not formatted_reference:
        return True
    lowered = formatted_reference.lower()
    return "internal database" in lowered or lowered.startswith("market report(") or lowered.startswith("analyst report(")


def _deduplicate_references(references: list[dict]) -> list[dict]:
    """Remove duplicate references while preserving order."""

    seen: set[tuple[str, str]] = set()
    deduplicated: list[dict] = []
    for item in references:
        key = (
            str(item.get("source_id", "")),
            str(item.get("formatted_reference", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)
    return deduplicated


def _make_failed(failure_type: AgentFailureType) -> MarketAgentOutput:
    return MarketAgentOutput(
        status=AgentStatus.FAILED,
        failure_type=failure_type,
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        llm_call_count=0,
        tool_call_log=[],
        market_context={},
        sources=[],
    )
