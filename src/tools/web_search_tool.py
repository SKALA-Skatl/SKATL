"""
Tavily 기반 web_search tool.

agentic_rag로 충분하지 않을 때 보완용으로 호출.
Agent 시스템 프롬프트에 호출 우선순위 명시:
  1. agentic_rag 먼저
  2. 결과 부족 시 web_search로 보완
"""

from __future__ import annotations
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool

from schemas.agent_io import SourceRecord, SourceType
from schemas.confidence import evaluate_source_credibility
from logging_utils import get_logger


logger = get_logger("web_search_tool")

_TAVILY_CLIENT = None  # lazy 초기화 — web_search 첫 호출 시 생성


def _tavily_result_to_source_record(result: dict, idx: int) -> SourceRecord:
    raw = SourceRecord(
        source_id=f"web_{idx:03d}",
        url=result.get("url", ""),
        title=result.get("title", ""),
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        source_type=SourceType.WEB,
        credibility_score=0,
        credibility_flags={},
    )
    return evaluate_source_credibility(raw, rag_cosine_score=None)


@tool
async def web_search(query: str) -> str:
    """
    Tavily를 통해 최신 웹 정보를 검색합니다.
    agentic_rag로 충분한 정보를 얻지 못했을 때 보완용으로 사용하세요.
    최신 뉴스, 규제 동향, 공식 발표 등 실시간 정보에 적합합니다.
    """
    global _TAVILY_CLIENT
    if _TAVILY_CLIENT is None:
        from tavily import AsyncTavilyClient
        _TAVILY_CLIENT = AsyncTavilyClient()

    logger.tool_call("web_search", query=query)

    try:
        response = await _TAVILY_CLIENT.search(
            query=query,
            max_results=5,
            search_depth="advanced",
            include_raw_content=True,   # 원문 보존용
        )
        results = response.get("results", [])

        logger.tool_result(
            "web_search",
            success=bool(results),
            metadata={"result_count": len(results)},
        )

        if not results:
            return "검색 결과가 없습니다."

        lines = [f"[Web 검색 결과] 쿼리: {query}\n"]
        for i, r in enumerate(results, 1):
            source = _tavily_result_to_source_record(r, i)
            content = r.get("raw_content") or r.get("content", "")
            lines.append(
                f"[{i}] {r.get('title', '')} "
                f"(신뢰도: {source['credibility_score']})\n"
                f"출처: {r.get('url', '')}\n"
                f"{content}\n"
            )
        return "\n".join(lines)

    except Exception as e:
        logger.error("web_search", e)
        return f"검색 중 오류 발생: {str(e)}"
