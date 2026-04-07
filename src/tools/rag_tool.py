"""
RAG Tool 전역 초기화 및 @tool 래핑.

공통 shared FAISS 인덱스를 한 번만 로드하고,
에이전트별 RAGPipeline이 메타데이터 필터로 다른 문서만 검색합니다.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool

from logging_utils import get_logger
from rag.config import RAGConfig
from rag.vectorstore import load_index
from tools.rag_pipeline import RAGPipeline, RAGResult


load_dotenv()

logger = get_logger("rag_tool")

_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_SKON_RAG: RAGPipeline | None = None
_CATL_RAG: RAGPipeline | None = None
_MARKET_RAG: RAGPipeline | None = None


def initialize_rag_pipelines_with_store(shared_vectorstore) -> None:
    """Initialize all RAG pipelines from one shared vectorstore."""

    global _SKON_RAG, _CATL_RAG, _MARKET_RAG

    _SKON_RAG = RAGPipeline(vectorstore=shared_vectorstore, collection_name="skon_agent")
    _CATL_RAG = RAGPipeline(vectorstore=shared_vectorstore, collection_name="catl_agent")
    _MARKET_RAG = RAGPipeline(vectorstore=shared_vectorstore, collection_name="market_agent")

    logger.node_exit(
        "init_with_store",
        duration_sec=0,
        status="ok",
        metadata={"loaded": ["shared_index", "skon_agent", "catl_agent", "market_agent"]},
    )


def initialize_rag_pipelines_with_stores(
    *,
    skon_vectorstore=None,
    catl_vectorstore=None,
    market_vectorstore=None,
    shared_vectorstore=None,
) -> None:
    """Backward-compatible alias for the old initializer name."""

    store = shared_vectorstore or market_vectorstore or skon_vectorstore or catl_vectorstore
    if store is None:
        raise ValueError("At least one vectorstore must be provided.")
    initialize_rag_pipelines_with_store(store)


def initialize_rag_pipelines() -> None:
    """Load the shared FAISS index and initialize all RAG pipelines."""

    project_root = Path(os.environ.get("PROJECT_ROOT", str(_DEFAULT_PROJECT_ROOT)))
    index_dir = Path(os.environ.get("INDEX_DIR", "data/vectorstores"))
    config = RAGConfig(index_dir=index_dir)

    logger.node_enter("global_init", {"step": "loading shared FAISS index"})
    shared_vs = load_index(config, project_root)
    initialize_rag_pipelines_with_store(shared_vs)


def _get_skon_rag() -> RAGPipeline:
    if _SKON_RAG is None:
        raise RuntimeError(
            "SKON RAG pipeline이 초기화되지 않았습니다. "
            "initialize_rag_pipelines()를 먼저 호출하세요."
        )
    return _SKON_RAG


def _get_catl_rag() -> RAGPipeline:
    if _CATL_RAG is None:
        raise RuntimeError(
            "CATL RAG pipeline이 초기화되지 않았습니다. "
            "initialize_rag_pipelines()를 먼저 호출하세요."
        )
    return _CATL_RAG


def _get_market_rag() -> RAGPipeline:
    if _MARKET_RAG is None:
        raise RuntimeError(
            "Market RAG pipeline이 초기화되지 않았습니다. "
            "initialize_rag_pipelines()를 먼저 호출하세요."
        )
    return _MARKET_RAG


def _format_rag_result(result: RAGResult) -> str:
    """Render RAG hits in a source-aware text format."""

    if not result.documents:
        return "관련 문서를 찾지 못했습니다."

    lines = [
        f"[RAG 검색 결과] 쿼리: {result.query_used} "
        f"(재작성 {result.rewrite_count}회"
        + (", 강제 반환" if result.forced_return else "")
        + ")\n"
    ]
    for i, doc in enumerate(result.documents, 1):
        block = (
            f"[{i}] source_id: rag_{doc.doc_id}\n"
            f"title: {doc.source_title}\n"
            f"page: {doc.page}\n"
            f"source_type: {result.source_type.value}\n"
            f"관련성: {doc.cosine_score:.3f}\n"
            f"출처: {doc.source_url}\n"
        )
        if doc.reference_text:
            block += f"REFERENCE: {doc.reference_text}\n"
        block += f"{doc.content}\n"
        lines.append(block)
    return "\n".join(lines)


def make_skon_rag_tool():
    @tool
    async def agentic_rag_skon(query: str) -> str:
        """
        SK On 관련 내부 PDF 문서에서 정보를 검색합니다.
        재무 데이터, 기술 로드맵, 전략 문서 등 상세 정보 조회 시 우선 사용하세요.
        """

        logger.tool_call("agentic_rag_skon", query=query)
        result = await _get_skon_rag().run(query)
        logger.tool_result(
            "agentic_rag_skon",
            success=bool(result.documents),
            metadata={
                "rewrite_count": result.rewrite_count,
                "forced_return": result.forced_return,
                "doc_count": len(result.documents),
            },
        )
        return _format_rag_result(result)

    return agentic_rag_skon


def make_market_rag_tool():
    @tool
    async def market_agent_rag(query: str) -> str:
        """
        시장 리포트 PDF에서 정보를 검색합니다.
        EV 성장률, 점유율, 기술 트렌드, 규제, 원가, ESS/HEV 전망 조회에 우선 사용하세요.
        """

        logger.tool_call("market_agent_rag", query=query)
        result = await _get_market_rag().run(query)
        logger.tool_result(
            "market_agent_rag",
            success=bool(result.documents),
            metadata={
                "rewrite_count": result.rewrite_count,
                "forced_return": result.forced_return,
                "doc_count": len(result.documents),
            },
        )
        return _format_rag_result(result)

    return market_agent_rag


def make_catl_rag_tool():
    @tool
    async def agentic_rag_catl(query: str) -> str:
        """
        CATL 관련 내부 PDF 문서에서 정보를 검색합니다.
        재무 데이터, 기술 로드맵, 전략 문서 등 상세 정보 조회 시 우선 사용하세요.
        """

        logger.tool_call("agentic_rag_catl", query=query)
        result = await _get_catl_rag().run(query)
        logger.tool_result(
            "agentic_rag_catl",
            success=bool(result.documents),
            metadata={
                "rewrite_count": result.rewrite_count,
                "forced_return": result.forced_return,
                "doc_count": len(result.documents),
            },
        )
        return _format_rag_result(result)

    return agentic_rag_catl
