"""
RAG Tool 전역 초기화 및 @tool 래핑.

초기화 전략:
  - src/rag/vectorstore.load_index() 로 LangChain FAISS 인덱스 로드
    → src/rag/vectorstore.build_and_save_indices() 로 빌드한 인덱스와 포맷 일치
  - 인덱스 경로: {INDEX_DIR}/{collection_name}/  (LangChain FAISS 폴더 구조)
  - market/skon/catl 컬렉션을 전역 파이프라인으로 올려 재사용
  - 테스트에서는 initialize_rag_pipelines() 호출 않고 tool factory를 mock

환경변수 (.env):
  INDEX_DIR        : 인덱스 루트 폴더 (기본: data/vectorstores)
  PROJECT_ROOT     : 프로젝트 루트 경로 (기본: rag_tool.py 기준 두 단계 상위)
"""

from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool

from rag.config import RAGConfig
from rag.vectorstore import load_index
from tools.rag_pipeline import RAGPipeline, RAGResult
from logging_utils import get_logger


logger = get_logger("rag_tool")

# rag_tool.py 위치: src/tools/rag_tool.py
# 프로젝트 루트:    src/tools/ → src/ → project_root
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ─────────────────────────────────────────────
# 전역 파이프라인 레지스트리
# ─────────────────────────────────────────────

_SKON_RAG:   RAGPipeline | None = None
_CATL_RAG:   RAGPipeline | None = None
_MARKET_RAG: RAGPipeline | None = None


def initialize_rag_pipelines_with_stores(
    skon_vectorstore,
    catl_vectorstore,
    market_vectorstore=None,
) -> None:
    """
    app.py에서 미리 로드한 FAISS 인스턴스를 주입해 전역 파이프라인 초기화.
    vectorstore를 한 번만 로드해 공유하는 권장 방식.

    Args:
        skon_vectorstore   : skon_agent FAISS (LangChain FAISS 인스턴스)
        catl_vectorstore   : catl_agent FAISS
        market_vectorstore : market_agent FAISS (선택)
    """
    global _SKON_RAG, _CATL_RAG, _MARKET_RAG

    _SKON_RAG = RAGPipeline(vectorstore=skon_vectorstore, collection_name="skon_agent")
    _CATL_RAG = RAGPipeline(vectorstore=catl_vectorstore, collection_name="catl_agent")
    if market_vectorstore is not None:
        _MARKET_RAG = RAGPipeline(vectorstore=market_vectorstore, collection_name="market_agent")

    loaded = ["skon_agent", "catl_agent"] + (["market_agent"] if market_vectorstore else [])
    logger.node_exit("init_with_stores", duration_sec=0, status="ok",
                     metadata={"loaded": loaded})


def initialize_rag_pipelines() -> None:
    """
    FAISS 인덱스를 내부에서 직접 로드해 전역 파이프라인 초기화.
    (테스트/단독 실행용. 앱 진입점에서는 initialize_rag_pipelines_with_stores 사용 권장)

    인덱스 위치:
      {INDEX_DIR}/skon_agent/    ← skon_agent 컬렉션
      {INDEX_DIR}/catl_agent/    ← catl_agent 컬렉션
      {INDEX_DIR}/market_agent/  ← market_agent 컬렉션

    빌드 방법 (app.py):
      python app.py build-indices
    """
    global _SKON_RAG, _CATL_RAG, _MARKET_RAG

    project_root = Path(os.environ.get("PROJECT_ROOT", str(_DEFAULT_PROJECT_ROOT)))
    index_dir    = Path(os.environ.get("INDEX_DIR", "data/vectorstores"))
    config       = RAGConfig(index_dir=index_dir)

    logger.node_enter("global_init", {"step": "loading FAISS indices"})
    skon_vs   = load_index(config, project_root, "skon_agent")
    catl_vs   = load_index(config, project_root, "catl_agent")
    market_vs = load_index(config, project_root, "market_agent")

    _SKON_RAG   = RAGPipeline(vectorstore=skon_vs,   collection_name="skon_agent")
    _CATL_RAG   = RAGPipeline(vectorstore=catl_vs,   collection_name="catl_agent")
    _MARKET_RAG = RAGPipeline(vectorstore=market_vs, collection_name="market_agent")

    logger.node_exit("global_init", duration_sec=0, status="ok",
                     metadata={"loaded": ["skon_agent", "catl_agent", "market_agent"]})


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
            "initialize_rag_pipelines_with_stores() 또는 initialize_rag_pipelines()를 먼저 호출하세요."
        )
    return _CATL_RAG


def _get_market_rag() -> RAGPipeline:
    if _MARKET_RAG is None:
        raise RuntimeError(
            "Market RAG pipeline이 초기화되지 않았습니다. "
            "initialize_rag_pipelines_with_stores() 또는 initialize_rag_pipelines()를 먼저 호출하세요."
        )
    return _MARKET_RAG


# ─────────────────────────────────────────────
# Tool 팩토리
# ─────────────────────────────────────────────

def _format_rag_result(result: RAGResult) -> str:
    """Render RAG hits in a source-aware text format."""

    if not result.documents:
        return "관련 문서를 찾지 못했습니다."
    lines = [
        f"[RAG 검색 결과] 쿼리: {result.query_used} "
        f"(재작성 {result.rewrite_count}회"
        + (", 강제 반환" if result.forced_return else "") + ")\n"
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
        lines.append(
            block
        )
    return "\n".join(lines)


def make_skon_rag_tool():
    @tool
    async def agentic_rag_skon(query: str) -> str:
        """
        SK On 관련 내부 PDF 문서에서 정보를 검색합니다.
        재무 데이터, 기술 로드맵, 전략 문서 등 상세 정보 조회 시 우선 사용하세요.
        web_search보다 먼저 호출하고, 결과가 부족할 때 web_search로 보완하세요.
        """
        logger.tool_call("agentic_rag_skon", query=query)
        result = await _get_skon_rag().run(query)
        logger.tool_result(
            "agentic_rag_skon",
            success=bool(result.documents),
            metadata={"rewrite_count": result.rewrite_count,
                      "forced_return": result.forced_return,
                      "doc_count": len(result.documents)},
        )
        return _format_rag_result(result)
    return agentic_rag_skon


def make_market_rag_tool():
    @tool
    async def market_agent_rag(query: str) -> str:
        """
        시장 리포트 PDF에서 정보를 검색합니다.
        EV 성장률, 점유율, 기술 트렌드, 규제, 원가, ESS/HEV 전망 조회에 우선 사용하세요.
        web_search보다 먼저 호출하고, 결과가 부족할 때 web_search로 보완하세요.
        """
        logger.tool_call("market_agent_rag", query=query)
        result = await _get_market_rag().run(query)
        logger.tool_result(
            "market_agent_rag",
            success=bool(result.documents),
            metadata={"rewrite_count": result.rewrite_count,
                      "forced_return": result.forced_return,
                      "doc_count": len(result.documents)},
        )
        return _format_rag_result(result)
    return market_agent_rag


def make_catl_rag_tool():
    @tool
    async def agentic_rag_catl(query: str) -> str:
        """
        CATL 관련 내부 PDF 문서에서 정보를 검색합니다.
        재무 데이터, 기술 로드맵, 전략 문서 등 상세 정보 조회 시 우선 사용하세요.
        web_search보다 먼저 호출하고, 결과가 부족할 때 web_search로 보완하세요.
        """
        logger.tool_call("agentic_rag_catl", query=query)
        result = await _get_catl_rag().run(query)
        logger.tool_result(
            "agentic_rag_catl",
            success=bool(result.documents),
            metadata={"rewrite_count": result.rewrite_count,
                      "forced_return": result.forced_return,
                      "doc_count": len(result.documents)},
        )
        return _format_rag_result(result)
    return agentic_rag_catl


def make_market_rag_tool():
    @tool
    async def agentic_rag_market(query: str) -> str:
        """
        배터리 시장 분석 관련 내부 PDF 문서에서 정보를 검색합니다.
        시장 성장률, 점유율, 기술 트렌드, 규제 현황 등 시장 배경 조사 시 우선 사용하세요.
        web_search보다 먼저 호출하고, 결과가 부족할 때 web_search로 보완하세요.
        """
        logger.tool_call("agentic_rag_market", query=query)
        result = await _get_market_rag().run(query)
        logger.tool_result(
            "agentic_rag_market",
            success=bool(result.documents),
            metadata={"rewrite_count": result.rewrite_count,
                      "forced_return": result.forced_return,
                      "doc_count": len(result.documents)},
        )
        return _format_rag_result(result)
    return agentic_rag_market
