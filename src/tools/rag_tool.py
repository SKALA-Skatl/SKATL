"""
RAG Tool 전역 초기화 및 @tool 래핑.

초기화 전략:
  - faiss, HuggingFaceEmbeddings는 initialize_rag_pipelines() 내부에서만 import
  - 테스트에서는 호출하지 않고 make_skon/catl_rag_tool을 mock
  - 애플리케이션 진입점에서 initialize_rag_pipelines() 명시적 호출

환경변수 (.env):
  SKON_FAISS_INDEX_PATH, SKON_DOCS_PATH
  CATL_FAISS_INDEX_PATH, CATL_DOCS_PATH
"""

from __future__ import annotations
import json
import os

from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool

from tools.rag_pipeline import RAGPipeline, RAGResult
from logging_utils import get_logger


logger = get_logger("rag_tool")


# ─────────────────────────────────────────────
# 전역 파이프라인 레지스트리
# ─────────────────────────────────────────────

_SKON_RAG: RAGPipeline | None = None
_CATL_RAG: RAGPipeline | None = None


def _load_documents(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def initialize_rag_pipelines() -> None:
    """
    FAISS 인덱스 + bge-m3 임베딩 모델을 로드해 전역 파이프라인 초기화.

    임베딩:
      langchain-huggingface의 HuggingFaceEmbeddings 사용.
      bge-m3 추론 전용 — langchain-huggingface의 HuggingFaceEmbeddings 사용.

    호출 시점:
      - 애플리케이션 진입점 (run_test.py, main.py 등)
      - Phase 1 진행 중 백그라운드에서 호출해 latency 제거

    테스트에서는 호출하지 않음 — make_skon/catl_rag_tool을 직접 mock.
    """
    global _SKON_RAG, _CATL_RAG

    import faiss
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.node_enter("global_init", {"step": "bge-m3 embedder (HuggingFaceEmbeddings)"})
    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.node_enter("global_init", {"step": "SKON FAISS index"})
    skon_index = faiss.read_index(
        os.environ.get("SKON_FAISS_INDEX_PATH", "./indices/skon.faiss")
    )
    skon_docs = _load_documents(
        os.environ.get("SKON_DOCS_PATH", "./indices/skon_docs.json")
    )

    logger.node_enter("global_init", {"step": "CATL FAISS index"})
    catl_index = faiss.read_index(
        os.environ.get("CATL_FAISS_INDEX_PATH", "./indices/catl.faiss")
    )
    catl_docs = _load_documents(
        os.environ.get("CATL_DOCS_PATH", "./indices/catl_docs.json")
    )

    _SKON_RAG = RAGPipeline(faiss_index=skon_index, documents=skon_docs, embedder=embedder)
    _CATL_RAG = RAGPipeline(faiss_index=catl_index, documents=catl_docs, embedder=embedder)

    logger.node_exit("global_init", duration_sec=0, status="ok",
                     metadata={"loaded": ["bge-m3", "skon_faiss", "catl_faiss"]})


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


# ─────────────────────────────────────────────
# Tool 팩토리
# ─────────────────────────────────────────────

def _format_rag_result(result: RAGResult) -> str:
    if not result.documents:
        return "관련 문서를 찾지 못했습니다."
    lines = [
        f"[RAG 검색 결과] 쿼리: {result.query_used} "
        f"(재작성 {result.rewrite_count}회"
        + (", 강제 반환" if result.forced_return else "") + ")\n"
    ]
    for i, doc in enumerate(result.documents, 1):
        lines.append(
            f"[{i}] {doc.source_title} (관련성: {doc.cosine_score:.3f})\n"
            f"출처: {doc.source_url}\n"
            f"{doc.content}\n"
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

