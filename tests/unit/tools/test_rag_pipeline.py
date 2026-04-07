"""
tools/rag_pipeline.py 단위 테스트.

FAISS, embedder, LLM을 mock으로 대체해 외부 의존성 없이
RAGPipeline 내부 로직만 검증.

검증 항목:
  RAGPipeline.run():
    - threshold 충족 시 즉시 반환 (rewrite_count=0)
    - threshold 미달 시 쿼리 재작성 호출
    - 재작성 후 성공 → source_type=RAG_REWRITTEN
    - max_rewrites 초과 시 강제 반환 (forced_return=True)
    - 강제 반환 시에도 documents가 존재 (best_result 보존)

  RAGResult.to_source_records():
    - source_id 형식 ("rag_{doc_id}")
    - 문서 수 = records 수
    - credibility_score 평가됨 (0 or 1)
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from tools.rag_pipeline import RAGPipeline, RAGDocument, RAGResult
from schemas.agent_io import SourceType


# ─────────────────────────────────────────────
# 픽스처
# ─────────────────────────────────────────────

@pytest.fixture
def mock_embedder():
    """정규화된 고정 벡터를 반환하는 mock embedder"""
    embedder = MagicMock()
    vec = np.ones(768, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    embedder.embed_query.return_value = vec.tolist()
    return embedder


@pytest.fixture
def high_score_faiss():
    """cosine similarity 0.90 반환 — threshold(0.75) 충족"""
    idx = MagicMock()
    idx.search.return_value = (
        np.array([[0.90, 0.85, 0.80, 0.75, 0.70]]),
        np.array([[0, 1, 2, 3, 4]]),
    )
    return idx


@pytest.fixture
def low_score_faiss():
    """cosine similarity 0.50 반환 — threshold 미달"""
    idx = MagicMock()
    idx.search.return_value = (
        np.array([[0.50, 0.45, 0.40, 0.35, 0.30]]),
        np.array([[0, 1, 2, 3, 4]]),
    )
    return idx


@pytest.fixture
def sample_documents():
    return [
        {"id": f"doc_{i}", "content": f"문서 내용 {i}",
         "url": f"https://src{i}.com", "title": f"제목 {i}"}
        for i in range(5)
    ]


def _make_pipeline(faiss_index, embedder, docs, max_rewrites=2):
    return RAGPipeline(
        faiss_index=faiss_index,
        documents=docs,
        embedder=embedder,
        relevance_threshold=0.75,
        max_rewrites=max_rewrites,
    )


# ─────────────────────────────────────────────
# 정상 검색 (threshold 충족)
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_returns_immediately_on_high_score(
    high_score_faiss, mock_embedder, sample_documents
):
    pipeline = _make_pipeline(high_score_faiss, mock_embedder, sample_documents)
    result = await pipeline.run("SK On ESS 전략")
    assert result.rewrite_count == 0
    assert result.forced_return is False
    assert result.source_type == SourceType.RAG_FAISS
    assert len(result.documents) > 0


@pytest.mark.asyncio
async def test_run_source_type_rag_faiss_on_first_hit(
    high_score_faiss, mock_embedder, sample_documents
):
    pipeline = _make_pipeline(high_score_faiss, mock_embedder, sample_documents)
    result = await pipeline.run("쿼리")
    assert result.source_type == SourceType.RAG_FAISS


# ─────────────────────────────────────────────
# 쿼리 재작성 루프
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_calls_rewrite_on_low_score(
    low_score_faiss, mock_embedder, sample_documents
):
    pipeline = _make_pipeline(
        low_score_faiss, mock_embedder, sample_documents, max_rewrites=1
    )
    with patch.object(pipeline, "_rewrite_query",
                      new=AsyncMock(return_value="재작성된 쿼리")) as mock_rw:
        await pipeline.run("원본 쿼리")
    assert mock_rw.call_count == 1


@pytest.mark.asyncio
async def test_run_source_type_rag_rewritten_after_rewrite(
    low_score_faiss, mock_embedder, sample_documents
):
    pipeline = _make_pipeline(
        low_score_faiss, mock_embedder, sample_documents, max_rewrites=1
    )
    with patch.object(pipeline, "_rewrite_query",
                      new=AsyncMock(return_value="재작성")):
        result = await pipeline.run("쿼리")
    assert result.source_type == SourceType.RAG_REWRITTEN


@pytest.mark.asyncio
async def test_run_rewrite_called_exactly_max_rewrites_times(
    low_score_faiss, mock_embedder, sample_documents
):
    pipeline = _make_pipeline(
        low_score_faiss, mock_embedder, sample_documents, max_rewrites=2
    )
    with patch.object(pipeline, "_rewrite_query",
                      new=AsyncMock(return_value="재작성")) as mock_rw:
        result = await pipeline.run("쿼리")
    assert mock_rw.call_count == 2
    assert result.forced_return is True


@pytest.mark.asyncio
async def test_run_forced_return_preserves_best_result(
    low_score_faiss, mock_embedder, sample_documents
):
    """강제 반환 시 documents가 비어있지 않아야 함"""
    pipeline = _make_pipeline(
        low_score_faiss, mock_embedder, sample_documents, max_rewrites=1
    )
    with patch.object(pipeline, "_rewrite_query",
                      new=AsyncMock(return_value="재작성")):
        result = await pipeline.run("쿼리")
    assert result.forced_return is True
    assert len(result.documents) > 0


# ─────────────────────────────────────────────
# RAGResult.to_source_records
# ─────────────────────────────────────────────

def test_to_source_records_length():
    result = RAGResult(
        documents=[
            RAGDocument("d1", "내용1", "https://a.com", "제목1", 0.85),
            RAGDocument("d2", "내용2", "https://b.com", "제목2", 0.70),
        ],
        query_used="쿼리",
        rewrite_count=0,
        source_type=SourceType.RAG_FAISS,
    )
    records = result.to_source_records()
    assert len(records) == 2


def test_to_source_records_source_id_format():
    result = RAGResult(
        documents=[RAGDocument("abc123", "내용", "https://x.com", "제목", 0.80)],
        query_used="쿼리",
        rewrite_count=0,
        source_type=SourceType.RAG_FAISS,
    )
    records = result.to_source_records()
    assert records[0]["source_id"] == "rag_abc123"


def test_to_source_records_credibility_is_binary():
    result = RAGResult(
        documents=[RAGDocument("d1", "내용", "https://a.com", "제목", 0.90)],
        query_used="쿼리",
        rewrite_count=0,
        source_type=SourceType.RAG_FAISS,
    )
    records = result.to_source_records()
    assert records[0]["credibility_score"] in (0, 1)


def test_to_source_records_empty_documents():
    result = RAGResult(
        documents=[],
        query_used="쿼리",
        rewrite_count=0,
    )
    assert result.to_source_records() == []


# ─────────────────────────────────────────────
# RAGPipeline 초기화
# ─────────────────────────────────────────────

def test_pipeline_lazy_llm_initialization(mock_embedder, high_score_faiss, sample_documents):
    """LLM은 실제 쿼리 재작성 시점에 초기화돼야 함"""
    pipeline = _make_pipeline(high_score_faiss, mock_embedder, sample_documents)
    assert pipeline._rewrite_llm is None


def test_pipeline_default_params(mock_embedder, high_score_faiss, sample_documents):
    pipeline = RAGPipeline(
        faiss_index=high_score_faiss,
        documents=sample_documents,
        embedder=mock_embedder,
    )
    assert pipeline.threshold == 0.75
    assert pipeline.max_rewrites == 3
    assert pipeline.top_k == 5
