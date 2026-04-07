"""
RAG Pipeline 모듈.

구조:
  RAGPipeline.run(query) →
    1. bge-m3 임베딩 — HuggingFaceEmbeddings.embed_query() 사용
       (동기 blocking → run_in_executor로 비동기 래핑)
    2. FAISS 검색 (동기 → run_in_executor)
    3. 관련성 평가 (cosine similarity >= threshold)
    4. 미달 시 gpt-4o-mini로 쿼리 재작성 → 재검색
    5. max_rewrites 초과 시 최선 결과로 강제 반환

embedder 인터페이스:
  HuggingFaceEmbeddings (langchain-huggingface)
    - embed_query(text: str) -> list[float]
    - encode_kwargs={"normalize_embeddings": True} 로 초기화 시
      반환 벡터가 이미 L2 정규화된 상태 → 별도 정규화 불필요
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Any

import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from schemas.agent_io import SourceRecord, SourceType
from schemas.confidence import evaluate_source_credibility
from logging_utils import get_logger


logger = get_logger("rag_pipeline")


# ─────────────────────────────────────────────
# 결과 타입
# ─────────────────────────────────────────────

@dataclass
class RAGDocument:
    doc_id:       str
    content:      str
    source_url:   str
    source_title: str
    cosine_score: float


@dataclass
class RAGResult:
    documents:     list[RAGDocument]
    query_used:    str
    rewrite_count: int
    forced_return: bool = False
    source_type:   SourceType = SourceType.RAG_FAISS

    def to_source_records(self) -> list[SourceRecord]:
        records = []
        for doc in self.documents:
            raw = SourceRecord(
                source_id=f"rag_{doc.doc_id}",
                url=doc.source_url,
                title=doc.source_title,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                source_type=self.source_type,
                credibility_score=0,
                credibility_flags={},
            )
            records.append(evaluate_source_credibility(
                raw, rag_cosine_score=doc.cosine_score
            ))
        return records


# ─────────────────────────────────────────────
# RAGPipeline
# ─────────────────────────────────────────────

class RAGPipeline:
    """
    Args:
        faiss_index         : faiss.Index (IndexFlatIP 권장 — 정규화 벡터와 inner product = cosine)
        documents           : [{id, content, url, title}, ...] — 인덱스 순서와 일치
        embedder            : HuggingFaceEmbeddings 인스턴스
                              encode_kwargs={"normalize_embeddings": True} 필수
        relevance_threshold : cosine similarity 하한 (기본 0.75)
        max_rewrites        : 쿼리 재작성 최대 횟수 (기본 3)
        top_k               : FAISS 검색 결과 수 (기본 5)
    """

    def __init__(
        self,
        faiss_index: Any,
        documents: list[dict],
        embedder: Any,
        relevance_threshold: float = 0.75,
        max_rewrites: int = 3,
        top_k: int = 5,
    ):
        self.index        = faiss_index
        self.documents    = documents
        self.embedder     = embedder
        self.threshold    = relevance_threshold
        self.max_rewrites = max_rewrites
        self.top_k        = top_k

        # lazy 초기화 — 실제 호출 시점에 생성 (테스트 시 mock 주입 가능)
        self._rewrite_llm       = None
        self._rewrite_llm_model = "gpt-4o-mini"

    # ── 퍼블릭 API ────────────────────────────

    async def run(self, query: str) -> RAGResult:
        rewrite_count = 0
        current_query = query
        best_result: RAGResult | None = None

        while rewrite_count <= self.max_rewrites:
            docs, scores = await self._search(current_query)

            logger.rag_search(
                query=current_query,
                cosine_scores=scores,
                rewrite_count=rewrite_count,
            )

            if scores and max(scores) >= self.threshold:
                return RAGResult(
                    documents=docs,
                    query_used=current_query,
                    rewrite_count=rewrite_count,
                    source_type=(
                        SourceType.RAG_FAISS if rewrite_count == 0
                        else SourceType.RAG_REWRITTEN
                    ),
                )

            best_result = RAGResult(
                documents=docs,
                query_used=current_query,
                rewrite_count=rewrite_count,
                source_type=SourceType.RAG_REWRITTEN,
            )

            if rewrite_count >= self.max_rewrites:
                break

            current_query = await self._rewrite_query(current_query, docs)
            rewrite_count += 1

        # 강제 탈출: 최선의 결과 반환
        logger.rag_search(
            query=current_query,
            cosine_scores=[],
            rewrite_count=rewrite_count,
            forced_return=True,
        )
        result = best_result or RAGResult(
            documents=[], query_used=current_query, rewrite_count=rewrite_count,
        )
        result.forced_return = True
        return result

    # ── 내부 메서드 ───────────────────────────

    async def _embed(self, text: str) -> np.ndarray:
        """
        HuggingFaceEmbeddings.embed_query()는 동기 blocking.
        run_in_executor로 비동기 래핑.

        encode_kwargs={"normalize_embeddings": True} 로 초기화했으므로
        반환 벡터는 이미 L2 정규화된 상태. 별도 정규화 불필요.
        """
        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(
            None,
            self.embedder.embed_query,  # embed_query(text) → list[float]
            text,
        )
        return np.array(vec, dtype=np.float32)

    async def _search(self, query: str) -> tuple[list[RAGDocument], list[float]]:
        """FAISS 검색 (동기 blocking → run_in_executor)."""
        embedding = await self._embed(query)

        loop = asyncio.get_event_loop()
        fn = partial(self.index.search, embedding.reshape(1, -1), self.top_k)
        distances, indices = await loop.run_in_executor(None, fn)

        docs: list[RAGDocument] = []
        scores: list[float] = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            raw    = self.documents[idx]
            cosine = float(dist)  # IndexFlatIP + 정규화 벡터 → inner product = cosine
            docs.append(RAGDocument(
                doc_id=raw.get("id", str(idx)),
                content=raw["content"],
                source_url=raw.get("url", ""),
                source_title=raw.get("title", ""),
                cosine_score=cosine,
            ))
            scores.append(cosine)

        return docs, scores

    async def _rewrite_query(self, original_query: str, docs: list[RAGDocument]) -> str:
        """관련성 낮은 검색 결과를 참고해 쿼리 재작성."""
        context_preview = "\n".join(f"- {d.content[:200]}" for d in docs[:3])
        prompt = (
            f"다음 쿼리로 검색했지만 관련성이 낮은 결과가 나왔습니다.\n"
            f"원본 쿼리: {original_query}\n\n"
            f"검색된 문서 일부:\n{context_preview}\n\n"
            f"배터리 시장 전략 분석에 더 적합한 검색 쿼리 1개를 한국어로 작성하세요. "
            f"쿼리만 출력하고 다른 설명은 하지 마세요."
        )
        if self._rewrite_llm is None:
            self._rewrite_llm = ChatOpenAI(
                model=self._rewrite_llm_model,
                temperature=0,
            )
        logger.llm_call(purpose="rag_query_rewrite", model=self._rewrite_llm_model)
        response = await self._rewrite_llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

