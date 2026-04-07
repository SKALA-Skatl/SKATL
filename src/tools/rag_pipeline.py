"""
RAG Pipeline 모듈.

구조:
  RAGPipeline.run(query) →
    1. _plan_queries() — 쿼리 확장 (엔티티 힌트 + 토픽 힌트 + 키워드)
    2. 멀티쿼리 LangChain FAISS 검색 및 결과 집계
       (cosine + 키워드 보너스로 재랭킹)
    3. LLM 충분성 평가 (_evaluate_and_rewrite):
         - SUFFICIENT → 즉시 반환
         - INSUFFICIENT → 새 검색 쿼리 생성 후 재검색
    4. max_rewrites 초과 시 최선 결과로 강제 반환

인덱스 포맷:
  src/rag/vectorstore.py 로 빌드한 LangChain FAISS 인덱스를 사용.
  (data/vectorstores/{collection_name}/ 폴더)
  L2 거리는 내부에서 cosine similarity로 변환:
    cosine = 1 - l2² / 2  (normalize_embeddings=True 전제)
"""

from __future__ import annotations
import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from rag.constants import (
    COMPANY_SOURCE_BY_COLLECTION,
    ENTITY_HINTS,
    NUMERIC_HINTS,
    STOPWORDS,
    STRATEGY_HINTS,
    TOPIC_HINTS,
)
from rag.source_metadata import resolve_source_metadata
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
    page:         int | None = None
    reference_text: str = ""
    source:       str = ""   # PDF 파일명 (예: "skon.pdf") — coverage 평가에 사용
    published_date: str | None = None


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
                published_date=doc.published_date,
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
        vectorstore         : LangChain FAISS 인스턴스
                              src/rag/vectorstore.load_index() 로 로드
        collection_name     : 컬렉션 이름 (예: "skon_agent", "catl_agent")
                              쿼리 확장 및 coverage 평가에 사용
        relevance_threshold : cosine similarity 하한 (기본 0.75)
                              LLM 평가 전 빠른 통과 기준으로 사용
        max_rewrites        : LLM 재검색 최대 횟수 (기본 3)
        top_k               : 최종 반환 청크 수 (기본 5)
    """

    def __init__(
        self,
        vectorstore: Any,             # langchain_community.vectorstores.FAISS
        collection_name: str = "",
        relevance_threshold: float = 0.75,
        max_rewrites: int = 3,
        top_k: int = 5,
    ):
        self.vectorstore     = vectorstore
        self.collection_name = collection_name
        self.threshold       = relevance_threshold
        self.max_rewrites    = max_rewrites
        self.top_k           = top_k

        self._llm       = None
        self._llm_model = "gpt-4o-mini"

    # ── 퍼블릭 API ────────────────────────────

    async def run(self, query: str) -> RAGResult:
        """
        Agentic 검색 루프.

        Round 0  : 멀티쿼리 확장으로 초기 검색 → cosine threshold 확인
        Round 1+ : LLM이 충분성 평가 후 새 쿼리 생성 → 재검색
        탈출      : SUFFICIENT 판정 또는 max_rewrites 초과
        """
        aggregated: dict[str, tuple[RAGDocument, float]] = {}

        # ── Round 0: 멀티쿼리 확장 검색 ──────
        initial_queries = self._plan_queries(query)
        await self._search_and_aggregate(initial_queries[:4], query, aggregated)

        top_docs = self._rank_aggregated(aggregated)
        top_scores = [d.cosine_score for d in top_docs]

        logger.rag_search(query=query, cosine_scores=top_scores, rewrite_count=0)

        if top_scores and max(top_scores) >= self.threshold:
            return RAGResult(
                documents=top_docs,
                query_used=query,
                rewrite_count=0,
                source_type=SourceType.RAG_FAISS,
            )

        best_result = RAGResult(
            documents=top_docs,
            query_used=query,
            rewrite_count=0,
            source_type=SourceType.RAG_FAISS,
        )

        # ── Round 1 ~ max_rewrites: LLM 평가 + 재검색 ──
        current_query = query
        for rewrite_count in range(1, self.max_rewrites + 1):

            sufficient, new_query = await self._evaluate_and_rewrite(
                original_query=query,
                current_query=current_query,
                docs=top_docs,
            )

            if sufficient:
                return RAGResult(
                    documents=top_docs,
                    query_used=current_query,
                    rewrite_count=rewrite_count - 1,
                    source_type=(
                        SourceType.RAG_FAISS if rewrite_count == 1
                        else SourceType.RAG_REWRITTEN
                    ),
                )

            if not new_query:
                break

            current_query = new_query
            await self._search_and_aggregate([current_query], query, aggregated)

            top_docs = self._rank_aggregated(aggregated)
            top_scores = [d.cosine_score for d in top_docs]

            logger.rag_search(
                query=current_query,
                cosine_scores=top_scores,
                rewrite_count=rewrite_count,
            )

            best_result = RAGResult(
                documents=top_docs,
                query_used=current_query,
                rewrite_count=rewrite_count,
                source_type=SourceType.RAG_REWRITTEN,
            )

            if top_scores and max(top_scores) >= self.threshold:
                return best_result

        # ── 강제 탈출 ──────────────────────────
        logger.rag_search(
            query=current_query,
            cosine_scores=[],
            rewrite_count=self.max_rewrites,
            forced_return=True,
        )
        best_result.forced_return = True
        return best_result

    # ── 검색 및 집계 ──────────────────────────

    async def _search_and_aggregate(
        self,
        queries: list[str],
        original_query: str,
        aggregated: dict[str, tuple[RAGDocument, float]],
    ) -> None:
        """여러 쿼리를 검색하고 결과를 집계. 동일 청크는 최고점만 유지."""
        for q in queries:
            docs, _ = await self._search(q)
            for doc in docs:
                kw_bonus = self._score_keywords(doc, original_query)
                combined = doc.cosine_score + kw_bonus
                existing = aggregated.get(doc.doc_id)
                if existing is None or combined > existing[1]:
                    aggregated[doc.doc_id] = (doc, combined)

    def _rank_aggregated(
        self,
        aggregated: dict[str, tuple[RAGDocument, float]],
    ) -> list[RAGDocument]:
        """combined score 기준 정렬 후 top_k 반환."""
        items = sorted(aggregated.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in items[: self.top_k]]

    # ── LangChain FAISS 검색 ──────────────────

    async def _search(self, query: str) -> tuple[list[RAGDocument], list[float]]:
        """
        LangChain FAISS similarity_search_with_score 비동기 래핑.

        LangChain FAISS 기본 인덱스(IndexFlatL2) + normalize_embeddings=True 조합에서
        L2 거리를 cosine similarity로 변환:
            cosine = 1 - l2² / 2
        """
        loop = asyncio.get_event_loop()
        fn = partial(
            self.vectorstore.similarity_search_with_score,
            query,
            k=self.top_k,
        )
        results = await loop.run_in_executor(None, fn)

        docs: list[RAGDocument] = []
        scores: list[float] = []

        for document, l2_dist in results:
            cosine = float(1.0 - (l2_dist ** 2) / 2.0)
            metadata = document.metadata
            source = str(metadata.get("source", ""))
            chunk_id = str(metadata.get("chunk_id", hash(document.page_content)))
            resolved = resolve_source_metadata(
                source_id=chunk_id,
                source=source,
                title=str(metadata.get("title", "")),
            )
            docs.append(RAGDocument(
                doc_id=chunk_id,
                content=document.page_content,
                source_url=str(metadata.get("url", "")) or resolved["url"] or source,
                source_title=str(metadata.get("title", "")) or resolved["title"] or source,
                cosine_score=cosine,
                page=metadata.get("page"),
                reference_text=str(metadata.get("reference_text", "")) or resolved["reference_text"],
                source=source,
                published_date=metadata.get("published_date"),
            ))
            scores.append(cosine)

        return docs, scores

    # ── LLM 충분성 평가 + 쿼리 재생성 ─────────

    async def _evaluate_and_rewrite(
        self,
        original_query: str,
        current_query: str,
        docs: list[RAGDocument],
    ) -> tuple[bool, str]:
        """
        LLM이 현재 검색 결과의 충분성을 스스로 평가하고,
        부족하면 개선된 검색 쿼리를 직접 생성한다.

        Returns:
            (sufficient, new_query)
        """
        if not docs:
            return False, original_query

        context_preview = "\n".join(
            f"[{i+1}] (cosine={d.cosine_score:.3f}) {d.content[:300]}"
            for i, d in enumerate(docs[:4])
        )

        prompt = (
            f"당신은 배터리 전략 분석 리서치 에이전트입니다.\n\n"
            f"원본 질문: {original_query}\n"
            f"마지막 검색 쿼리: {current_query}\n\n"
            f"검색된 문서 (상위 {min(4, len(docs))}개):\n{context_preview}\n\n"
            f"위 문서들이 원본 질문에 충분히 답할 수 있는지 평가하세요.\n"
            f"충분하면 sufficient=true, 부족하면 sufficient=false와 함께\n"
            f"어떤 정보가 누락되었는지 reason을 쓰고, "
            f"배터리 전략 분석에 더 적합한 검색 쿼리 1개를 new_query에 작성하세요.\n\n"
            f"JSON 형식으로만 응답하세요:\n"
            f'{{"sufficient": true/false, "reason": "...", "new_query": "..."}}'
        )

        llm = self._get_llm()
        logger.llm_call(purpose="rag_evaluate_and_rewrite", model=self._llm_model)
        response = await llm.ainvoke([HumanMessage(content=prompt)])

        try:
            start = response.content.find("{")
            end   = response.content.rfind("}") + 1
            parsed = json.loads(response.content[start:end])
            return bool(parsed.get("sufficient", False)), str(parsed.get("new_query", "")).strip()
        except (json.JSONDecodeError, ValueError):
            return False, current_query

    # ── 쿼리 확장 (src/rag/agentic.py 기반) ──

    def _plan_queries(self, question: str) -> list[str]:
        """단일 질문 → 멀티쿼리 확장 (엔티티 힌트 + 토픽 힌트 + 키워드)."""
        queries = [question.strip()]

        for hint in ENTITY_HINTS.get(self.collection_name, []):
            queries.append(f"{hint} {question.strip()}")

        keywords = self._extract_keywords(question)
        if keywords:
            queries.append(" ".join(keywords[:8]))

        for trigger, expansions in TOPIC_HINTS.items():
            if trigger in question:
                queries.extend(expansions)

        if self.collection_name in COMPANY_SOURCE_BY_COLLECTION:
            hints = ENTITY_HINTS.get(self.collection_name, [])
            if hints:
                queries.append(f"{hints[0]} strategy performance capacity")

        return self._deduplicate_queries(queries)

    # ── 키워드 점수 보너스 ────────────────────

    def _score_keywords(self, doc: RAGDocument, original_query: str) -> float:
        """cosine score에 더할 키워드 겹침 보너스. 스케일은 cosine 대비 작게 유지."""
        keywords = {k.lower() for k in self._extract_keywords(original_query)}
        text = doc.content.lower()
        overlap = sum(1 for kw in keywords if kw in text)
        score = overlap * 0.05
        if any(h.lower() in original_query.lower() for h in NUMERIC_HINTS):
            score += 0.05
        if any(h.lower() in original_query.lower() for h in STRATEGY_HINTS):
            score += 0.03
        return score

    # ── 키워드 추출 (src/rag/agentic.py 기반) ─

    @staticmethod
    def _extract_keywords(question: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9가-힣][A-Za-z0-9가-힣./+-]*", question)
        return [t for t in tokens if len(t) > 1 and t.lower() not in STOPWORDS]

    @staticmethod
    def _deduplicate_queries(queries: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for q in queries:
            normalized = " ".join(q.split()).strip()
            lowered = normalized.lower()
            if normalized and lowered not in seen:
                seen.add(lowered)
                result.append(normalized)
        return result

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(model=self._llm_model, temperature=0)
        return self._llm
