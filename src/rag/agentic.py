from __future__ import annotations

"""Agentic retrieval loop for local PDF RAG."""

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from .config import RAGConfig
from .constants import (
    COMPANY_SOURCE_BY_COLLECTION,
    ENTITY_HINTS,
    MARKET_SOURCES,
    NUMERIC_HINTS,
    STOPWORDS,
    STRATEGY_HINTS,
    TOPIC_HINTS,
)
from .vectorstore import load_index


@dataclass(slots=True)
class RetrievedHit:
    """Retrieved chunk with ranking metadata."""

    document: Document
    score: float
    matched_queries: list[str]


@dataclass(slots=True)
class RetrievalRound:
    """Trace of one retrieval round."""

    round_index: int
    queries: list[str]
    retrieved_count: int
    unique_sources: list[str]
    sufficient: bool
    missing_sources: list[str]


@dataclass(slots=True)
class AgenticRetrievalResult:
    """Final output of the retrieval loop."""

    collection_name: str
    question: str
    final_hits: list[RetrievedHit]
    rounds: list[RetrievalRound]
    sufficient: bool
    missing_sources: list[str]


def agentic_similarity_search(
    *,
    collection_name: str,
    question: str,
    config: RAGConfig,
    project_root: Path,
    final_k: int = 6,
    per_query_k: int = 4,
    max_rounds: int = 2,
) -> AgenticRetrievalResult:
    """Run collection-aware multi-step retrieval."""

    # 질문 확장 -> 검색 -> coverage 확인 -> 필요 시 재검색 순서로 동작합니다.
    vectorstore = load_index(config, project_root, collection_name)
    rounds: list[RetrievalRound] = []
    aggregated: dict[str, RetrievedHit] = {}
    pending_queries = _plan_queries(question, collection_name)

    for round_index in range(1, max_rounds + 1):
        current_queries = pending_queries[:4]
        for query in current_queries:
            for document in vectorstore.similarity_search(query, k=per_query_k):
                hit = _score_hit(document, question, query, collection_name)
                chunk_id = str(document.metadata.get("chunk_id"))
                existing = aggregated.get(chunk_id)
                if existing is None:
                    aggregated[chunk_id] = hit
                    continue
                existing.score = max(existing.score, hit.score)
                if query not in existing.matched_queries:
                    existing.matched_queries.append(query)

        ranked_hits = _rank_hits(aggregated.values())
        coverage = _assess_coverage(collection_name, question, ranked_hits[:final_k])
        rounds.append(
            RetrievalRound(
                round_index=round_index,
                queries=current_queries,
                retrieved_count=len(ranked_hits),
                unique_sources=coverage["unique_sources"],
                sufficient=coverage["sufficient"],
                missing_sources=coverage["missing_sources"],
            )
        )
        if coverage["sufficient"]:
            return AgenticRetrievalResult(
                collection_name=collection_name,
                question=question,
                final_hits=ranked_hits[:final_k],
                rounds=rounds,
                sufficient=True,
                missing_sources=[],
            )

        pending_queries = _build_follow_up_queries(question, collection_name, coverage["missing_sources"])

    ranked_hits = _rank_hits(aggregated.values())
    coverage = _assess_coverage(collection_name, question, ranked_hits[:final_k])
    return AgenticRetrievalResult(
        collection_name=collection_name,
        question=question,
        final_hits=ranked_hits[:final_k],
        rounds=rounds,
        sufficient=coverage["sufficient"],
        missing_sources=coverage["missing_sources"],
    )


def _plan_queries(question: str, collection_name: str) -> list[str]:
    """Build first-round queries."""

    # 원문 질문에 엔티티 힌트와 토픽 힌트를 붙여 1차 검색어를 만듭니다.
    queries = [question.strip()]
    queries.extend(f"{hint} {question.strip()}" for hint in ENTITY_HINTS.get(collection_name, []))

    keywords = _extract_keywords(question)
    if keywords:
        queries.append(" ".join(keywords[:8]))

    for trigger, expansions in TOPIC_HINTS.items():
        if trigger in question:
            queries.extend(expansions)

    if collection_name in COMPANY_SOURCE_BY_COLLECTION:
        queries.append(f"{ENTITY_HINTS[collection_name][0]} strategy performance capacity")

    if collection_name in {"swot_agent", "report_agent"}:
        queries.append(f"{question.strip()} market outlook comparison")
        queries.append(f"{question.strip()} financial capacity strategy")

    return _deduplicate_queries(queries)


def _build_follow_up_queries(question: str, collection_name: str, missing_sources: list[str]) -> list[str]:
    """Build follow-up queries for missing sources."""

    # 빠진 source를 채우도록 더 직접적인 검색어를 만듭니다.
    queries = [question.strip()]
    for source in missing_sources:
        if source == "skon.pdf":
            queries.extend(
                [
                    "SK On production capacity overseas entity",
                    "SK온 생산능력 해외 법인",
                    "SK온 재무 생산 실적",
                ]
            )
        elif source == "catl.pdf":
            queries.extend(
                [
                    "CATL annual report financial performance",
                    "CATL quarterly revenue production capacity",
                    "CATL 전략 재무 실적",
                ]
            )
        elif source in MARKET_SOURCES:
            queries.extend(
                [
                    "global EV market outlook battery demand",
                    "전기차 시장 전망 배터리 수요 정책",
                    "ESS demand outlook battery industry",
                ]
            )

    if collection_name in {"swot_agent", "report_agent"}:
        queries.append(f"{question.strip()} comparative evidence with citations")

    return _deduplicate_queries(queries)


def _extract_keywords(question: str) -> list[str]:
    """Extract lightweight keywords."""

    tokens = re.findall(r"[A-Za-z0-9가-힣][A-Za-z0-9가-힣./+-]*", question)
    keywords = []
    for token in tokens:
        lowered = token.lower()
        if len(lowered) <= 1:
            continue
        if lowered in STOPWORDS:
            continue
        keywords.append(token)
    return keywords


def _score_hit(document: Document, question: str, query: str, collection_name: str) -> RetrievedHit:
    """Score a retrieved document chunk."""

    # 질문과의 겹침, 회사 문서 우선순위, 표 청크 여부 등을 반영해 점수를 줍니다.
    metadata = document.metadata
    source = str(metadata.get("source", ""))
    chunk_type = str(metadata.get("chunk_type", ""))
    doc_role = str(metadata.get("doc_role", ""))

    question_keywords = {keyword.lower() for keyword in _extract_keywords(question)}
    query_keywords = {keyword.lower() for keyword in _extract_keywords(query)}
    text = document.page_content.lower()

    question_overlap = sum(1 for keyword in question_keywords if keyword in text)
    query_overlap = sum(1 for keyword in query_keywords if keyword in text)
    score = (question_overlap * 1.4) + (query_overlap * 0.8)

    if any(hint.lower() in question.lower() for hint in NUMERIC_HINTS) and chunk_type == "table":
        score += 2.0
    if any(hint.lower() in question.lower() for hint in STRATEGY_HINTS) and doc_role == "market":
        score += 1.0

    preferred_source = COMPANY_SOURCE_BY_COLLECTION.get(collection_name)
    if preferred_source and source == preferred_source:
        score += 2.5
    if collection_name == "market_agent" and source in MARKET_SOURCES:
        score += 2.5
    if collection_name in {"swot_agent", "report_agent"} and source in MARKET_SOURCES:
        score += 1.2

    if "context:" in document.page_content:
        score += 0.4
    if "title:" in document.page_content:
        score += 0.4

    return RetrievedHit(document=document, score=score, matched_queries=[query])


def _rank_hits(hits: list[RetrievedHit] | tuple[RetrievedHit, ...] | object) -> list[RetrievedHit]:
    """Sort hits by score."""

    materialized = list(hits)
    materialized.sort(
        key=lambda hit: (
            hit.score,
            len(hit.matched_queries),
            -int(hit.document.metadata.get("page", 0) or 0),
        ),
        reverse=True,
    )
    return materialized


def _assess_coverage(collection_name: str, question: str, hits: list[RetrievedHit]) -> dict[str, object]:
    """Check whether the current hits cover required sources."""

    # 에이전트별로 꼭 있어야 하는 source가 들어왔는지 확인합니다.
    present_sources = {str(hit.document.metadata.get("source")) for hit in hits}
    missing_sources: list[str] = []

    if collection_name == "market_agent":
        if not present_sources & MARKET_SOURCES:
            missing_sources.extend(sorted(MARKET_SOURCES))
    elif collection_name in COMPANY_SOURCE_BY_COLLECTION:
        company_source = COMPANY_SOURCE_BY_COLLECTION[collection_name]
        if company_source not in present_sources:
            missing_sources.append(company_source)
        if any(hint.lower() in question.lower() for hint in STRATEGY_HINTS) and not present_sources & MARKET_SOURCES:
            missing_sources.append("market_report.pdf")
    else:
        required_sources = {"skon.pdf", "catl.pdf", "market_report.pdf"}
        for source in sorted(required_sources):
            if source not in present_sources:
                missing_sources.append(source)

    asks_numeric = any(hint.lower() in question.lower() for hint in NUMERIC_HINTS)
    if asks_numeric and not any(hit.document.metadata.get("chunk_type") == "table" for hit in hits):
        preferred_source = COMPANY_SOURCE_BY_COLLECTION.get(collection_name)
        if preferred_source and preferred_source not in missing_sources:
            missing_sources.append(preferred_source)

    return {
        # source coverage를 만족하더라도 근거 청크 수가 너무 적으면 재검색을 시도합니다.
        "sufficient": not missing_sources and len(hits) >= 3,
        "missing_sources": missing_sources,
        "unique_sources": sorted(present_sources),
    }


def _deduplicate_queries(queries: list[str]) -> list[str]:
    """Remove repeated queries."""

    deduplicated: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join(query.split()).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduplicated.append(normalized)
    return deduplicated
