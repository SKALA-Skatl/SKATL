from __future__ import annotations

"""Pipeline-ready retrieval state builder."""

from dataclasses import asdict, dataclass
from pathlib import Path

from .agentic import AgenticRetrievalResult, RetrievedHit, agentic_similarity_search
from .config import RAGConfig
from .constants import TOPIC_KEYWORDS
from .profiles import AGENT_STATE_PROFILES, AgentStateProfile


@dataclass(slots=True)
class EvidencePacket:
    """Evidence unit for downstream agents."""

    rank: int
    score: float
    source: str
    page: int | None
    chunk_type: str
    entity: str | None
    doc_role: str | None
    topic_tags: list[str]
    matched_queries: list[str]
    content: str


@dataclass(slots=True)
class CoverageState:
    """Coverage summary for retrieved evidence."""

    present_sources: list[str]
    missing_sources: list[str]
    topic_coverage: dict[str, int]
    chunk_type_counts: dict[str, int]
    source_counts: dict[str, int]


@dataclass(slots=True)
class AgentRetrievalState:
    """Retrieval handoff state for one agent."""

    collection_name: str
    agent_role: str
    mission: str
    user_question: str
    sufficient: bool
    expected_sources: list[str]
    required_topics: list[str]
    handoff_notes: list[str]
    coverage: CoverageState
    evidence_packets: list[EvidencePacket]
    evidence_by_topic: dict[str, list[EvidencePacket]]
    retrieval_trace: list[dict[str, object]]
    prompt_context: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the state as a dictionary."""

        return {
            "collection_name": self.collection_name,
            "agent_role": self.agent_role,
            "mission": self.mission,
            "user_question": self.user_question,
            "sufficient": self.sufficient,
            "expected_sources": self.expected_sources,
            "required_topics": self.required_topics,
            "handoff_notes": self.handoff_notes,
            "coverage": asdict(self.coverage),
            "evidence_packets": [asdict(packet) for packet in self.evidence_packets],
            "evidence_by_topic": {
                topic: [asdict(packet) for packet in packets]
                for topic, packets in self.evidence_by_topic.items()
            },
            "retrieval_trace": self.retrieval_trace,
            "prompt_context": self.prompt_context,
        }


def build_agent_retrieval_state(
    *,
    collection_name: str,
    question: str,
    config: RAGConfig,
    project_root: Path,
    final_k: int = 6,
    per_query_k: int = 4,
    max_rounds: int = 2,
) -> AgentRetrievalState:
    """Build a pipeline-ready retrieval state."""

    # retrieval 결과를 다음 에이전트가 바로 쓸 수 있는 state로 정리합니다.
    profile = AGENT_STATE_PROFILES[collection_name]
    retrieval_result = agentic_similarity_search(
        collection_name=collection_name,
        question=question,
        config=config,
        project_root=project_root,
        final_k=final_k,
        per_query_k=per_query_k,
        max_rounds=max_rounds,
    )

    evidence_packets = _build_evidence_packets(retrieval_result.final_hits)
    evidence_by_topic = _group_packets_by_topic(evidence_packets, profile.required_topics)
    coverage = _build_coverage_state(profile, retrieval_result, evidence_packets)
    prompt_context = _render_prompt_context(profile, question, retrieval_result, evidence_packets, coverage)
    retrieval_trace = [
        {
            "round": round_.round_index,
            "queries": round_.queries,
            "retrieved_count": round_.retrieved_count,
            "unique_sources": round_.unique_sources,
            "sufficient": round_.sufficient,
            "missing_sources": round_.missing_sources,
        }
        for round_ in retrieval_result.rounds
    ]

    return AgentRetrievalState(
        collection_name=collection_name,
        agent_role=profile.agent_role,
        mission=profile.mission,
        user_question=question,
        sufficient=retrieval_result.sufficient,
        expected_sources=list(profile.expected_sources),
        required_topics=list(profile.required_topics),
        handoff_notes=list(profile.handoff_notes),
        coverage=coverage,
        evidence_packets=evidence_packets,
        evidence_by_topic=evidence_by_topic,
        retrieval_trace=retrieval_trace,
        prompt_context=prompt_context,
    )


def _build_evidence_packets(hits: list[RetrievedHit]) -> list[EvidencePacket]:
    """Convert hits into evidence packets."""

    packets: list[EvidencePacket] = []
    for index, hit in enumerate(hits, start=1):
        metadata = hit.document.metadata
        packets.append(
            EvidencePacket(
                rank=index,
                score=hit.score,
                source=str(metadata.get("source", "")),
                page=_to_int_or_none(metadata.get("page")),
                chunk_type=str(metadata.get("chunk_type", "")),
                entity=_to_optional_str(metadata.get("entity")),
                doc_role=_to_optional_str(metadata.get("doc_role")),
                topic_tags=_detect_topics(hit.document.page_content),
                matched_queries=list(hit.matched_queries),
                content=hit.document.page_content.strip(),
            )
        )
    return packets


def _group_packets_by_topic(
    packets: list[EvidencePacket],
    required_topics: tuple[str, ...],
) -> dict[str, list[EvidencePacket]]:
    """Group evidence by topic."""

    # 프롬프트 조립이 쉽도록 topic별 evidence 묶음을 만듭니다.
    grouped: dict[str, list[EvidencePacket]] = {topic: [] for topic in required_topics}
    grouped["general"] = []

    for packet in packets:
        matched = False
        for topic in required_topics:
            if topic in packet.topic_tags:
                grouped[topic].append(packet)
                matched = True
        if not matched:
            grouped["general"].append(packet)

    return {topic: values[:3] for topic, values in grouped.items() if values}


def _build_coverage_state(
    profile: AgentStateProfile,
    retrieval_result: AgenticRetrievalResult,
    packets: list[EvidencePacket],
) -> CoverageState:
    """Summarize retrieval coverage."""

    source_counts: dict[str, int] = {}
    chunk_type_counts: dict[str, int] = {}
    topic_coverage = {topic: 0 for topic in profile.required_topics}

    for packet in packets:
        source_counts[packet.source] = source_counts.get(packet.source, 0) + 1
        chunk_type_counts[packet.chunk_type] = chunk_type_counts.get(packet.chunk_type, 0) + 1
        for topic in packet.topic_tags:
            if topic in topic_coverage:
                topic_coverage[topic] += 1

    present_sources = sorted(source_counts)
    missing_sources = list(retrieval_result.missing_sources)

    for expected_source in profile.expected_sources:
        if expected_source not in present_sources and expected_source not in missing_sources:
            missing_sources.append(expected_source)

    return CoverageState(
        present_sources=present_sources,
        missing_sources=missing_sources,
        topic_coverage=topic_coverage,
        chunk_type_counts=chunk_type_counts,
        source_counts=source_counts,
    )


def _render_prompt_context(
    profile: AgentStateProfile,
    question: str,
    retrieval_result: AgenticRetrievalResult,
    packets: list[EvidencePacket],
    coverage: CoverageState,
) -> str:
    """Render prompt-ready context text."""

    # 다음 에이전트가 그대로 프롬프트에 넣을 수 있는 텍스트를 만듭니다.
    lines = [
        f"[Agent Role] {profile.agent_role}",
        f"[Mission] {profile.mission}",
        f"[Question] {question}",
        f"[Sufficient] {retrieval_result.sufficient}",
        "[Expected Sources] " + ", ".join(profile.expected_sources),
        "[Present Sources] " + (", ".join(coverage.present_sources) if coverage.present_sources else "(none)"),
        "[Missing Sources] " + (", ".join(coverage.missing_sources) if coverage.missing_sources else "(none)"),
        "[Required Topics] " + ", ".join(profile.required_topics),
        "[Handoff Notes]",
    ]
    for note in profile.handoff_notes:
        lines.append(f"- {note}")

    lines.append("[Evidence Packets]")
    for packet in packets:
        lines.extend(
            [
                f"Evidence {packet.rank}",
                f"source: {packet.source}",
                f"page: {packet.page}",
                f"chunk_type: {packet.chunk_type}",
                "topics: " + (", ".join(packet.topic_tags) if packet.topic_tags else "(none)"),
                "matched_queries: " + (" | ".join(packet.matched_queries) if packet.matched_queries else "(none)"),
                "content:",
                packet.content,
            ]
        )
    return "\n".join(lines)


def _detect_topics(text: str) -> list[str]:
    """Detect topic tags from text."""

    lowered = text.lower()
    topics: list[str] = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            topics.append(topic)
    return topics or ["general"]


def _to_int_or_none(value: object) -> int | None:
    """Normalize a page value to an integer."""

    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _to_optional_str(value: object) -> str | None:
    """Normalize optional metadata to a string."""

    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None
