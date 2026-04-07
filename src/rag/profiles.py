from __future__ import annotations

"""Agent-specific retrieval profiles."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentStateProfile:
    """Profile for one agent's retrieval state."""

    collection_name: str
    agent_role: str
    mission: str
    required_topics: tuple[str, ...]
    expected_sources: tuple[str, ...]
    handoff_notes: tuple[str, ...]


# 각 에이전트가 어떤 근거를 기대하는지 여기서 정의합니다.
AGENT_STATE_PROFILES: dict[str, AgentStateProfile] = {
    "market_agent": AgentStateProfile(
        collection_name="market_agent",
        agent_role="market analyst",
        mission="시장 전망, 수요, 정책, 산업 구조를 근거와 함께 요약한다.",
        required_topics=("market", "strategy"),
        expected_sources=("market_report.pdf", "analyst_report.pdf"),
        handoff_notes=(
            "시장 수치와 정책 변화는 반드시 source/page를 같이 전달",
            "기업 개별 분석보다 시장 배경 설명에 집중",
        ),
    ),
    "skon_agent": AgentStateProfile(
        collection_name="skon_agent",
        agent_role="SK On company analyst",
        mission="SK온의 생산, 재무, 해외 거점, 전략 포인트를 시장 배경과 함께 정리한다.",
        required_topics=("capacity", "finance", "overseas", "strategy"),
        expected_sources=("skon.pdf", "market_report.pdf"),
        handoff_notes=(
            "기업 사실은 가능하면 skon.pdf 근거를 우선 사용",
            "시장 배경을 붙일 때만 market_report.pdf 또는 analyst_report.pdf를 함께 사용",
        ),
    ),
    "catl_agent": AgentStateProfile(
        collection_name="catl_agent",
        agent_role="CATL company analyst",
        mission="CATL의 재무, 생산, 전략 포인트를 시장 배경과 함께 정리한다.",
        required_topics=("capacity", "finance", "strategy"),
        expected_sources=("catl.pdf", "market_report.pdf"),
        handoff_notes=(
            "CATL 수치는 catl.pdf 표 청크를 우선 사용",
            "시장 설명은 market 자료를 보조 근거로 사용",
        ),
    ),
    "swot_agent": AgentStateProfile(
        collection_name="swot_agent",
        agent_role="comparative SWOT analyst",
        mission="SK온과 CATL을 시장 배경 위에서 비교할 수 있는 근거를 수집한다.",
        required_topics=("market", "finance", "capacity", "strategy", "swot"),
        expected_sources=("skon.pdf", "catl.pdf", "market_report.pdf"),
        handoff_notes=(
            "반드시 SK온, CATL, 시장 자료가 모두 들어가야 함",
            "비교 문장은 만들지 말고 비교 가능한 근거를 정리해서 넘길 것",
        ),
    ),
    "report_agent": AgentStateProfile(
        collection_name="report_agent",
        agent_role="report writer",
        mission="시장, 기업, 비교 분석을 최종 보고서 작성에 맞게 근거 중심으로 정리한다.",
        required_topics=("market", "finance", "capacity", "strategy", "swot"),
        expected_sources=("skon.pdf", "catl.pdf", "market_report.pdf", "analyst_report.pdf"),
        handoff_notes=(
            "섹션별로 바로 쓸 수 있는 citation packet 형태를 유지",
            "요약보다 근거의 폭과 출처 균형을 우선",
        ),
    ),
}
