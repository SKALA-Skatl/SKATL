"""Build default HITL feedback when the reviewer leaves the field blank."""

from __future__ import annotations

from typing import Any


_MARKET_SECTION_LABELS = {
    "ev_growth_slowdown": "EV 성장 둔화",
    "market_share_ranking": "글로벌 점유율",
    "lfp_ncm_trend": "LFP/NCM 트렌드",
    "ess_hev_growth": "ESS/HEV 성장",
    "regulatory_status": "규제 현황",
    "cost_competitiveness": "원가 경쟁력",
}

_STRATEGY_AXIS_LABELS = {
    "ev_response": "캐즘 대응",
    "market_position": "시장 포지션",
    "tech_portfolio": "기술 포트폴리오",
    "ess_strategy": "ESS/HEV 전략",
    "regulatory_risk": "규제 리스크",
    "cost_structure": "원가 구조",
}


def build_default_feedback(payload: dict[str, Any], decision: str) -> str:
    """Return a reasonable default feedback string for redo/reject decisions."""

    if decision == "approve":
        return ""

    phase = str(payload.get("phase") or payload.get("stage") or "")
    if phase == "review_1":
        return _build_review_1_feedback(payload)
    if phase == "review_2":
        return _build_review_2_feedback(payload)
    if phase in {"review_3", "human_review_3"}:
        return _build_review_3_feedback(payload)
    return "현재 결과에서 부족한 근거와 출처를 보강해 다시 작성하세요."


def _build_review_1_feedback(payload: dict[str, Any]) -> str:
    market_result = payload.get("market_result", {}) or {}
    thin_sections = market_result.get("thin_sections", []) or []
    section_names = [
        _MARKET_SECTION_LABELS.get(key, key)
        for key in thin_sections
    ]

    market_context = market_result.get("market_context", {}) or {}
    references = market_context.get("references", []) or []
    source_records = market_context.get("source_records", []) or []
    web_count = sum(1 for item in source_records if str(item.get("source_type", "")).strip() == "web")

    instructions: list[str] = []
    if section_names:
        instructions.append(
            "다음 시장 축을 우선 보강하세요: "
            + ", ".join(section_names)
            + ". 각 축에 2024~2026 기준 수치와 전략적 해석을 더 구체적으로 넣으세요."
        )
    instructions.append("내부 PDF 근거뿐 아니라 최신 web_search 결과도 반드시 포함하세요.")
    if web_count == 0:
        instructions.append("웹 출처가 없으므로 최신 기사·공식 발표·기관 자료를 최소 1개 이상 추가하세요.")
    if len(references) < 2:
        instructions.append("REFERENCE와 source_records를 실제로 사용한 자료 중심으로 다시 채우세요.")
    instructions.append("각 섹션의 source_ids가 실제 사용한 출처와 연결되도록 다시 정리하세요.")
    return " ".join(instructions)


def _build_review_2_feedback(payload: dict[str, Any]) -> str:
    instructions: list[str] = []
    for company_key, company_label in (("skon", "SK On"), ("catl", "CATL")):
        info = payload.get(company_key, {}) or {}
        missing_axes = [
            label
            for key, label in _STRATEGY_AXIS_LABELS.items()
            if not isinstance(info.get(key), dict) or not str((info.get(key) or {}).get("content", "")).strip()
        ]
        sources = info.get("sources", []) or []
        web_count = sum(1 for item in sources if str(item.get("source_type", "")).strip() == "web")

        if missing_axes:
            instructions.append(
                f"{company_label}는 {', '.join(missing_axes)} 축의 내용이 비어 있거나 약하므로 우선 보강하세요."
            )
        instructions.append(
            f"{company_label} 각 축에는 수치 2개 이상, 고유 거점/고객/JV/제품명 1개 이상, 시장 데이터와의 직접 비교 문장을 포함하세요."
        )
        if web_count == 0:
            instructions.append(f"{company_label} 최신 웹 출처가 없으므로 web_search 근거를 최소 1개 이상 추가하세요.")
    instructions.append("source_ids와 sources를 실제 사용 근거에 맞게 다시 연결하고, RAG와 웹 출처를 함께 사용해 재조사하세요.")
    return " ".join(instructions)


def _build_review_3_feedback(payload: dict[str, Any]) -> str:
    report = payload.get("report") or (payload.get("action_requests") or [{}])[0].get("args", {}).get("report_draft") or {}
    references = report.get("references", []) or []
    market_background = report.get("market_background", []) or []
    focus_points = report.get("comparative_swot_focus_points", []) or []
    company_comparison = report.get("comparative_swot_company_comparison", []) or []
    implications = report.get("integrated_implications", []) or []

    instructions: list[str] = [
        "보고서 본문을 더 풍부하게 작성하고, 요약형 표현 대신 근거와 해석이 있는 완결된 문장으로 다시 써주세요.",
    ]
    if len(market_background) < 4:
        instructions.append("시장 배경 bullet 수가 부족하므로 6개 시장 축을 바탕으로 내용을 보강하세요.")
    if len(focus_points) < 4:
        instructions.append("SWOT 비교 기준(4-1)을 더 구체적으로 작성하세요.")
    if len(company_comparison) < 4:
        instructions.append("SWOT 기업별 비교(4-2)를 더 구체적으로 작성하세요.")
    if len(implications) < 5:
        instructions.append("종합 시사점은 최소 5개 이상으로 늘리고, 우위 기업과 우위 전환 조건을 명확히 쓰세요.")
    if len(references) < 3:
        instructions.append("Reference는 실제 사용한 자료 위주로 다시 정리하세요.")
    instructions.append("각 회사 섹션의 전략 방향과 watchpoints도 더 길고 구체적인 문장으로 보강하세요.")
    return " ".join(instructions)
