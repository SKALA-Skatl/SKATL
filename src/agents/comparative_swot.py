"""
Comparative SWOT Agent — Phase 3.

SK On vs CATL 비교 SWOT 분석을 생성한다.
노트북(Battery-Comparative-SWOT-ReportAgent.py)의 comparative_swot_agent를
모듈화한 버전이다.

입력:
  - user_request       : 사용자 요청 (강조 포인트)
  - market_context     : MarketContext (Market Agent 출력)
  - skon_result        : StrategyAgentOutput (SKON Strategy Agent 출력)
  - catl_result        : StrategyAgentOutput (CATL Strategy Agent 출력)
  - human_feedback     : HITL #3 거절 피드백 (재실행 시)
  - final_revision_mode: 최대 리뷰 횟수 도달 시 최종 수정 모드 플래그

출력:
  ComparativeSWOTOutput (dict)
"""

from __future__ import annotations
import json
from typing import List, Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from schemas.agent_io import StrategyAgentOutput
from schemas.market_context import MarketContext
from logging_utils import get_logger


logger = get_logger("comparative_swot")

# Market Agent가 포착한 시장 변화 대응 비교 축
COMPARISON_AXES = [
    "chasm_response_fit",
    "market_position_and_scale",
    "technology_trend_fit",
    "ess_hev_diversification_fit",
    "policy_and_regulatory_resilience",
    "cost_competitiveness_under_price_decline",
]


# ─────────────────────────────────────────────
# 출력 스키마
# ─────────────────────────────────────────────

class SWOTQuadrant(BaseModel):
    company: str = Field(description="회사명")
    strengths: List[str] = Field(description="강점")
    weaknesses: List[str] = Field(description="약점")
    opportunities: List[str] = Field(description="기회")
    threats: List[str] = Field(description="위협")


class ComparativeAxisResult(BaseModel):
    axis: str = Field(description="Market Agent가 포착한 시장 변화에 대한 대응 적합성을 비교하는 축")
    winner: Literal["SK On", "CATL", "tie"] = Field(description="해당 시장 변화에 더 적절히 대응한다고 판단되는 기업")
    rationale: str = Field(description="시장 데이터 수치와 기업 전략 수치를 직접 연결한 3문장 이상의 판단 근거")
    reversal_condition: str = Field(description="현재 패자가 역전할 수 있는 시장 조건 또는 임계값 (tie이면 균형을 깰 조건)")


class SWOTComparisonTableRow(BaseModel):
    category: Literal[
        "강점(S) - 내부 경쟁력",
        "약점(W) - 내부 취약점",
        "기회(O) - 외부 시장",
        "위협(T) - 외부 리스크",
    ] = Field(description="SWOT 구분")
    company_a_summary: str = Field(description="A기업(SK On) 요약")
    company_b_summary: str = Field(description="B기업(CATL) 요약")
    strategic_implication: str = Field(description="전략적 시사점 또는 비교 평가")


class ComparativeSWOTOutput(BaseModel):
    comparison_axes: List[ComparativeAxisResult] = Field(
        description="Market Agent 기준의 동일 기준 비교 결과"
    )
    swot_focus_points: List[str] = Field(
        description="시장 변화 대응 관점에서 본 S/W/O/T 비교 기준 및 주목 포인트. 무엇을 어떤 기준으로 비교하는지와 왜 중요한지를 설명"
    )
    strategic_interactions: List[str] = Field(
        description="한 기업의 시장 대응 강점이 상대 기업의 위협·약점으로 연결되는 전략적 상호작용"
    )
    skon_swot: SWOTQuadrant = Field(description="SK On SWOT")
    catl_swot: SWOTQuadrant = Field(description="CATL SWOT")
    swot_comparison_table: List[SWOTComparisonTableRow] = Field(
        description="SWOT 비교 표용 4개 행"
    )
    strategic_gaps: List[str] = Field(
        description="시장 변화 대응 관점에서 드러나는 양사 전략 차이"
    )
    decision_takeaways: List[str] = Field(
        description="의사결정자 관점에서 봐야 할 핵심 대응 포인트"
    )
    evidence_used: List[str] = Field(description="사용한 evidence id 목록")
    confidence: float = Field(description="0과 1 사이 신뢰도")


_SWOT_PROMPT = """
당신은 배터리 시장 전략 비교 분석가입니다.
입력으로 주어진 JSON만 사용해서 SK On과 CATL의 Comparative SWOT를 작성하세요.

목표:
- 동일 기준 기반 Comparative SWOT 비교
- 기업 자체의 절대적 우열보다, Market Agent가 포착한 시장 변화에 각 기업이 얼마나 잘 맞게 대응하고 있는지 비교
- 전략적 우위 및 리스크 요인 도출
- A기업 강점이 B기업 위협 또는 약점으로 이어지는 전략적 상호작용을 한눈에 보이게 정리
- 의사결정자가 바로 사용할 수 있는 비교 포인트 정리

규칙:
- 외부 지식을 추가하지 말 것
- evidence id를 반드시 evidence_used에 남길 것
- comparison_axes는 아래 축만 사용할 것
- winner는 SK On, CATL, tie 중 하나만 사용
- swot_focus_points는 S/W/O/T 각각의 비교 기준과 주목 포인트가 드러나게 작성하되, "무엇을 어떤 기준으로 비교하는지"와 "왜 중요한지"를 설명하는 문장으로 쓸 것
- swot_focus_points에서는 특정 기업이 더 우위라고 단정하는 비교 결과를 직접 쓰지 말 것
- 모든 서술형 문장과 bullet, 표 셀 내용은 반드시 한국어로 작성할 것
- 회사명, 배터리 화학계열(LFP, NCM), 제도명(IRA), 단위, 고유명사 외에는 영어 문장을 쓰지 말 것
- 분석의 중심 질문은 "누가 더 좋은 기업인가"가 아니라 "누가 현재 시장 변화에 더 적절히 대응하고 있는가"여야 함
- 반드시 아래 MarketContext 6개 섹션을 해석에 활용할 것: ev_growth_slowdown, market_share_ranking, lfp_ncm_trend, ess_hev_growth, regulatory_status, cost_competitiveness
- 각 comparison axis 의미:
  - chasm_response_fit: EV 성장 둔화와 지역별 캐즘에 대한 대응 적합성
  - market_position_and_scale: 글로벌 점유율과 체급 기반 방어력
  - technology_trend_fit: LFP vs NCM 트렌드와 기술 포트폴리오의 정합성
  - ess_hev_diversification_fit: ESS/HEV 성장 기회와 다각화 전략의 적합성
  - policy_and_regulatory_resilience: IRA, 관세, EU 규제에 대한 대응력과 리스크
  - cost_competitiveness_under_price_decline: 배터리 가격 하락 국면에서의 원가 방어력

분석 깊이 규칙 (반드시 준수):
- 각 comparison_axes rationale은 반드시 3문장 이상이어야 하며 다음을 모두 포함해야 함:
  ① market_context에서 가져온 수치 1개 이상 (%, GWh, $/kWh, 순위 등)
  ② 기업 전략 분석 결과에서 가져온 수치 또는 고유 자산명 1개 이상
  ③ 두 수치/사실을 연결해 "왜 이 기업이 현재 더 잘 대응하고 있는지" 설명하는 문장 1개
- rationale에 수치가 없으면 작성한 것으로 인정하지 않음
- reversal_condition은 다음 형식으로 작성:
  "만약 [관찰 가능한 시장 조건 또는 수치 임계값]이 발생하면, [패자]가 [분야]에서 역전 가능"
  예시: "만약 LFP 팩 가격이 $80/kWh 이하로 하락하면, NCM 중심 SK On의 원가 경쟁력이 구조적으로 열위로 전환됨"
- strategic_interactions는 최소 3개 작성하고 다음 형식을 따를 것:
  "[A기업 강점/수치] → [메커니즘] → [B기업에 미치는 구체적 위협/약점]"
- swot_comparison_table 각 행의 company_a_summary, company_b_summary는 추상적 서술 금지.
  반드시 수치 또는 고유 자산명(공장·프로그램·고객사)을 포함한 구체적 사실 기반 문장으로 작성할 것
- decision_takeaways는 최소 5개 작성하고 아래 5가지를 각각 하나씩 포함할 것:
  ① 현재 기준 우위 기업과 그 근거 수치
  ② 우위가 역전될 구체적 시장 조건 (임계값 포함)
  ③ SK On의 가장 시급한 대응 과제 (기한 또는 조건 포함)
  ④ CATL의 핵심 리스크와 모니터링해야 할 지표
  ⑤ 의사결정자가 주시해야 할 조기경보 시장 신호 (관찰 가능한 지표)
- review_2_feedback가 있으면 해당 보완 지시를 반영할 것
- human_feedback가 있으면 반드시 반영할 것
- final_revision_mode가 true이면 최대한 완성도 높게 작성할 것
- 사용자 요청은 기본 비교 축을 바꾸는 명령이 아니라 이번 비교에서 더 강조해서 해석해야 할 포인트다
- 예를 들어 규제, 원가, 점유율, 기술, 시장 성장 둔화 같은 키워드가 있으면 rationale, strategic_gaps, decision_takeaways에서 해당 관점을 더 자세히 드러낼 것
- 반드시 JSON으로만 답할 것

user_request:
{user_request}

comparison_axes:
{comparison_axes}

market_context:
{market_context}

human_feedback_from_previous_review:
{human_feedback}

review2_feedback_from_human_review_2:
{review_2_feedback}

final_revision_mode:
{final_revision_mode}

SK On 전략 분석 결과:
{skon_result}

CATL 전략 분석 결과:
{catl_result}

{format_instructions}
"""


# ─────────────────────────────────────────────
# 실행 함수
# ─────────────────────────────────────────────

async def run_comparative_swot(
    user_request: str,
    market_context: MarketContext,
    skon_result: StrategyAgentOutput,
    catl_result: StrategyAgentOutput,
    review_2_feedback: str = "",
    human_feedback: str = "",
    final_revision_mode: bool = False,
) -> dict:
    """
    Comparative SWOT Agent 실행.

    Args:
        user_request        : 사용자 요청(강조 포인트)
        market_context      : Market Agent가 생성한 시장 데이터
        skon_result         : SKON Strategy Agent 출력
        catl_result         : CATL Strategy Agent 출력
        review_2_feedback   : HITL #2 재조사/보완 지시
        human_feedback      : HITL #3 거절 피드백 (재실행 시)
        final_revision_mode : 리뷰 한계 도달 시 최종 수정 모드

    Returns:
        ComparativeSWOTOutput을 dict로 변환한 결과
    """
    with logger.node_span("comparative_swot_agent", {
        "has_feedback": bool(human_feedback),
        "final_revision": final_revision_mode,
    }):
        parser = JsonOutputParser(pydantic_object=ComparativeSWOTOutput)
        prompt = PromptTemplate(
            template=_SWOT_PROMPT,
            input_variables=[
                "user_request", "comparison_axes", "market_context",
                "human_feedback", "review_2_feedback", "final_revision_mode",
                "skon_result", "catl_result",
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        llm   = ChatOpenAI(model="gpt-4o", temperature=0)
        chain = prompt | llm | parser

        result = await chain.ainvoke({
            "user_request":        user_request,
            "comparison_axes":     json.dumps(COMPARISON_AXES, ensure_ascii=False),
            "market_context":      json.dumps(market_context,  ensure_ascii=False, indent=2),
            "human_feedback":      human_feedback or "No prior human feedback.",
            "review_2_feedback":   review_2_feedback or "No review_2 feedback.",
            "final_revision_mode": final_revision_mode,
            "skon_result":         json.dumps(skon_result, ensure_ascii=False, indent=2),
            "catl_result":         json.dumps(catl_result, ensure_ascii=False, indent=2),
        })

        return result if isinstance(result, dict) else result.model_dump()
