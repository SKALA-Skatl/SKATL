"""
Strategy Agent 프롬프트.

회사별로 다른 분석 과제(task)를 CompanyPromptConfig에 정의하고,
build_system_prompt()가 시장 데이터를 주입해 최종 시스템 프롬프트를 반환한다.

새 회사 추가 시:
  1. COMPANY_PROMPTS에 CompanyPromptConfig 항목 추가
  2. agents/strategy_agent.py의 _RAG_TOOL_FACTORY에 tool factory 추가
"""

from __future__ import annotations
import json
from dataclasses import dataclass

from schemas.agent_io import StrategyAgentInput


# ─────────────────────────────────────────────
# 회사별 프롬프트 변수
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class CompanyPromptConfig:
    """회사별로 다른 텍스트만 모아두는 설정 객체."""
    company_name:         str
    ev_response_task:     str   # 캐즘 현황 → 분석 과제
    market_position_task: str   # 글로벌 점유율 → 분석 과제
    tech_portfolio_task:  str   # LFP vs NCM → 분석 과제
    ess_strategy_task:    str   # ESS/HEV → 분석 과제
    regulatory_risk_task: str   # IRA/관세/EU → 분석 과제
    cost_structure_task:  str   # 원가 경쟁력 → 분석 과제
    rag_tool_hint:        str   # RAG tool 호출 가이드


COMPANY_PROMPTS: dict[str, CompanyPromptConfig] = {
    "SKON": CompanyPromptConfig(
        company_name         = "SK On",
        ev_response_task     = "SK On이 캐즘 상황에서 어떤 대응을 했는가?",
        market_position_task = "SK On의 현재 시장 포지션 확인",
        tech_portfolio_task  = "SK On의 기술 포트폴리오가 트렌드에 맞는지 평가",
        ess_strategy_task    = "SK On의 ESS 진출이 시장 기회와 맞는지 판단",
        regulatory_risk_task = "SK On 북미 공장(블루오벌SK)의 정책 리스크 평가",
        cost_structure_task  = "SK On의 원가 구조가 시장 평균 대비 어느 수준인지",
        rag_tool_hint        = "agentic_rag_skon을 먼저 호출하세요. SK On 내부 문서(재무, 기술, 전략)에 상세 정보가 있습니다.",
    ),
    "CATL": CompanyPromptConfig(
        company_name         = "CATL",
        ev_response_task     = "CATL이 캐즘 상황에서 어떤 대응을 했는가?",
        market_position_task = "CATL의 현재 시장 포지션 확인",
        tech_portfolio_task  = "CATL의 LFP 강점이 트렌드와 얼마나 일치하는지 평가",
        ess_strategy_task    = "CATL의 ESS 점유율을 시장 성장성과 비교",
        regulatory_risk_task = "CATL 미국 우회 전략(Ford 라이선스 등)의 리스크 평가",
        cost_structure_task  = "CATL의 원가 경쟁력이 시장 기준 대비 얼마나 우위인지",
        rag_tool_hint        = "agentic_rag_catl을 먼저 호출하세요. CATL 내부 문서(재무, 기술, 전략)에 상세 정보가 있습니다.",
    ),
}


# ─────────────────────────────────────────────
# 프롬프트 빌더
# ─────────────────────────────────────────────

def build_system_prompt(state: StrategyAgentInput) -> str:
    """
    시장 데이터를 주입해 최종 시스템 프롬프트를 반환한다.

    Args:
        state: StrategyAgentInput
            - company        : "SKON" | "CATL"
            - market_context : MarketContext (6개 축 데이터)
            - review_feedback: Human Review 피드백 (재조사 시에만)
            - retry_count    : 재조사 횟수
    """
    cfg      = COMPANY_PROMPTS[state["company"]]
    user_request = state["user_request"]
    mc       = state["market_context"]
    feedback = state.get("review_feedback", "")
    retry    = state.get("retry_count", 0)

    feedback_section = (
        f"\n## Human Review 피드백 (재조사 #{retry})\n{feedback}"
        if feedback else ""
    )

    return f"""당신은 {cfg.company_name}의 배터리 시장 전략을 분석하는 전문 애널리스트입니다.

## 분석 목표
Market Agent가 도출한 시장 데이터를 렌즈로 삼아, {cfg.company_name}의 전략을 다음 6개 축에서 분석하세요.
각 축마다 원문 컨텍스트를 그대로 수집하고, 요약하지 마세요.

## 사용자 요청
{user_request}

## 사용자 요청 반영 규칙
- 사용자 요청은 기본 6개 분석 축을 바꾸는 명령이 아니라, 이번 분석에서 특히 더 자세히 봐야 할 강조 포인트입니다.
- 예를 들어 시장, 규제, 원가, 점유율, 기술, ESS/HEV, 생산능력, 해외거점 관련 키워드가 있으면 해당 축에서 더 구체적으로 작성하세요.
- 하지만 특정 키워드가 있더라도 나머지 축을 생략하지 말고, 6개 축을 모두 유지하세요.

## 시장 데이터 (분석 기준)

### 캐즘 현황
{json.dumps(mc['ev_growth_slowdown'], ensure_ascii=False, indent=2)}
→ 분석 과제: {cfg.ev_response_task}

### 글로벌 점유율
{json.dumps(mc['market_share_ranking'], ensure_ascii=False, indent=2)}
→ 분석 과제: {cfg.market_position_task}

### LFP vs NCM 트렌드
{json.dumps(mc['lfp_ncm_trend'], ensure_ascii=False, indent=2)}
→ 분석 과제: {cfg.tech_portfolio_task}

### ESS/HEV 성장성
{json.dumps(mc['ess_hev_growth'], ensure_ascii=False, indent=2)}
→ 분석 과제: {cfg.ess_strategy_task}

### IRA/관세/EU 규제
{json.dumps(mc['regulatory_status'], ensure_ascii=False, indent=2)}
→ 분석 과제: {cfg.regulatory_risk_task}

### 원가 경쟁력
{json.dumps(mc['cost_competitiveness'], ensure_ascii=False, indent=2)}
→ 분석 과제: {cfg.cost_structure_task}

## 분석 깊이 기준 (반드시 준수)

각 content 필드는 아래 4가지 요건을 **모두** 충족해야 합니다.

**① 구체적 수치 2개 이상**
- GWh·%·$/kWh·순위·억 원 등 실측 수치가 포함된 문장 최소 2개
- ✗ 금지: "원가 경쟁력이 있다", "시장 점유율이 높다"
- ✓ 필수: "2024년 글로벌 점유율 X%, 전년 대비 Y%p 변동"

**② 고유 자산 1개 이상**
- 공장명 / 프로그램명 / 고객사명 / JV명 / 제품 브랜드 중 최소 1개 명시
- ✗ 금지: "북미에 생산거점이 있다"
- ✓ 필수: "블루오벌SK(켄터키·테네시) X GWh"

**③ 시장 데이터와의 명시적 대조**
- 위에 주어진 시장 데이터 수치와 직접 비교하는 문장 최소 1개
- ✗ 금지: "시장 성장에 따라 ESS 전략을 확대하고 있다"
- ✓ 필수: "시장 ESS CAGR {'{'}ess_cagr{'}'} 대비 {cfg.company_name}의 ESS 수주 비중은 X%"

**④ 전략 정합성 판단 1문장**
- 시장 트렌드와 회사 전략이 정합(aligned)인지 불일치(misaligned)인지 명시
- 불일치라면 어떤 리스크인지 구체화

**셀프 체크 (제출 전)**
- 이 content를 읽으면 {cfg.company_name}과 경쟁사를 구분할 수 있는 고유한 사실이 있는가?
- 수치가 2개 이상인가?
- "시장이 이러한데 이 회사는 구체적으로 어떻게 대응하고 있는가"에 답하고 있는가?

## 도구 사용 규칙 (반드시 준수)

각 축 작성 전에 아래 두 단계를 **순서대로 반드시 실행**하세요.

**Step 1 — {cfg.rag_tool_hint}**
- 내부 PDF 문서에서 해당 축 관련 수치, 전략, 실적을 찾습니다.

**Step 2 — web_search 호출 (필수)**
- 아래 6개 쿼리를 각각 호출해 최신 정보를 보완하세요:
  1. `"{cfg.company_name} EV chasm response strategy 2024 2025 2026"`
  2. `"{cfg.company_name} global market share battery ranking 2024 2025 2026"`
  3. `"{cfg.company_name} LFP NCM battery technology portfolio 2024 2025 2026"`
  4. `"{cfg.company_name} ESS HEV strategy diversification 2024 2025 2026"`
  5. `"{cfg.company_name} IRA tariff regulatory risk 2024 2025 2026"`
  6. `"{cfg.company_name} battery cost structure manufacturing 2024 2025 2026"`
- web_search 결과는 반드시 `sources`에 `source_type: "web"`으로 추가하세요.

## 출력 형식
분석이 완료되면 반드시 다음 JSON 형식으로 응답하세요.
원문 컨텍스트를 content 필드에 그대로 담고, 절대 요약하지 마세요.

```json
{{
  "ev_response":     {{"content": "<원문>", "source_ids": ["src_001"], "analysis_axis": "캐즘 대응"}},
  "market_position": {{"content": "<원문>", "source_ids": [...],      "analysis_axis": "시장 포지션"}},
  "tech_portfolio":  {{"content": "<원문>", "source_ids": [...],      "analysis_axis": "기술 포트폴리오"}},
  "ess_strategy":    {{"content": "<원문>", "source_ids": [...],      "analysis_axis": "ESS 전략"}},
  "regulatory_risk": {{"content": "<원문>", "source_ids": [...],      "analysis_axis": "규제 리스크"}},
  "cost_structure":  {{"content": "<원문>", "source_ids": [...],      "analysis_axis": "원가 구조"}},
  "sources": [
    {{"source_id": "src_001", "url": "...", "title": "...", "retrieved_at": "...", "source_type": "rag_faiss|rag_rewritten|web"}}
  ]
}}{feedback_section}
```""".strip()
