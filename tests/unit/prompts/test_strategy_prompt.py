"""
prompts/strategy_prompt.py 단위 테스트.

검증 항목:
  - COMPANY_PROMPTS에 SKON, CATL 등록 여부
  - CompanyPromptConfig 필드 완결성
  - build_system_prompt(): 회사명/분석 과제/RAG hint 주입 확인
  - review_feedback 포함 여부 (재조사 시)
  - 출력 형식 지시(JSON 포맷) 포함 여부
"""

import pytest
from schemas.agent_io import StrategyAgentInput
from schemas.market_context import MOCK_MARKET_CONTEXT
from prompts.strategy_prompt import (
    COMPANY_PROMPTS,
    CompanyPromptConfig,
    build_system_prompt,
)


# ─────────────────────────────────────────────
# 픽스처
# ─────────────────────────────────────────────

def _make_input(company, feedback="", retry=0) -> StrategyAgentInput:
    return StrategyAgentInput(
        company=company,
        market_context=MOCK_MARKET_CONTEXT,
        review_feedback=feedback,
        retry_count=retry,
    )


# ─────────────────────────────────────────────
# COMPANY_PROMPTS
# ─────────────────────────────────────────────

class TestCompanyPrompts:

    def test_skon_and_catl_registered(self):
        assert "SKON" in COMPANY_PROMPTS
        assert "CATL" in COMPANY_PROMPTS

    def test_config_is_frozen_dataclass(self):
        cfg = COMPANY_PROMPTS["SKON"]
        assert isinstance(cfg, CompanyPromptConfig)
        with pytest.raises((AttributeError, TypeError)):
            cfg.company_name = "변경 시도"  # frozen=True

    def test_all_config_fields_non_empty(self):
        required = [
            "company_name", "ev_response_task", "market_position_task",
            "tech_portfolio_task", "ess_strategy_task", "regulatory_risk_task",
            "cost_structure_task", "rag_tool_hint",
        ]
        for company, cfg in COMPANY_PROMPTS.items():
            for field in required:
                value = getattr(cfg, field)
                assert value, f"{company}.{field} 비어있음"

    def test_skon_company_name(self):
        assert COMPANY_PROMPTS["SKON"].company_name == "SK On"

    def test_catl_company_name(self):
        assert COMPANY_PROMPTS["CATL"].company_name == "CATL"

    def test_skon_rag_hint_references_skon_tool(self):
        assert "agentic_rag_skon" in COMPANY_PROMPTS["SKON"].rag_tool_hint

    def test_catl_rag_hint_references_catl_tool(self):
        assert "agentic_rag_catl" in COMPANY_PROMPTS["CATL"].rag_tool_hint


# ─────────────────────────────────────────────
# build_system_prompt
# ─────────────────────────────────────────────

class TestBuildSystemPrompt:

    def test_skon_prompt_contains_company_name(self):
        prompt = build_system_prompt(_make_input("SKON"))
        assert "SK On" in prompt

    def test_catl_prompt_contains_company_name(self):
        prompt = build_system_prompt(_make_input("CATL"))
        assert "CATL" in prompt

    def test_prompt_contains_market_data_axes(self):
        prompt = build_system_prompt(_make_input("SKON"))
        for section in ["캐즘 현황", "글로벌 점유율", "LFP vs NCM 트렌드",
                        "ESS/HEV 성장성", "IRA/관세/EU 규제", "원가 경쟁력"]:
            assert section in prompt, f"'{section}' 섹션 없음"

    def test_skon_prompt_contains_analysis_tasks(self):
        prompt = build_system_prompt(_make_input("SKON"))
        assert "블루오벌SK" in prompt   # SKON 전용 분석 과제
        assert "agentic_rag_skon" in prompt

    def test_catl_prompt_contains_analysis_tasks(self):
        prompt = build_system_prompt(_make_input("CATL"))
        assert "Ford 라이선스" in prompt  # CATL 전용 분석 과제
        assert "agentic_rag_catl" in prompt

    def test_prompt_contains_json_output_format(self):
        prompt = build_system_prompt(_make_input("SKON"))
        assert "ev_response"     in prompt
        assert "market_position" in prompt
        assert "source_ids"      in prompt

    def test_no_feedback_section_when_empty(self):
        prompt = build_system_prompt(_make_input("SKON", feedback=""))
        assert "Human Review 피드백" not in prompt

    def test_feedback_section_included_when_provided(self):
        prompt = build_system_prompt(_make_input(
            "SKON", feedback="ESS 데이터 보완 필요", retry=1
        ))
        assert "Human Review 피드백" in prompt
        assert "재조사 #1" in prompt
        assert "ESS 데이터 보완 필요" in prompt

    def test_feedback_shows_correct_retry_number(self):
        prompt = build_system_prompt(_make_input("CATL", feedback="추가 조사", retry=2))
        assert "재조사 #2" in prompt

    def test_market_data_injected_into_prompt(self):
        """시장 데이터가 실제로 프롬프트에 주입됐는지 확인"""
        prompt = build_system_prompt(_make_input("SKON"))
        # MOCK_MARKET_CONTEXT의 특정 값이 프롬프트에 들어있어야 함
        assert "SNE Research" in prompt       # market_share_ranking source
        assert "BloombergNEF" in prompt       # ev_growth_slowdown source

    def test_prompt_is_string_and_non_empty(self):
        prompt = build_system_prompt(_make_input("SKON"))
        assert isinstance(prompt, str)
        assert len(prompt) > 500  # 충분한 길이

    def test_skon_catl_prompts_differ(self):
        """두 회사의 프롬프트는 달라야 함"""
        skon_prompt = build_system_prompt(_make_input("SKON"))
        catl_prompt = build_system_prompt(_make_input("CATL"))
        assert skon_prompt != catl_prompt
