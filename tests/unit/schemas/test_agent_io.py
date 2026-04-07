"""
schemas/agent_io.py 단위 테스트.

검증 항목:
  - 열거형 값 정확성
  - StrategyAgentInput / StrategyAgentOutput TypedDict 필드 존재
  - validate_schema_version() 정상/불일치 케이스
  - SCHEMA_VERSION 상수 형식
"""

import pytest
from schemas.agent_io import (
    AgentStatus,
    AgentFailureType,
    SourceType,
    StrategyAgentInput,
    StrategyAgentOutput,
    ConfidenceScores,
    FindingWithSource,
    SourceRecord,
    SCHEMA_VERSION,
    validate_schema_version,
)


class TestEnums:
    def test_agent_status_values(self):
        assert AgentStatus.SUCCESS         == "success"
        assert AgentStatus.PARTIAL_SUCCESS == "partial_success"
        assert AgentStatus.FAILED          == "failed"

    def test_agent_failure_type_values(self):
        assert AgentFailureType.LLM_ERROR    == "llm_error"
        assert AgentFailureType.TOOL_ERROR   == "tool_error"
        assert AgentFailureType.TIMEOUT      == "timeout"
        assert AgentFailureType.MAX_ITER     == "max_iterations"
        assert AgentFailureType.SCHEMA_ERROR == "schema_error"

    def test_source_type_values(self):
        assert SourceType.WEB           == "web"
        assert SourceType.RAG_FAISS     == "rag_faiss"
        assert SourceType.RAG_REWRITTEN == "rag_rewritten"

    def test_enums_are_str_subclass(self):
        # str Enum이므로 문자열 비교 가능해야 함
        assert isinstance(AgentStatus.SUCCESS, str)
        assert isinstance(AgentFailureType.LLM_ERROR, str)
        assert isinstance(SourceType.WEB, str)


class TestSchemaVersion:
    def test_schema_version_format(self):
        parts = SCHEMA_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_validate_schema_version_match(self):
        output = StrategyAgentOutput(schema_version=SCHEMA_VERSION)
        assert validate_schema_version(output) is True

    def test_validate_schema_version_mismatch(self):
        output = StrategyAgentOutput(schema_version="0.0.0")
        assert validate_schema_version(output) is False

    def test_validate_schema_version_missing(self):
        output = StrategyAgentOutput()
        assert validate_schema_version(output) is False


class TestStrategyAgentInput:
    def test_required_fields_exist(self):
        fields = StrategyAgentInput.__annotations__
        assert "company"         in fields
        assert "market_context"  in fields
        assert "review_feedback" in fields
        assert "retry_count"     in fields

    def test_construct_skon_input(self):
        inp = StrategyAgentInput(
            company="SKON",
            market_context={},
            review_feedback="",
            retry_count=0,
        )
        assert inp["company"] == "SKON"
        assert inp["retry_count"] == 0

    def test_construct_catl_input(self):
        inp = StrategyAgentInput(
            company="CATL",
            market_context={},
            review_feedback="보완 필요",
            retry_count=1,
        )
        assert inp["company"] == "CATL"
        assert inp["retry_count"] == 1


class TestStrategyAgentOutput:
    def test_total_false_allows_partial(self):
        # total=False — 일부 필드만 있어도 유효
        output = StrategyAgentOutput(
            company="SKON",
            status=AgentStatus.FAILED,
        )
        assert output["company"] == "SKON"
        assert output["status"] == AgentStatus.FAILED
        assert "ev_response" not in output

    def test_analysis_axis_fields(self):
        fields = StrategyAgentOutput.__annotations__
        for axis in ["ev_response", "market_position", "tech_portfolio",
                     "ess_strategy", "regulatory_risk", "cost_structure"]:
            assert axis in fields

    def test_full_output_construction(self):
        from datetime import datetime, timezone
        output = StrategyAgentOutput(
            schema_version=SCHEMA_VERSION,
            company="CATL",
            status=AgentStatus.SUCCESS,
            failure_type=None,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            llm_call_count=3,
            tool_call_log=[],
            sources=[],
            confidence_scores=ConfidenceScores(
                ev_response=1, market_position=1, tech_portfolio=1,
                ess_strategy=1, regulatory_risk=1, cost_structure=1,
                overall=1.0,
            ),
        )
        assert output["status"] == AgentStatus.SUCCESS
        assert output["confidence_scores"]["overall"] == 1.0
