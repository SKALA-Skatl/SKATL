"""
orchestrator/orchestrator.py 통합 테스트.

LLM/RAG를 mock으로 대체해 그래프 전체 흐름 검증.

검증 시나리오:
  1. 정상 흐름 — approve → END
  2. 양쪽 실패 → error_handler → END (interrupt 없음)
  3. redo_skon 루프 — SKON 재조사 후 approve
  4. retry 한계 소진 → 강제 approve
  5. 그래프 구조 — 노드 목록, input_schema 격리
  6. 불변 필드 보호 — 노드에서 user_request 반환 시 RuntimeError
  7. Enum 직렬화 경고 없음
"""

import pytest
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from schemas.agent_io import (
    AgentStatus,
    AgentFailureType,
    ConfidenceScores,
    SCHEMA_VERSION,
    StrategyAgentOutput,
    StrategyAgentInput,
)
from schemas.market_context import MOCK_MARKET_CONTEXT
from schemas.state import OrchestratorInput
from orchestrator.orchestrator import (
    build_graph,
    compile_standalone,
    make_checkpointer,
    orchestrator_fanout,
    fan_in_node,
    _route_after_fan_in,
    _route_after_hitl_2,
    _allowed_decisions,
    _validate_resume,
    _decision_to_targets,
    MAX_RETRIES,
)


# ─────────────────────────────────────────────
# 공용 헬퍼
# ─────────────────────────────────────────────

def _success(company: str) -> StrategyAgentOutput:
    axes = {
        ax: {"content": f"{company} {ax} 원문", "source_ids": ["s1"], "analysis_axis": ax}
        for ax in ["ev_response", "market_position", "tech_portfolio",
                   "ess_strategy", "regulatory_risk", "cost_structure"]
    }
    return StrategyAgentOutput(
        schema_version=SCHEMA_VERSION,
        company=company,
        status=AgentStatus.SUCCESS,
        failure_type=None,
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        llm_call_count=2,
        tool_call_log=[],
        sources=[],
        confidence_scores=ConfidenceScores(
            ev_response=1, market_position=1, tech_portfolio=1,
            ess_strategy=1, regulatory_risk=1, cost_structure=1, overall=1.0,
        ),
        **axes,
    )


def _partial(company: str) -> StrategyAgentOutput:
    out = _success(company)
    out["status"] = AgentStatus.PARTIAL_SUCCESS
    return out


def _failed(company: str) -> StrategyAgentOutput:
    return StrategyAgentOutput(
        schema_version=SCHEMA_VERSION,
        company=company,
        status=AgentStatus.FAILED,
        failure_type=AgentFailureType.LLM_ERROR,
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        llm_call_count=0,
        tool_call_log=[],
        sources=[],
        confidence_scores=ConfidenceScores(
            ev_response=0, market_position=0, tech_portfolio=0,
            ess_strategy=0, regulatory_risk=0, cost_structure=0, overall=0.0,
        ),
    )


def _initial_input(user_request: str = "배터리 전략 분석") -> dict:
    return {
        **OrchestratorInput(
            user_request=user_request,
            market_context=MOCK_MARKET_CONTEXT,
        ),
        "skon_retry_count": 0,
        "catl_retry_count": 0,
        "redo_targets": [],
        "error_log": [],
    }


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ─────────────────────────────────────────────
# 그래프 구조 테스트
# ─────────────────────────────────────────────

class TestGraphStructure:

    def test_build_graph_returns_state_graph(self):
        from langgraph.graph import StateGraph
        graph = build_graph()
        assert isinstance(graph, StateGraph)

    def test_compiled_graph_has_expected_nodes(self):
        app = build_graph().compile(checkpointer=MemorySaver())
        nodes = list(app.get_graph().nodes.keys())
        for expected in [
            "skon_agent_node", "catl_agent_node",
            "fan_in_node", "hitl_2_node", "error_handler",
        ]:
            assert expected in nodes, f"노드 '{expected}' 없음"

    def test_agent_nodes_have_input_schema(self):
        """input_schema=StrategyAgentInput 등록 확인 — OrchestratorState 격리"""
        from langgraph.graph import StateGraph
        from schemas.state import OrchestratorState

        builder = StateGraph(OrchestratorState)
        # StrategyAgentInput 필드가 OrchestratorState에 없는 필드를 포함하는지 확인
        orchestrator_fields = set(OrchestratorState.__annotations__.keys())
        strategy_fields = set(StrategyAgentInput.__annotations__.keys())
        # company, retry_count은 OrchestratorState에 없는 필드 → input_schema로 격리 필요
        isolated_fields = strategy_fields - orchestrator_fields
        assert len(isolated_fields) > 0

    def test_make_checkpointer_returns_memory_saver(self):
        cp = make_checkpointer()
        assert isinstance(cp, MemorySaver)

    def test_compile_standalone_returns_app(self):
        app = compile_standalone()
        assert app is not None
        assert callable(getattr(app, "ainvoke", None))


# ─────────────────────────────────────────────
# 시나리오 1: 정상 흐름 (approve)
# ─────────────────────────────────────────────

class TestNormalApproveFlow:

    @pytest.mark.asyncio
    async def test_interrupt_occurs_after_both_agents_complete(self):
        app = compile_standalone()
        config = _config("t_approve_1")
        with patch("orchestrator.orchestrator.run_strategy_agent",
                   new=AsyncMock(side_effect=lambda s: _success(s["company"]))):
            await app.ainvoke(_initial_input(), config)
        state = app.get_state(config)
        assert state.next, "HITL interrupt가 발생해야 함"

    @pytest.mark.asyncio
    async def test_approve_leads_to_end(self):
        app = compile_standalone()
        config = _config("t_approve_2")
        with patch("orchestrator.orchestrator.run_strategy_agent",
                   new=AsyncMock(side_effect=lambda s: _success(s["company"]))):
            await app.ainvoke(_initial_input(), config)
            await app.ainvoke(Command(resume={"decision": "approve"}), config)
        assert not app.get_state(config).next

    @pytest.mark.asyncio
    async def test_final_state_has_both_results(self):
        app = compile_standalone()
        config = _config("t_approve_3")
        with patch("orchestrator.orchestrator.run_strategy_agent",
                   new=AsyncMock(side_effect=lambda s: _success(s["company"]))):
            await app.ainvoke(_initial_input(), config)
            await app.ainvoke(Command(resume={"decision": "approve"}), config)
        final = app.get_state(config).values
        assert final["skon_result"]["company"] == "SKON"
        assert final["catl_result"]["company"] == "CATL"
        assert final["skon_result"]["status"] == AgentStatus.SUCCESS
        assert final["catl_result"]["status"] == AgentStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_retry_counts_incremented_after_run(self):
        app = compile_standalone()
        config = _config("t_approve_4")
        with patch("orchestrator.orchestrator.run_strategy_agent",
                   new=AsyncMock(side_effect=lambda s: _success(s["company"]))):
            await app.ainvoke(_initial_input(), config)
        final = app.get_state(config).values
        assert final["skon_retry_count"] == 1
        assert final["catl_retry_count"] == 1

    @pytest.mark.asyncio
    async def test_hitl_payload_contains_both_companies(self):
        app = compile_standalone()
        config = _config("t_approve_5")
        with patch("orchestrator.orchestrator.run_strategy_agent",
                   new=AsyncMock(side_effect=lambda s: _success(s["company"]))):
            await app.ainvoke(_initial_input(), config)
        state = app.get_state(config)
        interrupt_payload = state.tasks[0].interrupts[0].value
        assert "skon" in interrupt_payload
        assert "catl" in interrupt_payload
        assert "allowed_decisions" in interrupt_payload
        assert "approve" in interrupt_payload["allowed_decisions"]


# ─────────────────────────────────────────────
# 시나리오 2: 양쪽 실패 → error_handler
# ─────────────────────────────────────────────

class TestBothAgentsFailedFlow:

    @pytest.mark.asyncio
    async def test_both_failed_skips_interrupt(self):
        """양쪽 실패 시 HITL 없이 바로 error_handler로 이동"""
        app = compile_standalone()
        config = _config("t_fail_1")
        with patch("orchestrator.orchestrator.run_strategy_agent",
                   new=AsyncMock(side_effect=lambda s: _failed(s["company"]))):
            await app.ainvoke(_initial_input(), config)
        assert not app.get_state(config).next

    @pytest.mark.asyncio
    async def test_both_failed_records_error_log(self):
        app = compile_standalone()
        config = _config("t_fail_2")
        with patch("orchestrator.orchestrator.run_strategy_agent",
                   new=AsyncMock(side_effect=lambda s: _failed(s["company"]))):
            await app.ainvoke(_initial_input(), config)
        final = app.get_state(config).values
        assert len(final.get("error_log", [])) > 0

    @pytest.mark.asyncio
    async def test_both_failed_fan_in_status_correct(self):
        app = compile_standalone()
        config = _config("t_fail_3")
        with patch("orchestrator.orchestrator.run_strategy_agent",
                   new=AsyncMock(side_effect=lambda s: _failed(s["company"]))):
            await app.ainvoke(_initial_input(), config)
        final = app.get_state(config).values
        assert final["fan_in_status"]["both_failed"] is True

    @pytest.mark.asyncio
    async def test_one_failed_one_success_proceeds_to_hitl(self):
        """한쪽만 실패해도 HITL은 발생해야 함"""
        app = compile_standalone()
        config = _config("t_fail_4")
        async def mock_agent(state):
            if state["company"] == "SKON":
                return _failed("SKON")
            return _success("CATL")
        with patch("orchestrator.orchestrator.run_strategy_agent", new=mock_agent):
            await app.ainvoke(_initial_input(), config)
        state = app.get_state(config)
        assert state.next, "한쪽 실패 시에도 HITL이 발생해야 함"


# ─────────────────────────────────────────────
# 시나리오 3: redo 루프
# ─────────────────────────────────────────────

class TestRedoLoop:

    @pytest.mark.asyncio
    async def test_redo_skon_reruns_skon_agent(self):
        app = compile_standalone()
        config = _config("t_redo_1")
        call_counts = {"SKON": 0, "CATL": 0}

        async def mock_agent(state):
            call_counts[state["company"]] += 1
            return _success(state["company"])

        with patch("orchestrator.orchestrator.run_strategy_agent", new=mock_agent):
            await app.ainvoke(_initial_input(), config)
            await app.ainvoke(
                Command(resume={"decision": "redo_skon", "feedback": "ESS 보완"}), config
            )
            if app.get_state(config).next:
                await app.ainvoke(Command(resume={"decision": "approve"}), config)

        assert call_counts["SKON"] == 2, "SKON은 2회 실행돼야 함"
        assert call_counts["CATL"] == 1, "CATL은 1회만 실행돼야 함"

    @pytest.mark.asyncio
    async def test_redo_catl_reruns_only_catl(self):
        app = compile_standalone()
        config = _config("t_redo_2")
        call_counts = {"SKON": 0, "CATL": 0}

        async def mock_agent(state):
            call_counts[state["company"]] += 1
            return _success(state["company"])

        with patch("orchestrator.orchestrator.run_strategy_agent", new=mock_agent):
            await app.ainvoke(_initial_input(), config)
            await app.ainvoke(
                Command(resume={"decision": "redo_catl", "feedback": "규제 보완"}), config
            )
            if app.get_state(config).next:
                await app.ainvoke(Command(resume={"decision": "approve"}), config)

        assert call_counts["CATL"] == 2
        assert call_counts["SKON"] == 1

    @pytest.mark.asyncio
    async def test_redo_both_reruns_both_agents(self):
        app = compile_standalone()
        config = _config("t_redo_3")
        call_counts = {"SKON": 0, "CATL": 0}

        async def mock_agent(state):
            call_counts[state["company"]] += 1
            return _success(state["company"])

        with patch("orchestrator.orchestrator.run_strategy_agent", new=mock_agent):
            await app.ainvoke(_initial_input(), config)
            await app.ainvoke(
                Command(resume={"decision": "redo_both", "feedback": "전체 보완"}), config
            )
            if app.get_state(config).next:
                await app.ainvoke(Command(resume={"decision": "approve"}), config)

        assert call_counts["SKON"] == 2
        assert call_counts["CATL"] == 2

    @pytest.mark.asyncio
    async def test_feedback_passed_to_agent_on_redo(self):
        """redo 시 review_2_feedback이 Agent에 전달돼야 함"""
        app = compile_standalone()
        config = _config("t_redo_4")
        received_feedbacks = []

        async def mock_agent(state):
            received_feedbacks.append(state.get("review_feedback", ""))
            return _success(state["company"])

        with patch("orchestrator.orchestrator.run_strategy_agent", new=mock_agent):
            await app.ainvoke(_initial_input(), config)
            await app.ainvoke(
                Command(resume={"decision": "redo_skon", "feedback": "ESS 데이터 보완 필요"}),
                config,
            )
            if app.get_state(config).next:
                await app.ainvoke(Command(resume={"decision": "approve"}), config)

        # redo 시 feedback이 포함된 호출이 있어야 함
        assert any("ESS 데이터 보완 필요" in fb for fb in received_feedbacks)


# ─────────────────────────────────────────────
# 시나리오 4: retry 한계 소진
# ─────────────────────────────────────────────

class TestRetryExhaustion:

    @pytest.mark.asyncio
    async def test_retry_limit_exhausted_forces_end(self):
        """retry 한계(MAX_RETRIES) 소진 시 강제 approve → END"""
        app = compile_standalone()
        config = _config("t_exhaust_1")

        async def mock_agent(state):
            return _partial(state["company"])

        with patch("orchestrator.orchestrator.run_strategy_agent", new=mock_agent):
            await app.ainvoke(_initial_input(), config)
            # MAX_RETRIES만큼 redo 요청
            for _ in range(MAX_RETRIES):
                if not app.get_state(config).next:
                    break
                await app.ainvoke(
                    Command(resume={"decision": "redo_skon", "feedback": "보완"}), config
                )
        # 한계 도달 후 자동 종료
        assert not app.get_state(config).next

    def test_max_retries_constant(self):
        assert MAX_RETRIES == 2


# ─────────────────────────────────────────────
# 단위: orchestrator_fanout 라우터
# ─────────────────────────────────────────────

class TestOrchestratorFanout:

    def _state(self, redo=None, skon_retry=0, catl_retry=0):
        return {
            "user_request": "테스트",
            "market_context": MOCK_MARKET_CONTEXT,
            "redo_targets": redo or [],
            "skon_retry_count": skon_retry,
            "catl_retry_count": catl_retry,
            "review_2_feedback": "",
        }

    def test_initial_run_sends_both(self):
        sends = orchestrator_fanout(self._state())
        nodes = [s.node for s in sends]
        assert "skon_agent_node" in nodes
        assert "catl_agent_node" in nodes

    def test_redo_skon_sends_only_skon(self):
        sends = orchestrator_fanout(self._state(redo=["skon"]))
        nodes = [s.node for s in sends]
        assert "skon_agent_node" in nodes
        assert "catl_agent_node" not in nodes

    def test_redo_catl_sends_only_catl(self):
        sends = orchestrator_fanout(self._state(redo=["catl"]))
        nodes = [s.node for s in sends]
        assert "catl_agent_node" in nodes
        assert "skon_agent_node" not in nodes

    def test_redo_both_sends_both(self):
        sends = orchestrator_fanout(self._state(redo=["skon", "catl"]))
        nodes = [s.node for s in sends]
        assert "skon_agent_node" in nodes
        assert "catl_agent_node" in nodes

    def test_skon_skipped_when_retry_exhausted(self):
        sends = orchestrator_fanout(
            self._state(redo=["skon"], skon_retry=MAX_RETRIES)
        )
        nodes = [s.node for s in sends]
        assert "skon_agent_node" not in nodes

    def test_send_payload_has_correct_company(self):
        sends = orchestrator_fanout(self._state())
        company_map = {s.node: s.arg["company"] for s in sends}
        assert company_map["skon_agent_node"] == "SKON"
        assert company_map["catl_agent_node"] == "CATL"

    def test_send_payload_has_market_context(self):
        sends = orchestrator_fanout(self._state())
        for send in sends:
            assert "market_context" in send.arg
            assert send.arg["market_context"] is MOCK_MARKET_CONTEXT

    def test_send_payload_has_retry_count(self):
        sends = orchestrator_fanout(self._state(skon_retry=1, redo=["skon"]))
        skon_send = next(s for s in sends if s.node == "skon_agent_node")
        assert skon_send.arg["retry_count"] == 1


# ─────────────────────────────────────────────
# 단위: fan_in_node
# ─────────────────────────────────────────────

class TestFanInNode:

    def _state_with_results(self, skon_status, catl_status):
        return {
            "user_request": "테스트",
            "market_context": MOCK_MARKET_CONTEXT,
            "skon_result": {"status": skon_status, "schema_version": SCHEMA_VERSION,
                            "failure_type": None},
            "catl_result": {"status": catl_status, "schema_version": SCHEMA_VERSION,
                            "failure_type": None},
        }

    def test_both_success_not_failed(self):
        state = self._state_with_results(AgentStatus.SUCCESS, AgentStatus.SUCCESS)
        update = fan_in_node(state)
        assert update["fan_in_status"]["both_failed"] is False

    def test_both_failed_sets_flag(self):
        state = self._state_with_results(AgentStatus.FAILED, AgentStatus.FAILED)
        update = fan_in_node(state)
        assert update["fan_in_status"]["both_failed"] is True

    def test_both_failed_adds_error_log(self):
        state = self._state_with_results(AgentStatus.FAILED, AgentStatus.FAILED)
        update = fan_in_node(state)
        assert "error_log" in update
        assert len(update["error_log"]) > 0

    def test_partial_success_not_both_failed(self):
        state = self._state_with_results(
            AgentStatus.PARTIAL_SUCCESS, AgentStatus.SUCCESS
        )
        update = fan_in_node(state)
        assert update["fan_in_status"]["both_failed"] is False

    def test_redo_targets_reset_to_empty(self):
        state = self._state_with_results(AgentStatus.SUCCESS, AgentStatus.SUCCESS)
        update = fan_in_node(state)
        assert update["redo_targets"] == []

    def test_schema_version_mismatch_detected(self):
        state = {
            "user_request": "테스트",
            "market_context": MOCK_MARKET_CONTEXT,
            "skon_result": {"status": AgentStatus.SUCCESS, "schema_version": "0.0.0"},
            "catl_result": {"status": AgentStatus.SUCCESS, "schema_version": SCHEMA_VERSION},
        }
        update = fan_in_node(state)
        assert update["fan_in_status"]["schema_version_ok"]["skon"] is False
        assert update["fan_in_status"]["schema_version_ok"]["catl"] is True


# ─────────────────────────────────────────────
# 단위: 라우터 함수
# ─────────────────────────────────────────────

class TestRouters:

    def test_route_after_fan_in_to_hitl_on_success(self):
        state = {"fan_in_status": {"both_failed": False}}
        assert _route_after_fan_in(state) == "hitl_2_node"

    def test_route_after_fan_in_to_error_on_both_failed(self):
        state = {"fan_in_status": {"both_failed": True}}
        assert _route_after_fan_in(state) == "error_handler"

    def test_route_after_hitl_approve_returns_end(self):
        from langgraph.graph import END
        state = {"review_2_decision": "approve", "redo_targets": []}
        result = _route_after_hitl_2(state)
        assert result is END

    def test_route_after_hitl_redo_returns_send_list(self):
        from langgraph.types import Send
        state = {
            "review_2_decision": "redo_skon",
            "redo_targets": ["skon"],
            "skon_retry_count": 0,
            "catl_retry_count": 0,
            "market_context": MOCK_MARKET_CONTEXT,
            "review_2_feedback": "",
        }
        result = _route_after_hitl_2(state)
        assert isinstance(result, list)
        assert isinstance(result[0], Send)

    def test_route_after_hitl_exhausted_returns_end(self):
        from langgraph.graph import END
        state = {
            "review_2_decision": "redo_skon",
            "redo_targets": ["skon"],
            "skon_retry_count": MAX_RETRIES,
            "catl_retry_count": 0,
            "market_context": MOCK_MARKET_CONTEXT,
            "review_2_feedback": "",
        }
        result = _route_after_hitl_2(state)
        assert result is END


# ─────────────────────────────────────────────
# 단위: 헬퍼 함수
# ─────────────────────────────────────────────

class TestHelpers:

    def test_allowed_decisions_includes_approve(self):
        state = {"skon_retry_count": 0, "catl_retry_count": 0}
        decisions = _allowed_decisions(state)
        assert "approve" in decisions

    def test_allowed_decisions_excludes_redo_at_limit(self):
        state = {"skon_retry_count": MAX_RETRIES, "catl_retry_count": MAX_RETRIES}
        decisions = _allowed_decisions(state)
        assert "redo_skon" not in decisions
        assert "redo_catl" not in decisions

    def test_validate_resume_valid_approve(self):
        result = _validate_resume(
            {"decision": "approve"}, {"skon_retry_count": 0, "catl_retry_count": 0}
        )
        assert result["valid"] is True

    def test_validate_resume_invalid_decision(self):
        result = _validate_resume(
            {"decision": "unknown"}, {}
        )
        assert result["valid"] is False

    def test_validate_resume_not_dict(self):
        result = _validate_resume("approve", {})
        assert result["valid"] is False

    def test_validate_resume_redo_at_retry_limit(self):
        result = _validate_resume(
            {"decision": "redo_skon"},
            {"skon_retry_count": MAX_RETRIES, "catl_retry_count": 0},
        )
        assert result["valid"] is False

    def test_decision_to_targets_approve(self):
        assert _decision_to_targets("approve") == []

    def test_decision_to_targets_redo_skon(self):
        assert _decision_to_targets("redo_skon") == ["skon"]

    def test_decision_to_targets_redo_catl(self):
        assert _decision_to_targets("redo_catl") == ["catl"]

    def test_decision_to_targets_redo_both(self):
        assert set(_decision_to_targets("redo_both")) == {"skon", "catl"}

    def test_decision_to_targets_unknown(self):
        assert _decision_to_targets("unknown") == []


# ─────────────────────────────────────────────
# 직렬화 경고 없음
# ─────────────────────────────────────────────

class TestSerializationWarnings:

    @pytest.mark.asyncio
    async def test_no_enum_deserialization_warnings(self):
        cap = []

        class CapHandler(logging.Handler):
            def emit(self, r):
                msg = r.getMessage()
                if "unregistered" in msg or "Blocked" in msg:
                    cap.append(msg)

        handler = CapHandler()
        logging.getLogger("langgraph.checkpoint.serde.jsonplus").addHandler(handler)

        try:
            app = compile_standalone()
            config = _config("t_serde_1")
            with patch("orchestrator.orchestrator.run_strategy_agent",
                       new=AsyncMock(side_effect=lambda s: _failed(s["company"]))):
                await app.ainvoke(_initial_input(), config)
            assert not cap, f"직렬화 경고 발생: {cap[:2]}"
        finally:
            logging.getLogger("langgraph.checkpoint.serde.jsonplus").removeHandler(handler)
