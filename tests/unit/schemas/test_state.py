"""
schemas/state.py 단위 테스트.

검증 항목:
  - OrchestratorInput / OrchestratorOutput 필드 정의
  - OrchestratorState가 Input/Output 필드를 포함
  - _last_write reducer: None 처리
  - _append_errors reducer: 누적 동작
  - assert_immutable_fields(): 위반 시 RuntimeError
"""

import pytest
from schemas.state import (
    OrchestratorInput,
    OrchestratorOutput,
    OrchestratorState,
    assert_immutable_fields,
    _last_write,
    _append_errors,
)


class TestOrchestratorInterface:

    def test_input_has_required_fields(self):
        fields = OrchestratorInput.__annotations__
        assert "user_request"   in fields
        assert "market_context" in fields

    def test_output_has_required_fields(self):
        fields = OrchestratorOutput.__annotations__
        assert "skon_result" in fields
        assert "catl_result" in fields

    def test_state_contains_input_fields(self):
        """OrchestratorState가 Input 필드를 포함해야 서브그래프 편입 시 자동 전달됨"""
        state_fields = OrchestratorState.__annotations__
        for key in OrchestratorInput.__annotations__:
            assert key in state_fields, f"Input 필드 '{key}'가 State에 없음"

    def test_state_contains_output_fields(self):
        """OrchestratorState가 Output 필드를 포함해야 결과 전파 가능"""
        state_fields = OrchestratorState.__annotations__
        for key in OrchestratorOutput.__annotations__:
            assert key in state_fields, f"Output 필드 '{key}'가 State에 없음"

    def test_state_has_internal_control_fields(self):
        fields = OrchestratorState.__annotations__
        assert "skon_retry_count"  in fields
        assert "catl_retry_count"  in fields
        assert "redo_targets"      in fields
        assert "review_2_decision" in fields
        assert "review_2_feedback" in fields
        assert "fan_in_status"     in fields
        assert "error_log"         in fields


class TestReducers:

    def test_last_write_returns_new(self):
        assert _last_write("old", "new") == "new"

    def test_last_write_returns_existing_when_new_is_none(self):
        assert _last_write("existing", None) == "existing"

    def test_last_write_both_none(self):
        assert _last_write(None, None) is None

    def test_last_write_with_dict(self):
        old = {"status": "failed"}
        new = {"status": "success"}
        assert _last_write(old, new) == new

    def test_append_errors_accumulates(self):
        existing = [{"event": "error_1"}]
        new      = [{"event": "error_2"}]
        result   = _append_errors(existing, new)
        assert len(result) == 2
        assert result[0]["event"] == "error_1"
        assert result[1]["event"] == "error_2"

    def test_append_errors_none_existing(self):
        result = _append_errors(None, [{"event": "first"}])
        assert len(result) == 1

    def test_append_errors_none_new(self):
        result = _append_errors([{"event": "existing"}], None)
        assert len(result) == 1

    def test_append_errors_both_none(self):
        result = _append_errors(None, None)
        assert result == []


class TestAssertImmutableFields:

    def test_mutable_fields_pass(self):
        """가변 필드만 포함된 update는 예외 없음"""
        update = {
            "skon_result": {},
            "skon_retry_count": 1,
            "fan_in_status": {},
        }
        assert_immutable_fields(update, "test_node")  # 예외 없어야 함

    def test_user_request_raises(self):
        with pytest.raises(RuntimeError, match="입력 필드 write 금지"):
            assert_immutable_fields({"user_request": "변경 시도"}, "bad_node")

    def test_market_context_raises(self):
        with pytest.raises(RuntimeError, match="입력 필드 write 금지"):
            assert_immutable_fields({"market_context": {}}, "bad_node")

    def test_error_message_includes_node_name(self):
        with pytest.raises(RuntimeError, match="my_node"):
            assert_immutable_fields({"user_request": "x"}, "my_node")

    def test_error_message_includes_field_name(self):
        with pytest.raises(RuntimeError, match="user_request"):
            assert_immutable_fields({"user_request": "x"}, "node")

    def test_empty_update_passes(self):
        assert_immutable_fields({}, "node")  # 예외 없어야 함
