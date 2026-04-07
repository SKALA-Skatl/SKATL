"""
tools/rag_tool.py 단위 테스트.

검증 항목:
  - initialize_rag_pipelines() 호출 전 get 함수가 RuntimeError 발생
  - make_skon_rag_tool() / make_catl_rag_tool()이 tool 객체 반환
  - tool에 적절한 docstring이 있음
  - _format_rag_result(): 문서 없을 때 / 있을 때 포맷
"""

import pytest
from unittest.mock import patch, MagicMock

from tools.rag_tool import (
    _get_skon_rag,
    _get_catl_rag,
    make_skon_rag_tool,
    make_catl_rag_tool,
    _format_rag_result,
)
from tools.rag_pipeline import RAGDocument, RAGResult
from schemas.agent_io import SourceType


class TestInitializationGuard:

    def test_get_skon_rag_raises_before_init(self):
        """초기화 전 호출 시 RuntimeError"""
        import tools.rag_tool as rt
        original = rt._SKON_RAG
        rt._SKON_RAG = None
        try:
            with pytest.raises(RuntimeError, match="initialize_rag_pipelines"):
                _get_skon_rag()
        finally:
            rt._SKON_RAG = original

    def test_get_catl_rag_raises_before_init(self):
        import tools.rag_tool as rt
        original = rt._CATL_RAG
        rt._CATL_RAG = None
        try:
            with pytest.raises(RuntimeError, match="initialize_rag_pipelines"):
                _get_catl_rag()
        finally:
            rt._CATL_RAG = original

    def test_get_skon_rag_returns_pipeline_after_mock_init(self):
        import tools.rag_tool as rt
        mock_pipeline = MagicMock()
        original = rt._SKON_RAG
        rt._SKON_RAG = mock_pipeline
        try:
            result = _get_skon_rag()
            assert result is mock_pipeline
        finally:
            rt._SKON_RAG = original


class TestToolFactory:

    def test_make_skon_rag_tool_returns_structured_tool(self):
        """LangChain StructuredTool은 callable()이 False — ainvoke로 확인"""
        from langchain_core.tools import BaseTool
        tool = make_skon_rag_tool()
        assert isinstance(tool, BaseTool)

    def test_make_catl_rag_tool_returns_structured_tool(self):
        from langchain_core.tools import BaseTool
        tool = make_catl_rag_tool()
        assert isinstance(tool, BaseTool)

    def test_skon_tool_has_docstring(self):
        tool = make_skon_rag_tool()
        assert tool.description
        assert "SK On" in tool.description or "skon" in tool.description.lower()

    def test_catl_tool_has_docstring(self):
        tool = make_catl_rag_tool()
        assert tool.description
        assert "CATL" in tool.description or "catl" in tool.description.lower()

    def test_tool_name_skon(self):
        tool = make_skon_rag_tool()
        assert "skon" in tool.name.lower()

    def test_tool_name_catl(self):
        tool = make_catl_rag_tool()
        assert "catl" in tool.name.lower()


class TestFormatRagResult:

    def test_empty_documents_returns_not_found_message(self):
        result = RAGResult(
            documents=[], query_used="쿼리", rewrite_count=0
        )
        formatted = _format_rag_result(result)
        assert "찾지 못했습니다" in formatted

    def test_documents_included_in_output(self):
        result = RAGResult(
            documents=[
                RAGDocument("d1", "문서 내용", "https://a.com", "제목1", 0.85),
            ],
            query_used="SK On 전략",
            rewrite_count=0,
            source_type=SourceType.RAG_FAISS,
        )
        formatted = _format_rag_result(result)
        assert "SK On 전략" in formatted
        assert "제목1" in formatted
        assert "문서 내용" in formatted

    def test_forced_return_indicated_in_output(self):
        result = RAGResult(
            documents=[RAGDocument("d1", "내용", "https://a.com", "제목", 0.50)],
            query_used="쿼리",
            rewrite_count=2,
            forced_return=True,
        )
        formatted = _format_rag_result(result)
        assert "강제 반환" in formatted

    def test_rewrite_count_shown_in_output(self):
        result = RAGResult(
            documents=[RAGDocument("d1", "내용", "https://a.com", "제목", 0.80)],
            query_used="쿼리",
            rewrite_count=1,
        )
        formatted = _format_rag_result(result)
        assert "재작성 1회" in formatted
