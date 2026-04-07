"""
tools/web_search_tool.py 단위 테스트.

검증 항목:
  - web_search가 LangChain tool로 등록됨
  - Tavily 클라이언트가 lazy 초기화됨
  - Tavily 결과를 적절한 형식으로 포맷
  - Tavily 오류 시 에러 메시지 반환 (예외 전파 없음)
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestWebSearchToolDefinition:

    def test_web_search_is_langchain_tool(self):
        """LangChain StructuredTool은 callable()이 False — BaseTool 인스턴스로 확인"""
        from langchain_core.tools import BaseTool
        from tools.web_search_tool import web_search
        assert isinstance(web_search, BaseTool)
        assert hasattr(web_search, "name")
        assert hasattr(web_search, "description")

    def test_web_search_tool_name(self):
        from tools.web_search_tool import web_search
        assert web_search.name == "web_search"

    def test_web_search_has_description(self):
        from tools.web_search_tool import web_search
        assert web_search.description
        assert len(web_search.description) > 10

    def test_tavily_client_lazy_initialized(self):
        """모듈 import 시점에 Tavily 클라이언트가 생성되지 않아야 함"""
        import tools.web_search_tool as ws
        assert ws._TAVILY_CLIENT is None


class TestWebSearchExecution:

    @pytest.mark.asyncio
    async def test_returns_results_on_success(self):
        mock_response = {
            "results": [
                {
                    "title": "SK On 배터리 전략",
                    "url": "https://example.com/skon",
                    "raw_content": "SK On은 캐즘 대응으로 비용 절감 전략을 추진했습니다.",
                }
            ]
        }
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value=mock_response)

        with patch("tools.web_search_tool._TAVILY_CLIENT", mock_client):
            from tools.web_search_tool import web_search
            result = await web_search.ainvoke({"query": "SK On 전략"})

        assert isinstance(result, str)
        assert "SK On 배터리 전략" in result
        assert "https://example.com/skon" in result

    @pytest.mark.asyncio
    async def test_returns_message_on_empty_results(self):
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value={"results": []})

        with patch("tools.web_search_tool._TAVILY_CLIENT", mock_client):
            from tools.web_search_tool import web_search
            result = await web_search.ainvoke({"query": "존재하지 않는 쿼리"})

        assert "검색 결과가 없습니다" in result

    @pytest.mark.asyncio
    async def test_returns_error_message_on_exception(self):
        """Tavily 예외 시 예외가 전파되지 않고 에러 메시지 반환"""
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(side_effect=Exception("API 오류"))

        with patch("tools.web_search_tool._TAVILY_CLIENT", mock_client):
            from tools.web_search_tool import web_search
            result = await web_search.ainvoke({"query": "쿼리"})

        assert isinstance(result, str)
        assert "오류" in result

    @pytest.mark.asyncio
    async def test_credibility_score_in_output(self):
        """출처 신뢰도 스코어가 결과에 포함돼야 함"""
        mock_response = {
            "results": [
                {"title": "제목", "url": "https://x.com", "raw_content": "내용"}
            ]
        }
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value=mock_response)

        with patch("tools.web_search_tool._TAVILY_CLIENT", mock_client):
            from tools.web_search_tool import web_search
            result = await web_search.ainvoke({"query": "쿼리"})

        assert "신뢰도" in result
