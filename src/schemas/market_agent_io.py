"""
Market Agent input/output schema.
"""

from __future__ import annotations

from typing import TypedDict

from schemas.agent_io import AgentFailureType, AgentStatus, SourceRecord
from schemas.market_context import MarketContext


class MarketAgentInput(TypedDict):
    """Input payload for the market agent."""

    user_request: str
    review_feedback: str
    retry_count: int


class MarketAgentOutput(TypedDict, total=False):
    """Output payload from the market agent."""

    status: AgentStatus
    failure_type: AgentFailureType | None
    analysis_timestamp: str
    llm_call_count: int
    tool_call_log: list[dict]
    market_context: MarketContext
    sources: list[SourceRecord]
