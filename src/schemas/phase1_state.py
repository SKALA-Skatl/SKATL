"""
Phase 1 market workflow state schema.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from schemas.market_agent_io import MarketAgentOutput
from schemas.market_context import MarketContext
from schemas.state import OrchestratorInput


def _last_write(existing, new):
    """Prefer the latest write in the graph."""

    return new if new is not None else existing


def _append_errors(existing: list, new: list) -> list:
    """Append error records."""

    return (existing or []) + (new or [])


class Phase1Input(TypedDict):
    """Initial input for the market workflow."""

    user_request: str


class Phase1Output(TypedDict):
    """Final output passed to the orchestrator."""

    user_request: str
    market_context: MarketContext


class Phase1State(TypedDict, total=False):
    """Full state used inside the Phase 1 graph."""

    user_request: str
    market_context: Annotated[MarketContext, _last_write]
    market_result: Annotated[MarketAgentOutput, _last_write]
    retry_count: int
    review_1_decision: Literal["approve", "redo"]
    review_1_feedback: str
    error_log: Annotated[list[dict], _append_errors]
    collected_sources: Annotated[list[dict], operator.add]


_IMMUTABLE = frozenset(Phase1Input.__annotations__.keys())


def assert_phase1_immutable_fields(update: dict, node_name: str) -> None:
    """Prevent nodes from overwriting immutable input fields."""

    violations = _IMMUTABLE & update.keys()
    if violations:
        raise RuntimeError(f"[{node_name}] 입력 필드 write 금지: {violations}")
