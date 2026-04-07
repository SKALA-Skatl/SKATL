"""
schemas 패키지 public API.

외부에서:
    from schemas import OrchestratorInput, OrchestratorOutput
    from schemas import AgentStatus, StrategyAgentOutput
    from schemas import MarketContext, MOCK_MARKET_CONTEXT
"""

from schemas.state import (
    OrchestratorInput,
    OrchestratorOutput,
    OrchestratorState,
    assert_immutable_fields,
)
from schemas.agent_io import (
    AgentStatus,
    AgentFailureType,
    SourceType,
    SourceRecord,
    FindingWithSource,
    ConfidenceScores,
    StrategyAgentInput,
    StrategyAgentOutput,
    SCHEMA_VERSION,
    validate_schema_version,
)
from schemas.confidence import (
    evaluate_source_credibility,
    calculate_confidence_scores,
    AXIS_FIELDS,
)
from schemas.market_context import (
    MarketContext,
    MOCK_MARKET_CONTEXT,
)
from schemas.market_agent_io import (
    MarketAgentInput,
    MarketAgentOutput,
)
from schemas.phase1_state import (
    Phase1Input,
    Phase1Output,
    Phase1State,
    assert_phase1_immutable_fields,
)

__all__ = [
    # state
    "OrchestratorInput",
    "OrchestratorOutput",
    "OrchestratorState",
    "assert_immutable_fields",
    # agent_io
    "AgentStatus",
    "AgentFailureType",
    "SourceType",
    "SourceRecord",
    "FindingWithSource",
    "ConfidenceScores",
    "StrategyAgentInput",
    "StrategyAgentOutput",
    "SCHEMA_VERSION",
    "validate_schema_version",
    # confidence
    "evaluate_source_credibility",
    "calculate_confidence_scores",
    "AXIS_FIELDS",
    # market_context
    "MarketContext",
    "MOCK_MARKET_CONTEXT",
    "MarketAgentInput",
    "MarketAgentOutput",
    "Phase1Input",
    "Phase1Output",
    "Phase1State",
    "assert_phase1_immutable_fields",
]
