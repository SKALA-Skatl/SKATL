"""
prompts 패키지 public API.

외부에서:
    from prompts import build_system_prompt
    from prompts import COMPANY_PROMPTS, CompanyPromptConfig
"""

from prompts.strategy_prompt import (
    build_system_prompt,
    COMPANY_PROMPTS,
    CompanyPromptConfig,
)
from prompts.market_prompt import build_market_system_prompt

__all__ = [
    "build_system_prompt",
    "build_market_system_prompt",
    "COMPANY_PROMPTS",
    "CompanyPromptConfig",
]
