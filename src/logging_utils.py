"""
로깅 유틸리티.

관측 가능성 목표:
  - 노드별 진입/종료 시간, duration
  - tool 호출 횟수 및 종류
  - RAG cosine score 분포
  - LLM 호출 횟수
  - 에러 발생 위치 및 유형
"""

from __future__ import annotations
import logging
import time
from contextlib import contextmanager
from typing import Any


# ─────────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────────

def get_logger(name: str) -> "Logger":
    return Logger(name)


class Logger:
    def __init__(self, name: str):
        self._log = logging.getLogger(f"battery_strategy.{name}")
        if not self._log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s %(message)s"
            ))
            self._log.addHandler(handler)
        self._log.setLevel(logging.INFO)

    # ── 노드 ──────────────────────────────────

    def node_enter(self, node_name: str, metadata: dict | None = None) -> None:
        self._log.info(
            "[NODE_ENTER] node=%s meta=%s",
            node_name,
            metadata or {},
        )

    def node_exit(
        self,
        node_name: str,
        duration_sec: float,
        status: str = "ok",
        metadata: dict | None = None,
    ) -> None:
        self._log.info(
            "[NODE_EXIT] node=%s status=%s duration=%.2fs meta=%s",
            node_name,
            status,
            duration_sec,
            metadata or {},
        )

    # ── Tool 호출 ─────────────────────────────

    def tool_call(self, tool_name: str, query: str, metadata: dict | None = None) -> None:
        self._log.info(
            "[TOOL_CALL] tool=%s query=%r meta=%s",
            tool_name,
            query[:120],    # 너무 길면 잘라서 로깅
            metadata or {},
        )

    def tool_result(
        self,
        tool_name: str,
        success: bool,
        metadata: dict | None = None,
    ) -> None:
        level = logging.INFO if success else logging.WARNING
        self._log.log(
            level,
            "[TOOL_RESULT] tool=%s success=%s meta=%s",
            tool_name,
            success,
            metadata or {},
        )

    # ── RAG ───────────────────────────────────

    def rag_search(
        self,
        query: str,
        cosine_scores: list[float],
        rewrite_count: int,
        forced_return: bool = False,
    ) -> None:
        self._log.info(
            "[RAG_SEARCH] query=%r scores=%s rewrite_count=%d forced=%s",
            query[:120],
            [round(s, 3) for s in cosine_scores],
            rewrite_count,
            forced_return,
        )

    # ── LLM ───────────────────────────────────

    def llm_call(self, purpose: str, model: str) -> None:
        self._log.info("[LLM_CALL] purpose=%s model=%s", purpose, model)

    # ── 에러 ──────────────────────────────────

    def error(self, node_name: str, error: Exception, metadata: dict | None = None) -> None:
        self._log.error(
            "[ERROR] node=%s type=%s msg=%s meta=%s",
            node_name,
            type(error).__name__,
            str(error),
            metadata or {},
        )

    # ── 편의 컨텍스트 매니저 ──────────────────

    @contextmanager
    def node_span(self, node_name: str, metadata: dict | None = None):
        """
        with logger.node_span("skon_agent") as span:
            span["llm_calls"] = 0
            ...
        자동으로 enter/exit 로그 + duration 측정.
        """
        span: dict[str, Any] = {}
        self.node_enter(node_name, metadata)
        t0 = time.perf_counter()
        try:
            yield span
            self.node_exit(
                node_name,
                duration_sec=time.perf_counter() - t0,
                status="ok",
                metadata=span,
            )
        except Exception as e:
            self.error(node_name, e, metadata=span)
            self.node_exit(
                node_name,
                duration_sec=time.perf_counter() - t0,
                status="error",
                metadata=span,
            )
            raise
