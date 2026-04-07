"""
SKATL 전체 파이프라인 진입점.

명령어:
  python app.py build-indices          : FAISS 인덱스 빌드
  python app.py run [--request "..."]  : 전체 워크플로우 실행 (대화형 HITL)

워크플로우 (다이어그램 참조):
  Phase 1  : Market Agent — 시장 배경·전기차 캐즘 분석
  HITL #1  : Human Review — 충분성·조사 범위 조정
               ↳ Phase 1 서브그래프 내부에서 interrupt 처리
  Phase 2  : Central Orchestrator — SKON/CATL 병렬 전략 분석 (Fan-out/Fan-in)
  HITL #2  : Human Review — 충분성·비교축 확인
               ↳ Orchestrator 서브그래프 내부에서 interrupt 처리
  Phase 3  : Comparative SWOT Agent — SK On vs CATL 비교·SWOT 생성
  Phase 4  : Report Agent — 최종 보고서 초안 생성
  HITL #3  : Human Review — 객관성·보고서 내용 확인
  Publish  : Word(.docx) 저장

전역 FAISS 공유 전략:
  앱 시작 시 모든 컬렉션 인덱스를 한 번 로드 →
  tools.rag_tool.initialize_rag_pipelines_with_stores() 로 주입 →
  Market / SKON / CATL 에이전트가 전역 RAGPipeline 인스턴스를 공유
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

# ── sys.path: src/ 내부 bare import 지원 ─────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── 환경변수 ──────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── 내부 모듈 (sys.path 설정 이후) ────────────
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from agents.comparative_swot import run_comparative_swot
from agents.report_agent import build_word_report, run_report_agent
from orchestrator.orchestrator import build_graph as build_orchestrator_graph
from phase1.market_phase import build_graph as build_phase1_graph
from rag import RAGConfig, build_documents_from_paths, get_collection_names
from rag.vectorstore import build_and_save_indices, load_index
from schemas.agent_io import StrategyAgentOutput
from schemas.market_context import MarketContext
from tools.rag_tool import initialize_rag_pipelines_with_stores
from logging_utils import get_logger


logger = get_logger("app")

MAX_REPORT_RETRIES = 3   # HITL #3 최대 거절 횟수


# ─────────────────────────────────────────────
# 전체 파이프라인 State
#
# Phase 1 서브그래프와 공유: user_request, market_context, error_log
# Orchestrator 서브그래프와 공유: user_request, market_context,
#                                 skon_result, catl_result, error_log
# ─────────────────────────────────────────────

def _last_write(existing, new):
    return new if new is not None else existing


def _append_list(existing: list, new: list) -> list:
    return (existing or []) + (new or [])


class PipelineState(TypedDict, total=False):
    # ── Phase 1 / Orchestrator 공유 키 ─────────
    user_request:   str
    market_context: Annotated[MarketContext, _last_write]
    skon_result:    Annotated[StrategyAgentOutput, _last_write]
    catl_result:    Annotated[StrategyAgentOutput, _last_write]
    error_log:      Annotated[list, _append_list]

    # ── Phase 3 ───────────────────────────────
    comparative_swot: dict

    # ── Phase 4 ───────────────────────────────
    report_draft: dict

    # ── HITL #3 ───────────────────────────────
    human_decision:     str           # "approve" | "reject"
    human_feedback:     str
    review_round:       int
    max_review_rounds:  int
    review_history:     Annotated[list, _append_list]
    final_revision_mode: bool

    # ── 최종 출력 ──────────────────────────────
    final_status:      str
    final_report_path: str


# ─────────────────────────────────────────────
# Phase 3: Comparative SWOT 노드
# ─────────────────────────────────────────────

async def swot_node(state: PipelineState) -> dict:
    with logger.node_span("swot_node"):
        result = await run_comparative_swot(
            market_context=state["market_context"],
            skon_result=state["skon_result"],
            catl_result=state["catl_result"],
            human_feedback=state.get("human_feedback", ""),
            final_revision_mode=state.get("final_revision_mode", False),
        )
    return {"comparative_swot": result}


# ─────────────────────────────────────────────
# Phase 4: Report 노드
# ─────────────────────────────────────────────

async def report_node(state: PipelineState) -> dict:
    with logger.node_span("report_node"):
        result = await run_report_agent(
            market_context=state["market_context"],
            skon_result=state["skon_result"],
            catl_result=state["catl_result"],
            comparative_swot=state.get("comparative_swot", {}),
            human_feedback=state.get("human_feedback", ""),
            final_revision_mode=state.get("final_revision_mode", False),
        )
    return {"report_draft": result}


# ─────────────────────────────────────────────
# HITL #3 노드
# ─────────────────────────────────────────────

def _normalize_review_3(response) -> dict:
    """interrupt resume 값을 {"decision": ..., "feedback": ...} 로 정규화."""
    if isinstance(response, dict) and "decisions" in response:
        decisions = response.get("decisions", [])
        if not decisions:
            return {"decision": "reject", "feedback": "No decision provided."}
        first = decisions[0]
        dt = str(first.get("type", "reject")).strip().lower()
        return {
            "decision": "approve" if dt == "approve" else "reject",
            "feedback": "" if dt == "approve" else str(first.get("message", "")).strip(),
        }
    if isinstance(response, dict):
        decision = str(response.get("decision", "reject")).strip().lower()
        feedback = str(response.get("feedback", "")).strip()
        return {"decision": decision if decision in {"approve", "reject"} else "reject",
                "feedback": feedback}
    return {"decision": "reject", "feedback": "Invalid response format."}


def hitl_3_node(state: PipelineState) -> dict:
    """
    Human Review #3 — 보고서 객관성·내용 확인.

    interrupt payload:
      stage, review_round, max_review_rounds, action_requests, review_configs

    resume 형식:
      {"decision": "approve"|"reject", "feedback": "<수정 지시>"}
      또는 {"decisions": [{"type": "approve"|"reject", "message": "..."}]}
    """
    next_round = state.get("review_round", 0) + 1
    response   = interrupt({
        "stage":              "human_review_3",
        "review_round":       next_round,
        "max_review_rounds":  state.get("max_review_rounds", MAX_REPORT_RETRIES),
        "action_requests": [{
            "name": "review_report_draft",
            "args": {
                "report_draft":    state.get("report_draft"),
                "comparative_swot": state.get("comparative_swot"),
            },
            "description": "보고서 초안과 Comparative SWOT를 검토하고 승인 또는 수정 요청",
        }],
        "review_configs": [{
            "action_name":       "review_report_draft",
            "allowed_decisions": ["approve", "reject"],
        }],
    })

    normalized = _normalize_review_3(response)
    new_history = [{
        "review_round": next_round,
        "decision":     normalized["decision"],
        "feedback":     normalized["feedback"],
    }]
    return {
        "human_decision":     normalized["decision"],
        "human_feedback":     normalized["feedback"],
        "review_round":       next_round,
        "review_history":     new_history,   # _append_list reducer로 누적
        "final_revision_mode": False,
    }


def _route_after_hitl_3(state: PipelineState) -> str:
    decision     = state.get("human_decision", "reject")
    review_round = state.get("review_round", 0)
    max_rounds   = state.get("max_review_rounds", MAX_REPORT_RETRIES)

    if decision == "approve":
        return "publish_node"
    if review_round >= max_rounds:
        return "final_revision_node"
    return "swot_node"


# ─────────────────────────────────────────────
# 최종 수정 모드 / 저장 노드
# ─────────────────────────────────────────────

def final_revision_node(_: PipelineState) -> dict:
    """최대 리뷰 횟수 도달 → final_revision_mode 활성화 후 SWOT 재실행."""
    return {
        "final_revision_mode": True,
        "final_status":        "auto_publish_after_max_rejects",
    }


def publish_node(state: PipelineState) -> dict:
    """보고서 초안을 Word(.docx) 또는 JSON으로 저장한다."""
    output_dir  = PROJECT_ROOT / "results"
    report_path = output_dir / "battery_market_strategy_report.docx"

    try:
        build_word_report(
            report=state.get("report_draft", {}),
            swot=state.get("comparative_swot", {}),
            output_path=report_path,
        )
        final_status = state.get("final_status", "approved")
    except ImportError as e:
        print(f"\n[경고] Word 저장 실패 — {e}")
        final_status = "approved_no_docx"
        report_path  = output_dir / "battery_market_strategy_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(state.get("report_draft", {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return {
        "final_status":      final_status,
        "final_report_path": str(report_path),
    }


# ─────────────────────────────────────────────
# 파이프라인 그래프 빌더
# ─────────────────────────────────────────────

def _make_checkpointer() -> MemorySaver:
    """Enum 타입을 허용 목록에 등록한 통합 MemorySaver."""
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("schemas.agent_io", "AgentStatus"),
            ("schemas.agent_io", "AgentFailureType"),
            ("schemas.agent_io", "SourceType"),
        ]
    )
    return MemorySaver(serde=serde)


def build_pipeline():
    """
    전체 파이프라인 그래프를 컴파일해 반환.

    서브그래프:
      phase1_node       : Phase1State — market agent + HITL #1 내장
      orchestrator_node : OrchestratorState — SKON/CATL + HITL #2 내장

    파이프라인 흐름:
      START → phase1_node → orchestrator_node
            → swot_node → report_node → hitl_3_node
            → [approve → publish_node → END]
            → [reject  → swot_node  (재실행 루프, max 3회)]
            → [max 도달 → final_revision_node → swot_node]
    """
    # 서브그래프: 자체 checkpointer 없이 컴파일 → 부모 checkpointer 공유
    phase1_subgraph       = build_phase1_graph().compile()
    orchestrator_subgraph = build_orchestrator_graph().compile()

    builder = StateGraph(PipelineState)

    builder.add_node("phase1_node",        phase1_subgraph)
    builder.add_node("orchestrator_node",  orchestrator_subgraph)
    builder.add_node("swot_node",          swot_node)
    builder.add_node("report_node",        report_node)
    builder.add_node("hitl_3_node",        hitl_3_node)
    builder.add_node("final_revision_node", final_revision_node)
    builder.add_node("publish_node",       publish_node)

    builder.add_edge(START,              "phase1_node")
    builder.add_edge("phase1_node",      "orchestrator_node")
    builder.add_edge("orchestrator_node", "swot_node")
    builder.add_edge("swot_node",        "report_node")
    builder.add_edge("report_node",      "hitl_3_node")

    builder.add_conditional_edges(
        "hitl_3_node",
        _route_after_hitl_3,
        {
            "publish_node":        "publish_node",
            "swot_node":           "swot_node",
            "final_revision_node": "final_revision_node",
        },
    )

    builder.add_edge("final_revision_node", "swot_node")
    builder.add_edge("publish_node",        END)

    return builder.compile(checkpointer=_make_checkpointer())


# ─────────────────────────────────────────────
# 전역 FAISS 로드
# ─────────────────────────────────────────────

def _load_and_init_vectorstores(config: RAGConfig) -> None:
    """
    모든 컬렉션 FAISS 인덱스를 한 번 로드하고
    RAG 파이프라인을 전역으로 초기화한다.

    로드 순서: market_agent → skon_agent → catl_agent
    (swot_agent, report_agent 컬렉션은 현재 에이전트에서 직접 사용하지 않음)
    """
    index_root = config.resolved_index_dir(PROJECT_ROOT)
    required   = ["market_agent", "skon_agent", "catl_agent"]

    for name in required:
        if not (index_root / name).exists():
            raise SystemExit(
                f"인덱스 미발견: {index_root / name}\n"
                f"먼저 실행하세요: python app.py build-indices"
            )

    print("FAISS 인덱스 로드 중...")
    market_vs = load_index(config, PROJECT_ROOT, "market_agent")
    print("  ✓ market_agent")
    skon_vs   = load_index(config, PROJECT_ROOT, "skon_agent")
    print("  ✓ skon_agent")
    catl_vs   = load_index(config, PROJECT_ROOT, "catl_agent")
    print("  ✓ catl_agent\n")

    initialize_rag_pipelines_with_stores(
        skon_vectorstore=skon_vs,
        catl_vectorstore=catl_vs,
        market_vectorstore=market_vs,
    )


# ─────────────────────────────────────────────
# 대화형 HITL 루프
# ─────────────────────────────────────────────

def _format_interrupt(payload: dict) -> str:
    """interrupt payload를 터미널 출력용 텍스트로 변환."""
    phase = payload.get("phase") or payload.get("stage", "unknown")
    lines = [f"\n{'='*60}", f"[Human Review] {phase}", "="*60]

    if phase == "review_1":
        ctx = payload.get("market_result", {})
        lines.append(f"재조사: {ctx.get('retry_count', 0)} / {ctx.get('max_retries', 2)}")
        lines.append(f"상태: {ctx.get('status', '?')}  |  tool: {ctx.get('tool_usage', {})}")
        thin = ctx.get("thin_sections", [])
        if thin:
            lines.append(f"미달 섹션(600자 이하): {thin}")
        lines.append(f"허용 결정: {payload.get('allowed_decisions', [])}")

    elif phase == "review_2":
        for company in ("skon", "catl"):
            info = payload.get(company, {})
            lines.append(
                f"{company.upper()}: status={info.get('status')} "
                f"retry={info.get('retry_count')}/{info.get('max_retries')}"
            )
        lines.append(f"허용 결정: {payload.get('allowed_decisions', [])}")

    elif phase == "human_review_3":
        action = (payload.get("action_requests") or [{}])[0]
        report = (action.get("args") or {}).get("report_draft") or {}
        lines.append(f"리뷰 라운드: {payload.get('review_round')}/{payload.get('max_review_rounds')}")
        lines.append(f"\n제목: {report.get('title', '(없음)')}")
        summary = report.get("summary", "")
        lines.append(f"\nSummary:\n{summary[:600]}{'...' if len(summary) > 600 else ''}")

    lines.append("="*60)
    return "\n".join(lines)


def _prompt_resume(payload: dict) -> dict:
    """터미널에서 사용자 입력을 받아 resume value를 반환."""
    print(_format_interrupt(payload))
    phase = payload.get("phase") or payload.get("stage", "")

    if phase == "review_1":
        allowed = payload.get("allowed_decisions", ["approve"])
        while True:
            decision = input(f"결정 ({'/'.join(allowed)}): ").strip().lower()
            if decision in allowed:
                break
        feedback = input("피드백 (Enter 생략): ").strip() if decision == "redo" else ""
        return {"decision": decision, "feedback": feedback}

    elif phase == "review_2":
        allowed = payload.get("allowed_decisions", ["approve"])
        while True:
            decision = input(f"결정 ({'/'.join(allowed)}): ").strip().lower()
            if decision in allowed:
                break
        feedback = input("재조사 지시 (Enter 생략): ").strip() if decision != "approve" else ""
        return {"decision": decision, "feedback": feedback}

    elif phase == "human_review_3":
        while True:
            decision = input("결정 (approve/reject): ").strip().lower()
            if decision in {"approve", "reject"}:
                break
        feedback = input("수정 지시 (Enter 생략): ").strip() if decision == "reject" else ""
        return {"decision": decision, "feedback": feedback}

    input(f"[{phase}] Enter를 눌러 계속...")
    return {"decision": "approve", "feedback": ""}


async def _run_interactive(app, initial: dict, config: dict) -> dict:
    """
    파이프라인을 스트리밍으로 실행하며 interrupt 발생 시 사용자 입력을 받는다.
    파이프라인 완료 후 최종 state를 반환한다.
    """
    current_input = initial

    while True:
        interrupted = False

        async for chunk in app.astream(current_input, config, stream_mode="updates"):
            for node_name, update in chunk.items():
                if node_name == "__interrupt__":
                    interrupted   = True
                    interrupt_obj = update[0]          # Interrupt 객체
                    payload       = interrupt_obj.value
                    resume_value  = _prompt_resume(payload)
                    current_input = Command(resume=resume_value)
                    break
                else:
                    _print_node_done(node_name, update)

            if interrupted:
                break

        if not interrupted:
            break

    return app.get_state(config).values


def _print_node_done(node_name: str, update: dict) -> None:
    """노드 완료 시 간략한 상태 출력."""
    if node_name.startswith("__"):
        return
    status_hint = ""
    if "market_context" in update:
        mc = update["market_context"] or {}
        filled = sum(1 for v in mc.values() if v and isinstance(v, dict))
        status_hint = f" (시장 데이터 {filled}개 축)"
    elif "skon_result" in update or "catl_result" in update:
        for company in ("skon", "catl"):
            result = update.get(f"{company}_result") or {}
            if result:
                status_hint += f" [{company.upper()} {result.get('status', '?')}]"
    elif "comparative_swot" in update:
        status_hint = " (SWOT 완료)"
    elif "report_draft" in update:
        status_hint = f" 제목: {(update['report_draft'] or {}).get('title', '')[:40]}"
    elif "final_report_path" in update:
        status_hint = f" → {update['final_report_path']}"

    print(f"  ✓ {node_name}{status_hint}")


# ─────────────────────────────────────────────
# CLI 명령어: run
# ─────────────────────────────────────────────

def _run_command(args) -> None:
    config = RAGConfig(
        index_dir=Path(args.index_dir),
        embedding_model=args.embedding_model,
    )

    # 전역 FAISS 로드 및 RAG 파이프라인 초기화
    _load_and_init_vectorstores(config)

    # 파이프라인 빌드
    pipeline   = build_pipeline()
    thread_id  = str(uuid.uuid4())
    run_config = {"configurable": {"thread_id": thread_id}}

    user_request = args.request or input("분석 요청을 입력하세요: ").strip()
    print(f"\n[파이프라인 시작] thread_id={thread_id}")
    print(f"요청: {user_request}\n")

    initial = {
        "user_request":       user_request,
        "review_round":       0,
        "max_review_rounds":  MAX_REPORT_RETRIES,
        "review_history":     [],
        "final_revision_mode": False,
        "error_log":          [],
    }

    final = asyncio.run(_run_interactive(pipeline, initial, run_config))

    status = final.get("final_status", "unknown")
    path   = final.get("final_report_path", "")
    print(f"\n[완료] 상태={status}")
    if path:
        print(f"보고서: {path}")


# ─────────────────────────────────────────────
# CLI 명령어: build-indices
# ─────────────────────────────────────────────

def _build_indices_command(args) -> None:
    config = RAGConfig(
        docs_dir=Path(args.docs_dir),
        index_dir=Path(args.index_dir),
        embedding_model=args.embedding_model,
        table_backend=args.table_backend,
    )

    docs_dir = config.resolved_docs_dir(PROJECT_ROOT)
    paths    = sorted(docs_dir.glob("*.pdf"))
    if not paths:
        raise SystemExit(f"PDF 파일이 없습니다: {docs_dir}")

    print(f"PDF {len(paths)}개 로드 중...")
    documents = build_documents_from_paths(paths, config)

    collection_names = args.collection or None
    summary = build_and_save_indices(
        documents=documents,
        config=config,
        project_root=PROJECT_ROOT,
        collection_names=collection_names,
    )

    print(f"저장 위치: {config.resolved_index_dir(PROJECT_ROOT)}")
    for name, vals in summary.items():
        print(f"  {name}: documents={vals['documents']} vectors={vals['vectors']}")


# ─────────────────────────────────────────────
# 공통 인수 / Entry Point
# ─────────────────────────────────────────────

def _add_index_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-dir",       default="data/vectorstores")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")


def main() -> None:
    parser = argparse.ArgumentParser(description="SKATL 배터리 시장 전략 분석 파이프라인")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build-indices
    build_p = subparsers.add_parser("build-indices", help="FAISS 인덱스 빌드")
    build_p.add_argument("--docs-dir",      default="data/docs")
    build_p.add_argument("--table-backend", choices=["auto", "heuristic", "pdfplumber", "camelot"], default="auto")
    build_p.add_argument("--collection",    action="append", choices=get_collection_names())
    _add_index_args(build_p)

    # run
    run_p = subparsers.add_parser("run", help="전체 파이프라인 실행")
    run_p.add_argument("--request", default="", help="분석 요청 (생략 시 대화형 입력)")
    _add_index_args(run_p)

    args = parser.parse_args()

    if args.command == "build-indices":
        _build_indices_command(args)
    elif args.command == "run":
        _run_command(args)


if __name__ == "__main__":
    main()
