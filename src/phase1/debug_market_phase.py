"""
Debug runner for market phase and HITL.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from phase1.market_phase import debug_interrupt_once, debug_resume_once
from tools.rag_tool import initialize_rag_pipelines


SECTION_LABELS = {
    "ev_growth_slowdown": "EV 성장 둔화",
    "market_share_ranking": "글로벌 점유율/순위",
    "lfp_ncm_trend": "LFP/NCM 트렌드",
    "ess_hev_growth": "ESS/HEV 성장",
    "regulatory_status": "규제 현황",
    "cost_competitiveness": "원가 경쟁력",
}


def _prompt_resume_decision() -> tuple[str | None, str]:
    """Ask the user for a HITL decision in the terminal."""

    print("\nHITL #1 decision")
    print("  approve : 현재 market_context를 승인합니다.")
    print("  redo    : 피드백을 주고 시장 조사를 다시 수행합니다.")
    print("  skip    : 여기서 종료합니다.")

    decision = input("decision [approve/redo/skip]: ").strip().lower()
    if decision in {"", "skip"}:
        return None, ""
    if decision not in {"approve", "redo"}:
        print("알 수 없는 입력이라 종료합니다.")
        return None, ""
    feedback = ""
    if decision == "redo":
        feedback = input("feedback: ").strip()
    return decision, feedback


def _print_section(title: str) -> None:
    """Print a simple section divider."""

    print(f"\n=== {title} ===")


def _print_compact_json(value: object) -> None:
    """Print a compact JSON block for nested values."""

    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _print_market_context(market_context: dict) -> None:
    """Render the market context in a human-readable format."""

    for key, label in SECTION_LABELS.items():
        section = market_context.get(key)
        if not section:
            continue

        _print_section(label)

        source = section.get("source")
        if source:
            print(f"source: {source}")

        key_narrative = section.get("key_narrative")
        if key_narrative:
            print(f"key_narrative: {key_narrative}")

        detailed_analysis = section.get("detailed_analysis")
        if detailed_analysis:
            print(f"detailed_analysis: {detailed_analysis}")

        source_ids = section.get("source_ids", [])
        if source_ids:
            print("source_ids:")
            for source_id in source_ids:
                print(f"- {source_id}")

        extra_fields = {
            field: value
            for field, value in section.items()
            if field not in {"source", "key_narrative", "detailed_analysis", "source_ids"}
        }
        if extra_fields:
            print("details:")
            _print_compact_json(extra_fields)


def _print_sources(sources: list[dict]) -> None:
    """Render source records for human review."""

    if not sources:
        return

    _print_section("사용한 Source Records")
    for index, source in enumerate(sources, start=1):
        print(f"[{index}] {source.get('source_id', '')}")
        print(f"  title: {source.get('title', '')}")
        print(f"  url: {source.get('url', '')}")
        print(f"  source_type: {source.get('source_type', '')}")
        print(f"  retrieved_at: {source.get('retrieved_at', '')}")


def _print_references(references: list[dict]) -> None:
    """Render final report-ready references."""

    if not references:
        return

    _print_section("REFERENCE")
    for item in references:
        print(f"- {item.get('formatted_reference', '')}")


def _print_review_payload(payload: dict) -> None:
    """Render the HITL payload for terminal review."""

    market_result = payload.get("market_result", {})
    market_context = market_result.get("market_context", {})

    _print_section("HITL Review Summary")
    print(f"phase: {payload.get('phase', '')}")
    print(f"status: {market_result.get('status', '')}")
    print(f"retry_count: {market_result.get('retry_count', '')}/{market_result.get('max_retries', '')}")
    print(f"allowed_decisions: {', '.join(payload.get('allowed_decisions', []))}")
    print(f"retry_limit_reached: {payload.get('retry_limit_reached', False)}")

    _print_market_context(market_context)
    _print_sources(market_context.get("source_records", []))
    _print_references(market_context.get("references", []))


def _print_resumed_state(state: dict) -> None:
    """Render the resumed state after approve/redo."""

    _print_section("Resumed State Summary")
    print(f"review_1_decision: {state.get('review_1_decision', '')}")
    print(f"retry_count: {state.get('retry_count', '')}")

    market_context = state.get("market_context", {})
    if market_context:
        _print_market_context(market_context)
        _print_sources(market_context.get("source_records", []))
        _print_references(market_context.get("references", []))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Phase 1 market workflow")
    parser.add_argument("--question", required=True)
    parser.add_argument("--resume-decision", choices=["approve", "redo"])
    parser.add_argument("--feedback", default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--show-json", action="store_true")
    args = parser.parse_args()

    initialize_rag_pipelines()

    interrupt_result = await debug_interrupt_once(args.question)
    if interrupt_result.get("interrupted"):
        _print_review_payload(interrupt_result.get("payload", {}))
    else:
        print("interrupt_result")
        print(json.dumps(interrupt_result, ensure_ascii=False, indent=2, default=str))

    if args.show_json:
        _print_section("interrupt_result JSON")
        print(json.dumps(interrupt_result, ensure_ascii=False, indent=2, default=str))

    decision = args.resume_decision
    feedback = args.feedback

    if args.interactive and not decision:
        decision, feedback = _prompt_resume_decision()

    if decision:
        resumed = await debug_resume_once(args.question, decision, feedback)
        _print_resumed_state(resumed)
        if args.show_json:
            _print_section("resumed_state JSON")
            print(json.dumps(resumed, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
