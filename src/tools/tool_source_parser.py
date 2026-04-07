"""Extract structured source records from tool messages."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from schemas.agent_io import SourceType


_WEB_RESULT_RE = re.compile(
    r"^\[(?P<idx>\d+)\]\s*(?P<title>.+?)(?:\s+\(신뢰도:.*?(?:\|\s*작성일:\s*(?P<date>\d{4}-\d{2}-\d{2}))?\))?$"
)


def extract_source_records_from_messages(messages: list[Any] | tuple[Any, ...]) -> list[dict]:
    """Parse tool outputs and recover structured sources when the LLM omits them."""

    records: list[dict] = []
    for message in messages or []:
        content = _message_text(getattr(message, "content", ""))
        if not content:
            continue
        name = str(getattr(message, "name", "") or "").strip().lower()

        if name == "web_search" or content.startswith("[Web 검색 결과]"):
            records.extend(_parse_web_results(content))
        elif "source_id:" in content and "출처:" in content:
            records.extend(_parse_rag_results(content))

    return _deduplicate(records)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content or "")


def _parse_web_results(content: str) -> list[dict]:
    records: list[dict] = []
    current: dict | None = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _WEB_RESULT_RE.match(line)
        if match:
            current = {
                "source_id": f"web_{int(match.group('idx')):03d}",
                "title": match.group("title").strip(),
                "published_date": match.group("date"),
                "source_type": SourceType.WEB,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "credibility_score": 0,
                "credibility_flags": {},
            }
            continue
        if current and line.startswith("출처:"):
            current["url"] = line.removeprefix("출처:").strip()
            records.append(current)
            current = None

    return records


def _parse_rag_results(content: str) -> list[dict]:
    records: list[dict] = []
    current: dict | None = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and "source_id:" in line:
            if current and current.get("source_id"):
                records.append(current)
            source_id = line.split("source_id:", 1)[1].strip()
            current = {
                "source_id": source_id,
                "url": "",
                "title": "",
                "source_type": SourceType.RAG_FAISS,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "credibility_score": 0,
                "credibility_flags": {},
            }
            continue
        if current is None:
            continue
        if line.startswith("title:"):
            current["title"] = line.removeprefix("title:").strip()
        elif line.startswith("출처:"):
            current["url"] = line.removeprefix("출처:").strip()
        elif line.startswith("source_type:"):
            value = line.removeprefix("source_type:").strip()
            if value in {SourceType.RAG_FAISS.value, SourceType.RAG_REWRITTEN.value, SourceType.WEB.value}:
                current["source_type"] = value

    if current and current.get("source_id"):
        records.append(current)

    return records


def _deduplicate(records: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for record in records:
        key = (str(record.get("source_id", "")), str(record.get("url", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped
