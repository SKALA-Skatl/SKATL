from __future__ import annotations

"""PDF chunking and document ingestion."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from pypdf import PdfReader

from .config import RAGConfig
from .table_backends import camelot_available, extract_tables_with_camelot, extract_tables_with_pdfplumber


TABLE_LABEL = "table"
TEXT_LABEL = "text"

logging.getLogger("pypdf").setLevel(logging.ERROR)

DOCUMENT_PROFILES = {
    "market_report.pdf": {
        "language": "en",
        "doc_role": "market",
        "publisher": "IEA",
        "entity": "global_ev_market",
        "table_mode": "off",
        "preferred_table_backend": "heuristic",
    },
    "analyst_report.pdf": {
        "language": "ko",
        "doc_role": "market",
        "publisher": "Eugene Investment",
        "entity": "k_battery_market",
        "table_mode": "off",
        "preferred_table_backend": "heuristic",
    },
    "catl.pdf": {
        "language": "en",
        "doc_role": "company",
        "publisher": "CATL",
        "entity": "CATL",
        "table_mode": "default",
        "preferred_table_backend": "camelot",
    },
    "skon.pdf": {
        "language": "ko",
        "doc_role": "company",
        "publisher": "SK Innovation",
        "entity": "SK On",
        "table_mode": "default",
        "preferred_table_backend": "pdfplumber",
    },
}


@dataclass(slots=True)
class PageBlock:
    """Intermediate block extracted from one PDF page."""

    block_index: int
    block_type: str
    page: int
    text: str


def build_documents_from_paths(paths: Iterable[Path], config: RAGConfig) -> list[Document]:
    """Build documents from multiple PDF paths."""

    documents: list[Document] = []
    for path in paths:
        documents.extend(build_documents_from_path(path, config))
    return documents


def build_documents_from_path(path: Path, config: RAGConfig) -> list[Document]:
    """Parse one PDF into text and table documents."""

    reader = PdfReader(str(path))
    documents: list[Document] = []
    document_id = _slugify(path.stem)
    source_profile = DOCUMENT_PROFILES.get(path.name, {})
    page_contexts: dict[int, list[str]] = {}

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = _extract_page_text(page)
        if not page_text.strip():
            continue

        page_profile = source_profile
        if config.table_backend != "heuristic":
            page_profile = {**source_profile, "table_mode": "off"}

        blocks = _build_page_blocks(page_text, page_number, page_profile)
        page_contexts[page_number] = _build_page_context(blocks)
        for block in blocks:
            if block.block_type == TABLE_LABEL:
                # heuristic table block은 행 단위 청크로 바꿉니다.
                documents.extend(
                    _table_block_to_documents(
                        path=path,
                        document_id=document_id,
                        block=block,
                        row_group_size=config.table_row_group_size,
                        source_profile=source_profile,
                    )
                )
                continue

            documents.extend(
                _text_block_to_documents(
                    path=path,
                    document_id=document_id,
                    block=block,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    source_profile=source_profile,
                )
            )

    if source_profile.get("table_mode") != "off":
        backend = _resolve_table_backend(config.table_backend, source_profile)
        # 문서별로 더 잘 맞는 표 추출기를 따로 적용합니다.
        if backend == "camelot":
            documents.extend(extract_tables_with_camelot(path, source_profile, document_id, page_contexts))
        elif backend == "pdfplumber":
            documents.extend(extract_tables_with_pdfplumber(path, source_profile, document_id, page_contexts))

    return documents


def _resolve_table_backend(configured_backend: str, source_profile: dict[str, str]) -> str:
    """Choose the table backend for a source document."""

    if configured_backend == "auto":
        preferred = source_profile.get("preferred_table_backend")
        if preferred in {"camelot", "pdfplumber"}:
            if preferred == "camelot" and camelot_available():
                return preferred
            if preferred == "pdfplumber":
                return preferred
        return "camelot" if camelot_available() else "pdfplumber"
    return configured_backend


def _extract_page_text(page) -> str:
    """Extract text from a PDF page with a small fallback."""

    for extraction_mode in ("layout", None):
        try:
            if extraction_mode is None:
                text = page.extract_text() or ""
            else:
                text = page.extract_text(extraction_mode=extraction_mode) or ""
            if text.strip():
                return text
        except TypeError:
            continue
    return ""


def _build_page_context(blocks: list[PageBlock]) -> list[str]:
    """Collect short context lines from nearby text blocks."""

    # 표 청크에 제목 문맥을 붙여 retrieval 품질을 높입니다.
    context_lines: list[str] = []
    for block in blocks:
        if block.block_type != TEXT_LABEL:
            continue
        for line in block.text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            if len(cleaned) > 120:
                continue
            if cleaned in context_lines:
                continue
            context_lines.append(cleaned)
            if len(context_lines) >= 4:
                return context_lines
    return context_lines


def _build_page_blocks(
    page_text: str,
    page_number: int,
    source_profile: dict[str, str],
) -> list[PageBlock]:
    """Split one page into text-like and table-like blocks."""

    normalized = page_text.replace("\x00", " ").replace("\u2003", " ")
    raw_blocks = [block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()]

    blocks: list[PageBlock] = []
    for index, raw_block in enumerate(raw_blocks):
        prepared = _prepare_block(raw_block)
        if not prepared:
            continue

        table_mode = source_profile.get("table_mode", "default")
        block_type = TABLE_LABEL if _looks_like_table_block(prepared, table_mode) else TEXT_LABEL
        cleaned = _clean_table_block(prepared) if block_type == TABLE_LABEL else _clean_text_block(prepared)
        blocks.append(
            PageBlock(
                block_index=index,
                block_type=block_type,
                page=page_number,
                text=cleaned,
            )
        )

    return blocks


def _prepare_block(block: str) -> str:
    lines = [line.rstrip() for line in block.splitlines()]
    compact = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            compact.append(line.strip())
    return "\n".join(compact).strip()


def _clean_text_block(block: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in block.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _clean_table_block(block: str) -> str:
    lines = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        preserved = re.sub(r"\t+", "    ", stripped)
        lines.append(preserved)
    return "\n".join(lines).strip()


def _looks_like_table_block(block: str, table_mode: str = "default") -> bool:
    """Heuristically detect whether a block looks tabular."""

    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    tabular_rows = sum(_looks_like_table_row(line) for line in lines)
    numeric_rows = sum(bool(re.findall(r"\b\d[\d,./%:-]*\b", line)) for line in lines)
    average_length = sum(len(line) for line in lines) / len(lines)
    ratio = tabular_rows / len(lines)
    has_many_short_lines = len(lines) >= 4 and sum(len(line) <= 40 for line in lines) >= 3

    if tabular_rows < 2:
        return False
    if table_mode == "off":
        return False
    if table_mode == "conservative":
        return numeric_rows >= 2 and ratio >= 0.6 and average_length <= 90
    if numeric_rows >= 1 and ratio >= 0.4:
        return True
    if has_many_short_lines and ratio >= 0.7 and average_length <= 70:
        return True
    return False


def _looks_like_table_row(line: str) -> bool:
    if "|" in line or "\t" in line:
        return True

    columns = [part.strip() for part in re.split(r"\s{2,}", line) if part.strip()]
    numeric_tokens = re.findall(r"\b\d[\d,./%:-]*\b", line)
    short_columns = sum(len(column) <= 25 for column in columns)

    if len(columns) >= 2 and numeric_tokens:
        return True
    if len(columns) >= 3 and short_columns >= 3 and len(line) <= 100:
        return True
    if len(numeric_tokens) >= 2 and len(line) <= 120:
        return True
    return False


def _table_block_to_documents(
    *,
    path: Path,
    document_id: str,
    block: PageBlock,
    row_group_size: int,
    source_profile: dict[str, str],
) -> list[Document]:
    rows = [_split_table_row(line) for line in block.text.splitlines() if line.strip()]
    if not rows:
        return []

    header = rows[0]
    data_rows = rows[1:] if len(rows) > 1 else []
    if not data_rows:
        data_rows = [header]
        header = ["value"]

    documents: list[Document] = []
    for group_index in range(0, len(data_rows), row_group_size):
        group = data_rows[group_index : group_index + row_group_size]
        row_start = group_index + 1
        row_end = group_index + len(group)
        serialized = _render_table_chunk(header, group)
        chunk_id = f"{document_id}-p{block.page}-b{block.block_index}-t{group_index // row_group_size}"

        documents.append(
            Document(
                page_content=serialized,
                metadata={
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "source": path.name,
                    "source_path": str(path),
                    "page": block.page,
                    "block_index": block.block_index,
                    "chunk_type": TABLE_LABEL,
                    "row_start": row_start,
                    "row_end": row_end,
                    **source_profile,
                },
            )
        )

    return documents


def _split_table_row(line: str) -> list[str]:
    if "|" in line:
        return [cell.strip() for cell in line.split("|") if cell.strip()]

    columns = [part.strip() for part in re.split(r"\s{2,}", line) if part.strip()]
    if columns:
        return columns

    return [line.strip()]


def _render_table_chunk(header: list[str], rows: list[list[str]]) -> str:
    rendered = ["[TABLE]"]
    rendered.append("columns: " + " | ".join(header))
    rendered.append("rows:")
    for row in rows:
        normalized = row + [""] * max(0, len(header) - len(row))
        rendered.append("- " + " | ".join(normalized[: len(header)]))
    return "\n".join(rendered)


def _text_block_to_documents(
    *,
    path: Path,
    document_id: str,
    block: PageBlock,
    chunk_size: int,
    chunk_overlap: int,
    source_profile: dict[str, str],
) -> list[Document]:
    chunks = _chunk_text(block.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents: list[Document] = []

    for chunk_index, chunk in enumerate(chunks):
        chunk_id = f"{document_id}-p{block.page}-b{block.block_index}-c{chunk_index}"
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "source": path.name,
                    "source_path": str(path),
                    "page": block.page,
                    "block_index": block.block_index,
                    "chunk_type": TEXT_LABEL,
                    **source_profile,
                },
            )
        )

    return documents


def _chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        tentative_end = min(len(cleaned), start + chunk_size)
        end = _find_breakpoint(cleaned, start, tentative_end)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(0, end - chunk_overlap)
        if start >= end:
            start = end
    return chunks


def _find_breakpoint(text: str, start: int, tentative_end: int) -> int:
    if tentative_end >= len(text):
        return len(text)

    window = text[start:tentative_end]
    candidates = [
        window.rfind("\n\n"),
        window.rfind(". "),
        window.rfind("! "),
        window.rfind("? "),
        window.rfind("\n"),
        window.rfind(" "),
    ]
    best = max(candidates)
    if best < int(len(window) * 0.6):
        return tentative_end
    return start + best + 1


def _slugify(value: str) -> str:
    lowered = value.lower().strip()
    slug = re.sub(r"[^0-9a-zA-Z가-힣]+", "-", lowered)
    return slug.strip("-") or "document"
