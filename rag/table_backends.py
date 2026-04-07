from __future__ import annotations

"""Table extraction backends for PDF documents."""

import shutil
import warnings
from pathlib import Path

from langchain_core.documents import Document


def camelot_available() -> bool:
    """Check whether Camelot can be imported."""

    try:
        import camelot  # noqa: F401
    except ImportError:
        return False
    return True


def ghostscript_available() -> bool:
    """Check whether Ghostscript is installed."""

    return shutil.which("gs") is not None


def extract_tables_with_pdfplumber(
    path: Path,
    source_profile: dict[str, str],
    document_id: str,
    page_contexts: dict[int, list[str]] | None = None,
) -> list[Document]:
    """Extract table documents with pdfplumber."""

    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber is not installed. Install `pdfplumber` to use `table_backend=pdfplumber`."
        ) from exc

    documents: list[Document] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table_index, table in enumerate(tables):
                # 첫 행을 header로 보고 행별 문장으로 변환합니다.
                rows = [_normalize_row(row) for row in table if row and any(cell and str(cell).strip() for cell in row)]
                if len(rows) < 2:
                    continue

                header = rows[0]
                body = rows[1:]
                if not any(header):
                    continue

                lines = _render_table_as_sentences(
                    extractor="pdfplumber",
                    header=header,
                    body=body,
                    context_lines=(page_contexts or {}).get(page_number, []),
                )

                chunk_id = f"{document_id}-pdfplumber-p{page_number}-t{table_index}"
                documents.append(
                    Document(
                        page_content="\n".join(lines),
                        metadata={
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "source": path.name,
                            "source_path": str(path),
                            "page": page_number,
                            "chunk_type": "table",
                            "table_extractor": "pdfplumber",
                            **source_profile,
                        },
                    )
                )

    return _deduplicate_documents(documents)


def extract_tables_with_camelot(
    path: Path,
    source_profile: dict[str, str],
    document_id: str,
    page_contexts: dict[int, list[str]] | None = None,
) -> list[Document]:
    """Extract table documents with Camelot."""

    try:
        import camelot
    except ImportError as exc:
        raise RuntimeError(
            "Camelot is not installed. Install `camelot-py` and its system dependencies to use `table_backend=camelot`."
        ) from exc

    documents: list[Document] = []
    flavors = ["stream"]
    if ghostscript_available():
        # Ghostscript가 있으면 선 기반 표도 추가로 시도합니다.
        flavors.append("lattice")

    for flavor in flavors:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tables = camelot.read_pdf(str(path), pages="all", flavor=flavor)
        except Exception:
            continue
        for index, table in enumerate(tables):
            dataframe = table.df.fillna("")
            normalized_rows = [
                _normalize_row(row)
                for row in dataframe.values.tolist()
                if any(str(value).strip() for value in row)
            ]
            if not _looks_like_structured_table(normalized_rows):
                continue

            title, header, body = _normalize_camelot_table(normalized_rows)
            if not header or not body:
                continue
            content_lines = _render_table_as_sentences(
                extractor=f"camelot:{flavor}",
                header=header,
                body=body,
                title=title,
                context_lines=(page_contexts or {}).get(_safe_int(table.page), []),
            )

            page_number = _safe_int(table.page)
            chunk_id = f"{document_id}-camelot-{flavor}-p{page_number}-t{index}"
            documents.append(
                Document(
                    page_content="\n".join(content_lines),
                    metadata={
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "source": path.name,
                        "source_path": str(path),
                        "page": page_number,
                        "chunk_type": "table",
                        "table_extractor": f"camelot:{flavor}",
                        **source_profile,
                    },
                )
            )

    return _deduplicate_documents(documents)


def _deduplicate_documents(documents: list[Document]) -> list[Document]:
    """Drop duplicate table chunks."""

    deduplicated: list[Document] = []
    seen: set[tuple[int | None, str]] = set()
    for document in documents:
        key = (document.metadata.get("page"), document.page_content)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(document)
    return deduplicated


def _normalize_row(row: list[str | None]) -> list[str]:
    normalized = []
    for cell in row:
        value = "" if cell is None else str(cell)
        value = " ".join(value.replace("\n", " ").split())
        normalized.append(value)
    return normalized


def _normalize_camelot_table(rows: list[list[str]]) -> tuple[str, list[str], list[list[str]]]:
    """Normalize Camelot rows into title, header, and body."""

    trimmed_rows = _drop_empty_columns(rows)
    if len(trimmed_rows) < 2:
        return "", [], []

    header_index = _select_header_index(trimmed_rows)
    title_lines = [
        " ".join(cell for cell in row if cell).strip()
        for row in trimmed_rows[:header_index]
        if any(row)
    ]
    title = " / ".join(line for line in title_lines if line)

    header = _fill_blank_headers(trimmed_rows[header_index])
    body = _merge_multiline_rows(trimmed_rows[header_index + 1 :], len(header))
    body = [row for row in body if any(cell for cell in row)]
    return title, header, body


def _row_to_sentence(header: list[str], row: list[str]) -> str:
    pairs = []
    for column, value in zip(header, row):
        if not column and not value:
            continue
        column_name = column or "value"
        if not value:
            continue
        pairs.append(f"{column_name}={value}")
    return "; ".join(pairs)


def _render_table_as_sentences(
    *,
    extractor: str,
    header: list[str],
    body: list[list[str]],
    title: str | None = None,
    context_lines: list[str] | None = None,
) -> list[str]:
    """Render a table into retrieval-friendly text lines."""

    # 표를 그대로 넣기보다 문장형 key=value 라인으로 바꿉니다.
    lines = [f"[TABLE:{extractor}]"]
    if context_lines:
        lines.append("context: " + " / ".join(context_lines))
    if title:
        lines.append(f"title: {title}")
    lines.append("columns: " + " | ".join(header))
    lines.append("rows:")
    for row in body:
        sentence = _row_to_sentence(header, row)
        if sentence:
            lines.append(f"- {sentence}")
    return lines


def _drop_empty_columns(rows: list[list[str]]) -> list[list[str]]:
    width = max(len(row) for row in rows)
    keep_indexes = []
    for idx in range(width):
        if any(idx < len(row) and row[idx] for row in rows):
            keep_indexes.append(idx)

    trimmed_rows: list[list[str]] = []
    for row in rows:
        expanded = row + [""] * (width - len(row))
        trimmed_rows.append([expanded[idx] for idx in keep_indexes])
    return trimmed_rows


def _select_header_index(rows: list[list[str]]) -> int:
    candidate_count = min(6, len(rows))
    best_index = 0
    best_score = float("-inf")

    for idx in range(candidate_count):
        row = rows[idx]
        non_empty = [cell for cell in row if cell]
        if not non_empty:
            continue

        numeric_only = sum(_is_numeric_like(cell) for cell in non_empty)
        short_cells = sum(len(cell) <= 24 for cell in non_empty)
        score = (len(non_empty) * 10) + (short_cells * 3) - (numeric_only * 2) - idx
        if score > best_score:
            best_score = score
            best_index = idx

    return best_index


def _fill_blank_headers(header: list[str]) -> list[str]:
    normalized = []
    for idx, cell in enumerate(header, start=1):
        value = cell.strip()
        normalized.append(value or f"column_{idx}")
    return normalized


def _merge_multiline_rows(rows: list[list[str]], width: int) -> list[list[str]]:
    merged: list[list[str]] = []
    pending_label_parts: list[str] = []

    for row in rows:
        padded = row + [""] * (width - len(row))
        non_empty = [cell for cell in padded if cell]
        digit_cells = sum(_is_numeric_like(cell) for cell in non_empty)

        if non_empty and digit_cells == 0 and len(non_empty) <= 1:
            pending_label_parts.extend(non_empty)
            continue

        if pending_label_parts:
            first_idx = next((idx for idx, cell in enumerate(padded) if cell), 0)
            prefix = " ".join(pending_label_parts).strip()
            padded[first_idx] = f"{prefix} {padded[first_idx]}".strip()
            pending_label_parts = []

        merged.append(padded)

    if pending_label_parts:
        merged.append([" ".join(pending_label_parts).strip()] + [""] * (width - 1))

    return merged


def _is_numeric_like(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    candidate = stripped.replace(",", "").replace("%", "").replace("’", "").replace("'", "")
    candidate = candidate.replace("(", "").replace(")", "").replace("-", "")
    candidate = candidate.replace(".", "", 1)
    return candidate.isdigit()


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _looks_like_structured_table(rows: list[list[str]]) -> bool:
    if len(rows) < 2:
        return False

    non_empty_counts = [sum(bool(cell) for cell in row) for row in rows]
    max_columns = max(non_empty_counts, default=0)
    numeric_rows = 0
    structured_rows = 0

    for row in rows:
        non_empty = [cell for cell in row if cell]
        if len(non_empty) >= 3:
            structured_rows += 1
        if sum(any(char.isdigit() for char in cell) for cell in non_empty) >= 1:
            numeric_rows += 1

    average_cell_length = 0.0
    all_cells = [cell for row in rows for cell in row if cell]
    if all_cells:
        average_cell_length = sum(len(cell) for cell in all_cells) / len(all_cells)

    if max_columns <= 2 and average_cell_length > 60:
        return False
    if structured_rows >= 2 and numeric_rows >= 1:
        return True
    if structured_rows >= 3:
        return True
    return False
