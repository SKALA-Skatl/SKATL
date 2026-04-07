from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


@dataclass(slots=True)
class RAGConfig:
    docs_dir: Path = Path("data/docs")
    index_dir: Path = Path("data/vectorstores")
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    model_cache_dir: Path | None = None
    table_backend: str = "auto"
    chunk_size: int = 900
    chunk_overlap: int = 150
    table_row_group_size: int = 8
    search_k: int = 5
    device: str | None = None

    def resolved_docs_dir(self, root: Path) -> Path:
        return (root / self.docs_dir).resolve() if not self.docs_dir.is_absolute() else self.docs_dir

    def resolved_index_dir(self, root: Path) -> Path:
        return (root / self.index_dir).resolve() if not self.index_dir.is_absolute() else self.index_dir

    def resolved_collection_dir(self, root: Path, collection_name: str) -> Path:
        return self.resolved_index_dir(root) / collection_name
