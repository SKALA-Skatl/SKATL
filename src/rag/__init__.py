"""RAG utilities for document ingestion, indexing, and retrieval."""

from .agentic import AgenticRetrievalResult, agentic_similarity_search
from .collections import COLLECTIONS, filter_documents_for_collection, get_collection, get_collection_names
from .config import RAGConfig
from .pdf_ingest import build_documents_from_path, build_documents_from_paths
from .state import AgentRetrievalState, build_agent_retrieval_state

__all__ = [
    "AgentRetrievalState",
    "AgenticRetrievalResult",
    "COLLECTIONS",
    "RAGConfig",
    "agentic_similarity_search",
    "build_agent_retrieval_state",
    "build_documents_from_path",
    "build_documents_from_paths",
    "filter_documents_for_collection",
    "get_collection",
    "get_collection_names",
]
