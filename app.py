from __future__ import annotations

import argparse
from pathlib import Path

from rag import RAGConfig, build_documents_from_paths, get_collection_names


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="SKATL RAG utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-indices", help="Build FAISS indices for agent collections")
    _add_common_config_args(build_parser)
    build_parser.add_argument("--collection", action="append", choices=get_collection_names())

    args = parser.parse_args()
    config = RAGConfig(
        docs_dir=Path(args.docs_dir),
        index_dir=Path(args.index_dir),
        embedding_model=args.embedding_model,
        table_backend=args.table_backend,
    )

    if args.command == "build-indices":
        from rag.vectorstore import build_and_save_indices

        documents = _load_documents(config)
        summary = build_and_save_indices(
            documents=documents,
            config=config,
            project_root=PROJECT_ROOT,
            collection_names=args.collection,
        )
        print(f"stored_at={config.resolved_index_dir(PROJECT_ROOT)}")
        for collection_name, values in summary.items():
            print(f"{collection_name}: documents={values['documents']} vectors={values['vectors']}")


def _load_documents(config: RAGConfig):
    docs_dir = config.resolved_docs_dir(PROJECT_ROOT)
    paths = sorted(docs_dir.glob("*.pdf"))
    if not paths:
        raise SystemExit(f"No PDF files found in {docs_dir}")
    return build_documents_from_paths(paths, config)


def _add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--docs-dir", default="data/docs")
    parser.add_argument("--index-dir", default="data/vectorstores")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--table-backend", choices=["auto", "heuristic", "pdfplumber", "camelot"], default="auto")


if __name__ == "__main__":
    main()
