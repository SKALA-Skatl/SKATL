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
        return

    if args.command == "query":
        from src.rag.vectorstore import similarity_search

        results = similarity_search(
            collection_name=args.collection,
            question=args.question,
            config=config,
            project_root=PROJECT_ROOT,
            k=args.k,
        )
        for idx, doc in enumerate(results, start=1):
            print("=" * 80)
            print(f"rank={idx}")
            print(f"source={doc.metadata.get('source')} page={doc.metadata.get('page')} chunk_type={doc.metadata.get('chunk_type')}")
            print(doc.page_content)
        return

    if args.command == "agentic-query":
        from src.rag import agentic_similarity_search

        result = agentic_similarity_search(
            collection_name=args.collection,
            question=args.question,
            config=config,
            project_root=PROJECT_ROOT,
            final_k=args.k,
            per_query_k=args.per_query_k,
            max_rounds=args.max_rounds,
        )
        print(f"collection={result.collection_name}")
        print(f"sufficient={result.sufficient}")
        print("missing_sources=" + (", ".join(result.missing_sources) if result.missing_sources else "(none)"))
        print("")
        print("retrieval_trace")
        for retrieval_round in result.rounds:
            print("-" * 80)
            print(f"round={retrieval_round.round_index}")
            print("queries=" + " || ".join(retrieval_round.queries))
            print(f"retrieved_count={retrieval_round.retrieved_count}")
            print("unique_sources=" + ", ".join(retrieval_round.unique_sources))
            print(f"sufficient={retrieval_round.sufficient}")
            print("missing_sources=" + (", ".join(retrieval_round.missing_sources) if retrieval_round.missing_sources else "(none)"))
        print("")
        print("final_hits")
        for idx, hit in enumerate(result.final_hits, start=1):
            print("=" * 80)
            print(f"rank={idx} score={hit.score:.2f}")
            print(
                "source={source} page={page} chunk_type={chunk_type} matched_queries={queries}".format(
                    source=hit.document.metadata.get("source"),
                    page=hit.document.metadata.get("page"),
                    chunk_type=hit.document.metadata.get("chunk_type"),
                    queries=" | ".join(hit.matched_queries),
                )
            )
            print(hit.document.page_content)
        return

    if args.command == "build-state":
        from rag import build_agent_retrieval_state

        state = build_agent_retrieval_state(
            collection_name=args.collection,
            question=args.question,
            config=config,
            project_root=PROJECT_ROOT,
            final_k=args.k,
            per_query_k=args.per_query_k,
            max_rounds=args.max_rounds,
        )
        print(f"collection={state.collection_name}")
        print(f"agent_role={state.agent_role}")
        print(f"sufficient={state.sufficient}")
        print("present_sources=" + (", ".join(state.coverage.present_sources) if state.coverage.present_sources else "(none)"))
        print("missing_sources=" + (", ".join(state.coverage.missing_sources) if state.coverage.missing_sources else "(none)"))
        print("required_topics=" + ", ".join(state.required_topics))
        print("")
        print("topic_coverage")
        for topic, count in state.coverage.topic_coverage.items():
            print(f"- {topic}: {count}")
        print("")
        print("evidence_packets")
        for packet in state.evidence_packets:
            print("=" * 80)
            print(
                "rank={rank} score={score:.2f} source={source} page={page} chunk_type={chunk_type}".format(
                    rank=packet.rank,
                    score=packet.score,
                    source=packet.source,
                    page=packet.page,
                    chunk_type=packet.chunk_type,
                )
            )
            print("topics=" + (", ".join(packet.topic_tags) if packet.topic_tags else "(none)"))
            print(packet.content)

        if args.save_json:
            save_json_path = Path(args.save_json).resolve()
            save_json_path.parent.mkdir(parents=True, exist_ok=True)
            save_json_path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nsaved_json={save_json_path}")

        if args.save_context:
            save_context_path = Path(args.save_context).resolve()
            save_context_path.parent.mkdir(parents=True, exist_ok=True)
            save_context_path.write_text(state.prompt_context, encoding="utf-8")
            print(f"saved_context={save_context_path}")


def inspect_documents(config: RAGConfig, sample_size: int) -> None:
    documents = _load_documents(config)
    print(f"documents={len(documents)}")

    type_counts: dict[str, int] = {}
    for doc in documents:
        chunk_type = doc.metadata.get("chunk_type", "unknown")
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1

    for chunk_type, count in sorted(type_counts.items()):
        print(f"{chunk_type}_chunks={count}")

    text_limit = max(1, sample_size // 2)
    table_limit = max(1, sample_size - text_limit)

    print("\nText samples\n")
    _print_samples([doc for doc in documents if doc.metadata.get("chunk_type") == "text"][:text_limit])

    print("\nTable samples\n")
    _print_samples([doc for doc in documents if doc.metadata.get("chunk_type") == "table"][:table_limit])


def inspect_tables(config: RAGConfig, source: str | None, limit: int, save_path: Path | None) -> None:
    documents = _load_documents(config)
    table_documents = [doc for doc in documents if doc.metadata.get("chunk_type") == "table"]

    if source:
        table_documents = [doc for doc in table_documents if doc.metadata.get("source") == source]

    print(f"table_documents={len(table_documents)}")
    selected = table_documents[:limit]
    rendered = _render_samples(selected)
    print(rendered)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(rendered, encoding="utf-8")
        print(f"\nsaved_to={save_path}")


def list_collections() -> None:
    for collection in COLLECTIONS:
        print("=" * 80)
        print(f"name={collection.name}")
        print(f"description={collection.description}")
        print("sources=" + ", ".join(collection.sources))


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
