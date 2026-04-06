#!/usr/bin/env python3
"""
Index ETL Codebase into Vector Store
--------------------------------------
Walks the etl/ directory, applies code-aware chunking (Python via ast,
SAS via regex), embeds every chunk with sentence-transformers, and
persists them in a dedicated ChromaDB collection called `etl_codebase`.

The indexed collection can then be queried via the standard RAG pipeline
or the query_etl_codebase.py script.

Usage
-----
    # Index the default etl/ directory
    python scripts/index_etl_codebase.py

    # Index a different directory
    python scripts/index_etl_codebase.py --dir src/

    # Force re-index (drops the existing collection first)
    python scripts/index_etl_codebase.py --rebuild

    # Dry-run: show what would be chunked without storing anything
    python scripts/index_etl_codebase.py --dry-run

    # Query the index after building it
    python scripts/index_etl_codebase.py --query "How does the SAS macro validate nulls?"
"""

from __future__ import annotations

import argparse
import os
import sys

# Make project root importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.codebase_loader import CodebaseLoader
from src.processing.code_chunker import chunk_file, get_chunker
from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ETL_DIR       = "etl"
DEFAULT_COLLECTION    = "etl_codebase"
DEFAULT_VECTOR_DIR    = "data/vector_store"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Rebuild helper
# ---------------------------------------------------------------------------

def drop_collection(collection_name: str, vector_store_dir: str) -> None:
    """Delete a ChromaDB collection so it can be rebuilt from scratch."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=vector_store_dir)
        client.delete_collection(collection_name)
        print(f"  Dropped existing collection '{collection_name}'.")
    except Exception as exc:
        print(f"  Note: could not drop collection (may not exist yet): {exc}")


# ---------------------------------------------------------------------------
# Dry-run: show chunk preview without embedding/storing
# ---------------------------------------------------------------------------

def dry_run(root_dir: str) -> None:
    from pathlib import Path

    loader = CodebaseLoader(root_dir=root_dir)
    files  = loader.discover_files()

    if not files:
        print("No indexable files found.")
        return

    total_chunks = 0
    print(f"\n{'='*70}")
    print(f"  DRY RUN – {root_dir}")
    print(f"{'='*70}\n")

    for path in files:
        try:
            chunks = chunk_file(str(path))
            rel    = path.relative_to(Path(root_dir).parent)
            print(f"  {rel}")
            for c in chunks:
                print(f"    [{c.chunk_type:<12s}] {c.name:<40s} lines {c.start_line}-{c.end_line}  ({len(c.content)} chars)")
            total_chunks += len(chunks)
        except Exception as exc:
            print(f"  ERROR: {path}: {exc}")

    print(f"\n  Total: {len(files)} files, {total_chunks} chunks")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------

def run_query(question: str, collection: str, vector_dir: str, top_k: int = 5) -> None:
    embedder = Embedder(model_name=DEFAULT_EMBEDDING_MODEL)
    store    = VectorStore(collection_name=collection, persist_dir=vector_dir)

    print(f"\nQuery: {question!r}")
    print(f"{'─'*60}")

    query_embedding = embedder.embed_query(question)
    hits = store.search(query_embedding, top_k=top_k)

    if not hits:
        print("No results found. Have you run the indexer yet?")
        return

    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        dist = hit["distance"]
        print(f"\n  [{i}] {meta.get('language','?').upper()} / {meta.get('chunk_type','?')}"
              f"  –  {meta.get('name','?')}")
        print(f"       File: {meta.get('source','?')}  (lines {meta.get('start_line','?')}-{meta.get('end_line','?')})")
        print(f"       Distance: {dist:.4f}")
        print(f"       ── Content preview ──")
        # Print first 300 chars of the chunk
        preview = hit["content"][:300].replace("\n", "\n       ")
        print(f"       {preview}")
        if len(hit["content"]) > 300:
            print(f"       … ({len(hit['content'])} chars total)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index the ETL codebase (Python + SAS) into a ChromaDB vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        default=DEFAULT_ETL_DIR,
        help=f"Root directory to index (default: {DEFAULT_ETL_DIR})",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--vector-dir",
        default=DEFAULT_VECTOR_DIR,
        help=f"ChromaDB persist directory (default: {DEFAULT_VECTOR_DIR})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate the collection before indexing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show chunking plan without embedding or storing anything",
    )
    parser.add_argument(
        "--query",
        metavar="QUESTION",
        help="Run a similarity search against the collection after indexing",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return for --query (default: 5)",
    )
    args = parser.parse_args()

    # ── Dry run: just show the chunking plan ──────────────────────────────
    if args.dry_run:
        dry_run(args.dir)
        return

    # ── Optional rebuild ──────────────────────────────────────────────────
    if args.rebuild:
        drop_collection(args.collection, args.vector_dir)

    # ── Index ─────────────────────────────────────────────────────────────
    loader = CodebaseLoader(
        root_dir        = args.dir,
        collection_name = args.collection,
        vector_store_dir= args.vector_dir,
        embedding_model = DEFAULT_EMBEDDING_MODEL,
    )
    stats = loader.index()

    # ── Optional query ────────────────────────────────────────────────────
    if args.query:
        run_query(args.query, args.collection, args.vector_dir, args.top_k)


if __name__ == "__main__":
    main()
