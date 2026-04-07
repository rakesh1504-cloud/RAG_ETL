#!/usr/bin/env python3
"""
browse_chromadb.py – Inspect and query the ChromaDB vector store
----------------------------------------------------------------

Usage examples:

  # Show all collections and document counts
  python scripts/browse_chromadb.py --stats

  # List all documents (id, language, type, name, source, lines)
  python scripts/browse_chromadb.py --list

  # Filter by language
  python scripts/browse_chromadb.py --list --lang sas
  python scripts/browse_chromadb.py --list --lang python

  # Filter by chunk type
  python scripts/browse_chromadb.py --list --type macro
  python scripts/browse_chromadb.py --list --type function
  python scripts/browse_chromadb.py --list --type yaml_section

  # Show full content of a specific document by id
  python scripts/browse_chromadb.py --show etl/sas/macros/etl_macros.sas_2

  # Run a similarity search (no LLM – raw vector search only)
  python scripts/browse_chromadb.py --search "null check macro"
  python scripts/browse_chromadb.py --search "customer age validation" --top-k 5

  # Export entire collection to a CSV or JSON file
  python scripts/browse_chromadb.py --export chromadb_export.csv
  python scripts/browse_chromadb.py --export chromadb_export.json

  # Use a different collection
  python scripts/browse_chromadb.py --collection etl_codebase_openai --list
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESET = "\033[0m"; BOLD = "\033[1m"; CYAN = "\033[36m"
GREEN = "\033[32m"; YELLOW = "\033[33m"; GREY = "\033[90m"; RED = "\033[31m"

def c(text, code): return f"{code}{text}{RESET}" if sys.stdout.isatty() else text
def divider(): print(c("─" * 80, GREY))


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def get_client(vector_dir: str):
    import chromadb
    return chromadb.PersistentClient(path=vector_dir)


def get_collection(client, name: str):
    try:
        return client.get_collection(name)
    except Exception:
        print(c(f"Collection '{name}' not found.", RED))
        sys.exit(1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_stats(client) -> None:
    """Show all collections and their document counts."""
    collections = client.list_collections()
    if not collections:
        print("No collections found.")
        return

    print(c(f"\n{'Collection':<35} {'Documents':>10}", BOLD))
    divider()
    for col in collections:
        c_obj = client.get_collection(col.name)
        count = c_obj.count()
        meta  = col.metadata or {}
        space = meta.get("hnsw:space", "l2")
        print(f"  {c(col.name, CYAN):<45} {c(str(count), GREEN):>10}  "
              f"{c(f'[{space}]', GREY)}")
    print()


def cmd_list(collection, lang: str | None, chunk_type: str | None) -> None:
    """List all documents with metadata (filtered optionally)."""
    where: dict | None = None
    if lang and chunk_type:
        where = {"$and": [{"language": {"$eq": lang}}, {"chunk_type": {"$eq": chunk_type}}]}
    elif lang:
        where = {"language": {"$eq": lang}}
    elif chunk_type:
        where = {"chunk_type": {"$eq": chunk_type}}

    kwargs = dict(include=["metadatas", "documents"], limit=10_000)
    if where:
        kwargs["where"] = where

    result = collection.get(**kwargs)
    ids   = result["ids"]
    metas = result["metadatas"]
    docs  = result["documents"]

    if not ids:
        print("No documents match the filter.")
        return

    print(c(f"\n  {'#':<5} {'Language':<10} {'Type':<15} {'Name':<30} {'Lines':<12} Source", BOLD))
    divider()
    for i, (doc_id, meta, doc) in enumerate(zip(ids, metas, docs), 1):
        lang_v  = meta.get("language", "?")
        ctype   = meta.get("chunk_type", "?")
        name    = meta.get("name", "?")[:28]
        source  = meta.get("source", "?")
        sl      = meta.get("start_line", "?")
        el      = meta.get("end_line", "?")
        lang_col = CYAN if lang_v == "python" else YELLOW if lang_v == "sas" else GREEN
        print(
            f"  {c(str(i),''):>5} "
            f"{c(lang_v, lang_col):<18} "
            f"{ctype:<15} "
            f"{name:<30} "
            f"{str(sl)+'-'+str(el):<12} "
            f"{c(source, GREY)}"
        )
    print()
    print(c(f"  Total: {len(ids)} document(s)", BOLD))
    print()


def cmd_show(collection, doc_id: str) -> None:
    """Print full content of a specific document."""
    result = collection.get(ids=[doc_id], include=["documents", "metadatas"])
    if not result["ids"]:
        print(c(f"Document '{doc_id}' not found.", RED))
        return

    meta = result["metadatas"][0]
    doc  = result["documents"][0]

    print()
    print(c(f"  Document: {doc_id}", BOLD + CYAN))
    divider()
    print(c(f"  Language   : ", GREY) + meta.get("language", "?"))
    print(c(f"  Chunk type : ", GREY) + meta.get("chunk_type", "?"))
    print(c(f"  Name       : ", GREY) + meta.get("name", "?"))
    print(c(f"  Source     : ", GREY) + meta.get("source", "?"))
    print(c(f"  Lines      : ", GREY) + f"{meta.get('start_line')} – {meta.get('end_line')}")
    divider()
    print(c("  Content:\n", BOLD))
    for line in doc.splitlines():
        print(f"    {line}")
    print()


def cmd_search(collection, query: str, top_k: int, vector_dir: str) -> None:
    """Run a raw vector similarity search (no LLM)."""
    from src.embedding.embedder import Embedder
    embedder = Embedder(tfidf_persist=f"{vector_dir}/tfidf_embedder.pkl")
    query_embedding = embedder.embed_query(query)

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids   = result["ids"][0]
    metas = result["metadatas"][0]
    docs  = result["documents"][0]
    dists = result["distances"][0]

    print(c(f"\n  Search: \"{query}\"  →  top {top_k} results\n", BOLD))
    divider()

    for rank, (doc_id, meta, doc, dist) in enumerate(zip(ids, metas, docs, dists), 1):
        similarity = 1 - dist
        bar_len = int(similarity * 25)
        bar     = "█" * bar_len + "░" * (25 - bar_len)
        lang    = meta.get("language", "?")
        ctype   = meta.get("chunk_type", "?")
        name    = meta.get("name", "?")
        source  = meta.get("source", "?")
        sl      = meta.get("start_line", "?")
        el      = meta.get("end_line", "?")

        print(
            f"  {c(f'[{rank}]', YELLOW)}  "
            f"{c(lang.upper(), CYAN)} / {ctype}  –  {c(name, BOLD)}\n"
            f"       {c(source, GREY)}  lines {sl}–{el}\n"
            f"       Similarity: {c(bar, GREEN)} {similarity:.4f}"
        )
        preview = doc[:150].replace("\n", " ")
        print(f"       {c(preview + '…', GREY)}")
        print()


def cmd_export(collection, path: str) -> None:
    """Export the entire collection to CSV or JSON."""
    result = collection.get(include=["documents", "metadatas"], limit=10_000)
    ids    = result["ids"]
    metas  = result["metadatas"]
    docs   = result["documents"]
    rows   = [
        {"id": i, **m, "content": d}
        for i, m, d in zip(ids, metas, docs)
    ]

    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
    else:
        # Default to CSV
        if not path.endswith(".csv"):
            path += ".csv"
        fields = ["id", "language", "chunk_type", "name", "source",
                  "start_line", "end_line", "content"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    print(c(f"\n  Exported {len(rows)} documents → {path}\n", GREEN))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Browse and inspect the ChromaDB vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--vector-dir", default="data/vector_store",
                   help="ChromaDB persist directory. Default: data/vector_store")
    p.add_argument("--collection", default="etl_codebase",
                   help="Collection name. Default: etl_codebase")
    p.add_argument("--stats",  action="store_true", help="Show all collections and counts")
    p.add_argument("--list",   action="store_true", help="List all documents")
    p.add_argument("--lang",   help="Filter --list by language (python, sas, markdown, yaml)")
    p.add_argument("--type",   help="Filter --list by chunk_type (function, macro, heading, ...)")
    p.add_argument("--show",   metavar="DOC_ID", help="Print full content of one document")
    p.add_argument("--search", metavar="QUERY",  help="Vector similarity search (no LLM)")
    p.add_argument("--top-k",  type=int, default=5, help="Results for --search. Default: 5")
    p.add_argument("--export", metavar="FILE", help="Export collection to .csv or .json")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args  = build_parser().parse_args()
    client = get_client(args.vector_dir)

    if args.stats:
        cmd_stats(client)
        return

    col = get_collection(client, args.collection)

    if args.list:
        cmd_list(col, args.lang, args.type)
    elif args.show:
        cmd_show(col, args.show)
    elif args.search:
        cmd_search(col, args.search, args.top_k, args.vector_dir)
    elif args.export:
        cmd_export(col, args.export)
    else:
        # Default: show stats + brief list
        cmd_stats(client)
        print(c(f"  Collection: {args.collection}\n", BOLD))
        cmd_list(col, None, None)


if __name__ == "__main__":
    main()
