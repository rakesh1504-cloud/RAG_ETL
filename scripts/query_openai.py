#!/usr/bin/env python3
"""
query_openai.py – Interactive RAG query CLI powered by OpenAI GPT-4o
---------------------------------------------------------------------

Usage
-----
  # First-time setup: re-index codebase with OpenAI embeddings
  python scripts/query_openai.py --reindex

  # Interactive session (OpenAI embeddings + GPT-4o)
  python scripts/query_openai.py

  # Use existing TF-IDF index for retrieval, GPT-4o for generation (no re-index needed)
  python scripts/query_openai.py --embedding-backend tfidf

  # One-shot query
  python scripts/query_openai.py --query "What macros handle null checks?"

  # Specify organisation name for contextualised insights
  python scripts/query_openai.py --org "Acme Data Engineering"

  # Adjust retrieval depth
  python scripts/query_openai.py --top-k 8 --max-context 10

Environment
-----------
  OPENAI_API_KEY   Required.  Set before running.

Features
--------
  • Streaming GPT-4o responses
  • Multi-turn conversation history preserved across questions
  • Source attribution with file, chunk type, name, line range, similarity score
  • Organisational insights appended to every answer by the system prompt
  • Optional first-time re-indexing with OpenAI text-embedding-3-small
  • Graceful handling of missing API key and empty collections
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

# Ensure project root is on the path when run from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# ANSI helpers (no extra dependencies)
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
GREY   = "\033[90m"
RED    = "\033[31m"
BLUE   = "\033[34m"


def colour(text: str, code: str) -> str:
    """Wrap text in ANSI colour if stdout is a TTY."""
    if sys.stdout.isatty():
        return f"{code}{text}{RESET}"
    return text


def banner() -> None:
    print(colour("━" * 70, CYAN))
    print(colour("  OpenAI RAG Query CLI", BOLD + CYAN))
    print(colour("  ETL Codebase · GPT-4o · Organisational Insights", GREY))
    print(colour("━" * 70, CYAN))
    print()


def divider(char: str = "─", width: int = 70) -> None:
    print(colour(char * width, GREY))


# ---------------------------------------------------------------------------
# Source display
# ---------------------------------------------------------------------------

def print_sources(sources) -> None:
    if not sources:
        print(colour("  (no sources retrieved)", GREY))
        return

    print(colour("\nSources:", BOLD + YELLOW))
    for src in sources:
        similarity = 1 - src.distance
        bar_len    = int(similarity * 20)
        bar        = "█" * bar_len + "░" * (20 - bar_len)
        print(
            f"  {colour(f'[{src.rank}]', YELLOW)} "
            f"{colour(src.language.upper(), GREEN)} / {src.chunk_type}"
            f"  {colour('–', GREY)}  {colour(src.name, BOLD)}\n"
            f"       {colour(src.file, GREY)}"
            f"  lines {src.start_line}–{src.end_line}\n"
            f"       Similarity: {colour(bar, GREEN)} {similarity:.3f}"
        )
        if src.content_preview:
            preview = src.content_preview[:120].replace("\n", " ")
            print(f"       {colour(preview + '…', GREY)}")
        print()


# ---------------------------------------------------------------------------
# Usage / stats display
# ---------------------------------------------------------------------------

def print_stats(result, embedding_backend: str) -> None:
    divider()
    embed_label = (
        colour(result.embedding_model, CYAN)
        if embedding_backend == "openai"
        else colour("TF-IDF (offline)", CYAN)
    )
    print(
        colour("  Model: ", GREY) + colour(result.model, CYAN) + "   "
        f"{colour('Embed:', GREY)} {embed_label}   "
        f"{colour('Tokens in/out:', GREY)} "
        f"{colour(str(result.input_tokens), CYAN)}"
        f"{colour('/', GREY)}"
        f"{colour(str(result.output_tokens), CYAN)}   "
        f"{colour('Collections:', GREY)} "
        + colour(', '.join(result.collections_searched), CYAN)
    )


# ---------------------------------------------------------------------------
# Single query helper
# ---------------------------------------------------------------------------

def run_one_query(
    engine,
    question: str,
    history: list[dict],
    embedding_backend: str,
) -> list[dict]:
    """Stream a single query, print results, return updated history."""
    print()
    print(colour("Answer:", BOLD + GREEN))
    print()

    try:
        gen = engine.query_stream(question, history=history)
        for token in gen:
            print(token, end="", flush=True)
        print("\n")

        result = engine.get_last_result()
        if result:
            print_sources(result.sources)
            print_stats(result, embedding_backend)
            return engine.get_last_history()

    except Exception as exc:
        print(colour(f"\n  Error: {exc}", RED))
        _print_troubleshooting(exc)

    return history


def _print_troubleshooting(exc: Exception) -> None:
    msg = str(exc).lower()
    if "api_key" in msg or "authentication" in msg or "401" in msg:
        print(colour(
            "\n  Fix: set your OpenAI API key:\n"
            "    export OPENAI_API_KEY=sk-...\n", YELLOW
        ))
    elif "quota" in msg or "429" in msg:
        print(colour("\n  Fix: check your OpenAI account quota/billing.\n", YELLOW))
    elif "empty" in msg or "index" in msg:
        print(colour(
            "\n  Fix: the OpenAI collection is empty.\n"
            "    Run: python scripts/query_openai.py --reindex\n", YELLOW
        ))


# ---------------------------------------------------------------------------
# Re-index helper
# ---------------------------------------------------------------------------

def do_reindex(engine) -> None:
    print(colour("\nRe-indexing codebase with OpenAI embeddings …\n", YELLOW))
    try:
        summary = engine.reindex()
        print(colour(
            f"\n  Indexed {summary['chunks_indexed']} chunks "
            f"into '{summary['collection']}'.",
            GREEN,
        ))
    except Exception as exc:
        print(colour(f"\n  Re-index failed: {exc}", RED))
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive RAG CLI powered by OpenAI GPT-4o.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # One-time re-index (first run with OpenAI embeddings):
          python scripts/query_openai.py --reindex

          # Interactive session with OpenAI embeddings:
          python scripts/query_openai.py

          # Interactive session reusing existing TF-IDF index (no API embed cost):
          python scripts/query_openai.py --embedding-backend tfidf

          # Targeted org query:
          python scripts/query_openai.py --org "Acme Corp" \\
              --query "What data quality checks exist in our SAS pipelines?"
        """),
    )
    p.add_argument(
        "--reindex", action="store_true",
        help="Re-embed the ETL codebase with OpenAI embeddings before querying.",
    )
    p.add_argument(
        "--embedding-backend", choices=["openai", "tfidf"], default="openai",
        help="Embedding backend for retrieval. 'tfidf' reuses the existing "
             "offline index (no OpenAI embedding API call). Default: openai.",
    )
    p.add_argument(
        "--llm-model", default="gpt-4o",
        help="OpenAI chat model for generation. Default: gpt-4o.",
    )
    p.add_argument(
        "--embedding-model", default="text-embedding-3-small",
        help="OpenAI embedding model. Default: text-embedding-3-small.",
    )
    p.add_argument(
        "--collection", default=None,
        help="ChromaDB collection to search. Defaults to 'etl_codebase_openai' "
             "for openai backend, 'etl_codebase' for tfidf backend.",
    )
    p.add_argument(
        "--vector-dir", default="data/vector_store",
        help="ChromaDB persist directory. Default: data/vector_store.",
    )
    p.add_argument(
        "--top-k", type=int, default=5,
        help="Chunks to retrieve per collection. Default: 5.",
    )
    p.add_argument(
        "--max-context", type=int, default=8,
        help="Max chunks passed to GPT-4o after re-ranking. Default: 8.",
    )
    p.add_argument(
        "--org", default="our organisation",
        help="Organisation name injected into the system prompt.",
    )
    p.add_argument(
        "--query", "-q", default=None,
        help="Run a single query and exit (non-interactive mode).",
    )
    p.add_argument(
        "--no-stream", action="store_true",
        help="Disable streaming; print the full answer at once.",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    # Check API key (warn early, but allow tfidf-only flow to proceed)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(colour(
            "\n  Warning: OPENAI_API_KEY is not set.\n"
            "  Set it with:  export OPENAI_API_KEY=sk-...\n",
            YELLOW,
        ))
        if args.embedding_backend == "openai" or args.reindex:
            print(colour("  Aborting: OpenAI API key required for this mode.", RED))
            sys.exit(1)

    # Build engine
    from src.llm.openai_retrieval_engine import OpenAIRetrievalEngine

    collections = [args.collection] if args.collection else None
    engine = OpenAIRetrievalEngine(
        llm_model           = args.llm_model,
        embedding_model     = args.embedding_model,
        embedding_backend   = args.embedding_backend,
        collections         = collections,
        vector_store_dir    = args.vector_dir,
        top_k_per_collection = args.top_k,
        max_context_chunks  = args.max_context,
        api_key             = api_key,
        organisation        = args.org,
    )

    # Optional re-index
    if args.reindex:
        do_reindex(engine)

    # One-shot mode
    if args.query:
        banner()
        print(colour(f"Q: {args.query}", BOLD))
        if args.no_stream:
            result, _ = engine.query(args.query)
            print()
            print(colour("Answer:", BOLD + GREEN))
            print(result.answer)
            print()
            print_sources(result.sources)
            print_stats(result, args.embedding_backend)
        else:
            run_one_query(engine, args.query, [], args.embedding_backend)
        print()
        return

    # Interactive session
    banner()
    print(colour(
        f"  LLM: {args.llm_model}   "
        f"Embed: {args.embedding_backend} ({args.embedding_model})\n"
        f"  Org: {args.org}\n"
        f"  Type 'quit' or Ctrl-C to exit.  'reset' to clear history.\n",
        GREY,
    ))
    divider()

    history: list[dict] = []

    while True:
        try:
            question = input(colour("\nYou: ", BOLD + CYAN)).strip()
        except (KeyboardInterrupt, EOFError):
            print(colour("\n\n  Goodbye.\n", GREY))
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print(colour("\n  Goodbye.\n", GREY))
            break

        if question.lower() == "reset":
            history = []
            print(colour("  Conversation history cleared.\n", YELLOW))
            continue

        if question.lower() in {"help", "?"}:
            print(colour(
                "\n  Commands:\n"
                "    reset   – clear conversation history\n"
                "    quit    – exit\n"
                "    help    – show this help\n",
                GREY,
            ))
            continue

        history = run_one_query(engine, question, history, args.embedding_backend)
        print()
        divider()


if __name__ == "__main__":
    main()
