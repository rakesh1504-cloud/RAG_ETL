#!/usr/bin/env python3
"""
query_azure.py – Interactive RAG CLI powered by Azure AI Search + Azure OpenAI GPT-4o
--------------------------------------------------------------------------------------

Setup (one-time)
----------------
  1. Provision Azure resources (see config/azure_config.yaml for guidance)
  2. Set environment variables:
       export AZURE_OPENAI_ENDPOINT="https://my-aoai.openai.azure.com/"
       export AZURE_OPENAI_API_KEY="..."
       export AZURE_OPENAI_EMBED_DEPLOY="text-embedding-3-small"
       export AZURE_OPENAI_CHAT_DEPLOY="gpt-4o"
       export AZURE_SEARCH_ENDPOINT="https://my-search.search.windows.net"
       export AZURE_SEARCH_API_KEY="..."
       # optional for blob backup:
       export AZURE_STORAGE_CONNECTION_STR="DefaultEndpointsProtocol=https;..."

  3. Index the codebase (first time):
       python scripts/query_azure.py --index

Usage
-----
  # Interactive session
  python scripts/query_azure.py

  # Filter to SAS only
  python scripts/query_azure.py --filter-lang sas

  # Filter to specific chunk type
  python scripts/query_azure.py --filter-type macro

  # One-shot query
  python scripts/query_azure.py --query "What null checks exist in our SAS pipeline?"

  # Index specific directories
  python scripts/query_azure.py --index --dirs etl docs data/schemas

  # Full rebuild (drop + recreate index)
  python scripts/query_azure.py --index --rebuild

  # Disable semantic reranking (Basic tier search service)
  python scripts/query_azure.py --no-semantic
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ANSI colours
RESET  = "\033[0m"; BOLD   = "\033[1m"; CYAN   = "\033[36m"
GREEN  = "\033[32m"; YELLOW = "\033[33m"; GREY   = "\033[90m"
RED    = "\033[31m"; BLUE   = "\033[34m"; PURPLE = "\033[35m"

def c(text, code): return f"{code}{text}{RESET}" if sys.stdout.isatty() else text
def divider(w=72): print(c("─" * w, GREY))


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def banner(args) -> None:
    print(c("━" * 72, CYAN))
    print(c("  Azure Knowledge Base – ETL Codebase RAG", BOLD + CYAN))
    print(c(f"  Search: {args.search_endpoint or 'AZURE_SEARCH_ENDPOINT'}  "
            f"│  LLM: {args.chat_deploy}", GREY))
    print(c("━" * 72, CYAN))
    print()


def print_sources(sources, semantic: bool) -> None:
    if not sources:
        print(c("  (no sources retrieved)", GREY)); return

    tag = c("[Azure Semantic Reranking]", PURPLE) if semantic else c("[Hybrid BM25+Vector]", CYAN)
    print(c(f"\nSources  {tag}", BOLD + YELLOW))

    for src in sources:
        bar_len = min(int(src.score / 4.0 * 25), 25)
        bar     = "█" * bar_len + "░" * (25 - bar_len)
        lang_col = CYAN if src.language == "python" else YELLOW if src.language == "sas" else GREEN
        print(
            f"  {c(f'[{src.rank}]', YELLOW)}  "
            f"{c(src.language.upper(), lang_col)} / {src.chunk_type}"
            f"  {c('–', GREY)}  {c(src.name, BOLD)}\n"
            f"       {c(src.file, GREY)}  lines {src.start_line}–{src.end_line}\n"
            f"       Score: {c(bar, GREEN)} {src.score:.3f}"
        )
        preview = src.content_preview[:120].replace("\n", " ")
        print(f"       {c(preview + '…', GREY)}")
        print()


def print_stats(result) -> None:
    divider()
    conf_col = GREEN if result.confidence == "high" else YELLOW if result.confidence == "medium" else RED
    print(
        c("  Azure: ", GREY) + c(result.llm_model, CYAN) + "  │  " +
        c("Embed: ", GREY)   + c(result.embed_model, CYAN) + "  │  " +
        c("Index: ", GREY)   + c(result.index_name, CYAN) + "\n  " +
        c("Tokens in/out: ", GREY) +
        c(str(result.input_tokens), CYAN) + c("/", GREY) +
        c(str(result.output_tokens), CYAN) + "  │  " +
        c("Confidence: ", GREY) + c(result.confidence.upper(), conf_col) +
        f"  ({result.citation_count} citations)"
    )


def _print_env_error() -> None:
    print(c("\n  Missing Azure environment variables. Required:\n", RED))
    vars_ = [
        ("AZURE_OPENAI_ENDPOINT",    "https://my-aoai.openai.azure.com/"),
        ("AZURE_OPENAI_API_KEY",     "your Azure OpenAI key  (or use managed identity)"),
        ("AZURE_OPENAI_EMBED_DEPLOY","text-embedding-3-small  (deployment name)"),
        ("AZURE_OPENAI_CHAT_DEPLOY", "gpt-4o  (deployment name)"),
        ("AZURE_SEARCH_ENDPOINT",    "https://my-search.search.windows.net"),
        ("AZURE_SEARCH_API_KEY",     "your Azure Search key  (or use managed identity)"),
    ]
    for var, example in vars_:
        val = os.environ.get(var, "")
        status = c("✓", GREEN) if val else c("✗", RED)
        print(f"    {status}  {var:<35} {c(example if not val else val[:40], GREY)}")
    print(c("\n  See config/azure_config.yaml for provisioning guidance.\n", YELLOW))


# ---------------------------------------------------------------------------
# Index command
# ---------------------------------------------------------------------------

def cmd_index(args) -> None:
    from src.ingestion.azure_loader import AzureKnowledgeBaseLoader
    dirs = args.dirs or ["etl", "docs", "data/schemas"]
    print(c(f"\n  Indexing directories: {dirs}\n", YELLOW))

    loader = AzureKnowledgeBaseLoader(
        root_dirs            = dirs,
        index_name           = args.index_name,
        aoai_endpoint        = args.aoai_endpoint,
        aoai_api_key         = args.aoai_api_key,
        embed_deployment     = args.embed_deploy,
        search_endpoint      = args.search_endpoint,
        search_api_key       = args.search_api_key,
    )
    summary = loader.index(
        upload_blobs  = args.upload_blobs,
        drop_existing = args.rebuild,
    )
    print(c(f"\n  Done. Files={summary['files']} Chunks={summary['chunks']} "
            f"Indexed={summary['stored']}\n", GREEN))


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _build_where(args) -> dict | None:
    where: dict = {}
    if args.filter_lang:
        where["language"] = args.filter_lang
    if args.filter_type:
        where["chunk_type"] = args.filter_type
    return where if where else None


def run_query(engine, question: str, history: list, args) -> list:
    where = _build_where(args)
    print()
    print(c("Answer:", BOLD + GREEN))
    print()
    try:
        gen = engine.query_stream(question, history=history, where=where)
        for token in gen:
            print(token, end="", flush=True)
        print("\n")
        result = engine.get_last_result()
        if result:
            print_sources(result.sources, args.semantic)
            print_stats(result)
            return engine.get_last_history()
    except Exception as exc:
        msg = str(exc)
        print(c(f"\n  Error: {msg}", RED))
        if "401" in msg or "key" in msg.lower():
            _print_env_error()
        elif "404" in msg or "index" in msg.lower():
            print(c("\n  Hint: run  python scripts/query_azure.py --index  first.\n", YELLOW))
    return history


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Azure RAG query CLI – ETL Codebase Knowledge Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python scripts/query_azure.py --index
          python scripts/query_azure.py --query "What SAS macros handle null checks?"
          python scripts/query_azure.py --filter-lang sas --filter-type macro
          python scripts/query_azure.py --no-semantic   # Basic tier search
        """),
    )
    # Azure service config
    az = p.add_argument_group("Azure services")
    az.add_argument("--aoai-endpoint",   default=os.environ.get("AZURE_OPENAI_ENDPOINT",""),
                    help="Azure OpenAI endpoint URL")
    az.add_argument("--aoai-api-key",    default=os.environ.get("AZURE_OPENAI_API_KEY"),
                    help="Azure OpenAI API key (leave unset for managed identity)")
    az.add_argument("--embed-deploy",    default=os.environ.get("AZURE_OPENAI_EMBED_DEPLOY",
                                                                  "text-embedding-3-small"))
    az.add_argument("--chat-deploy",     default=os.environ.get("AZURE_OPENAI_CHAT_DEPLOY","gpt-4o"))
    az.add_argument("--search-endpoint", default=os.environ.get("AZURE_SEARCH_ENDPOINT",""),
                    help="Azure AI Search endpoint URL")
    az.add_argument("--search-api-key",  default=os.environ.get("AZURE_SEARCH_API_KEY"),
                    help="Azure AI Search admin key")
    az.add_argument("--index-name",      default=os.environ.get("AZURE_SEARCH_INDEX","etl-codebase"))
    az.add_argument("--no-semantic",     dest="semantic", action="store_false", default=True,
                    help="Disable semantic reranking (needed for Basic tier)")

    # Indexing
    idx = p.add_argument_group("Indexing")
    idx.add_argument("--index",       action="store_true", help="Index the codebase before querying")
    idx.add_argument("--rebuild",     action="store_true", help="Drop + recreate index before indexing")
    idx.add_argument("--dirs",        nargs="+", metavar="DIR",
                     help="Directories to index (default: etl docs data/schemas)")
    idx.add_argument("--upload-blobs",action="store_true",
                     help="Upload raw files to Azure Blob Storage")

    # Query
    qry = p.add_argument_group("Query")
    qry.add_argument("--query",       "-q", default=None, help="One-shot query (non-interactive)")
    qry.add_argument("--filter-lang", choices=["python","sas","markdown","yaml"],
                     help="Filter results by language")
    qry.add_argument("--filter-type", help="Filter by chunk type (macro, function, heading, ...)")
    qry.add_argument("--top-k",       type=int, default=8)
    qry.add_argument("--org",         default="our organisation",
                     help="Organisation name for system prompt context")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    # Validate minimum required env
    if not args.aoai_endpoint or not args.search_endpoint:
        _print_env_error()
        sys.exit(1)

    # Optional index step
    if args.index:
        cmd_index(args)
        if not args.query:
            return   # index-only run

    # Build engine
    from src.llm.azure_retrieval_engine import AzureRetrievalEngine
    engine = AzureRetrievalEngine(
        aoai_endpoint      = args.aoai_endpoint,
        aoai_api_key       = args.aoai_api_key,
        chat_deployment    = args.chat_deploy,
        embed_deployment   = args.embed_deploy,
        search_endpoint    = args.search_endpoint,
        search_api_key     = args.search_api_key,
        index_name         = args.index_name,
        top_k              = args.top_k,
        semantic_reranking = args.semantic,
        organisation       = args.org,
    )

    # One-shot mode
    if args.query:
        banner(args)
        print(c(f"Q: {args.query}", BOLD))
        run_query(engine, args.query, [], args)
        print()
        return

    # Interactive session
    banner(args)
    filters = []
    if args.filter_lang: filters.append(f"lang={args.filter_lang}")
    if args.filter_type: filters.append(f"type={args.filter_type}")
    filter_label = "  Filters: " + ", ".join(filters) if filters else ""
    print(c(
        f"  LLM: {args.chat_deploy}  │  Embed: {args.embed_deploy}  │  "
        f"Semantic: {'on' if args.semantic else 'off'}\n"
        f"  Org: {args.org}" +
        (f"\n{filter_label}" if filter_label else "") +
        "\n\n  Type 'quit' to exit  │  'reset' to clear history  │  "
        "'filter lang sas' to narrow results\n",
        GREY,
    ))
    divider()

    history: list[dict] = []

    while True:
        try:
            question = input(c("\nYou: ", BOLD + CYAN)).strip()
        except (KeyboardInterrupt, EOFError):
            print(c("\n\n  Goodbye.\n", GREY))
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print(c("\n  Goodbye.\n", GREY)); break
        if question.lower() == "reset":
            history = []
            print(c("  History cleared.\n", YELLOW)); continue
        if question.lower().startswith("filter lang "):
            args.filter_lang = question.split()[-1]
            print(c(f"  Language filter → {args.filter_lang}\n", YELLOW)); continue
        if question.lower().startswith("filter type "):
            args.filter_type = question.split()[-1]
            print(c(f"  Chunk-type filter → {args.filter_type}\n", YELLOW)); continue
        if question.lower() == "filter clear":
            args.filter_lang = args.filter_type = None
            print(c("  Filters cleared.\n", YELLOW)); continue

        history = run_query(engine, question, history, args)
        print()
        divider()


if __name__ == "__main__":
    main()
