"""
Azure Retrieval Engine
----------------------
Full RAG pipeline using Azure-native services:

  Azure Blob Storage         raw file storage (optional)
  Azure OpenAI Embeddings    text-embedding-3-small  →  1536-dim vectors
  Azure AI Search            hybrid vector+BM25+semantic reranking index
  Azure OpenAI GPT-4o        answer generation with organisational insights

Architecture
    User query
        ↓
    Azure OpenAI text-embedding-3-small  (query embedding)
        ↓
    Azure AI Search  (hybrid: vector ANN + BM25 + semantic reranker)
        with optional OData $filter  e.g. "language eq 'sas'"
        ↓
    Context assembly  (numbered [N] snippets)
        ↓
    Azure OpenAI GPT-4o  (streaming, org-specific system prompt)
        ↓
    RetrievalResult  (answer + sources + confidence + token usage)

Environment variables required
    AZURE_OPENAI_ENDPOINT        https://my-aoai.openai.azure.com/
    AZURE_OPENAI_API_KEY         optional (use managed identity if unset)
    AZURE_OPENAI_EMBED_DEPLOY    deployment name for embedding model
    AZURE_OPENAI_CHAT_DEPLOY     deployment name for chat model (gpt-4o)
    AZURE_OPENAI_API_VERSION     e.g. 2024-02-01
    AZURE_SEARCH_ENDPOINT        https://my-search.search.windows.net
    AZURE_SEARCH_API_KEY         optional (use managed identity if unset)
    AZURE_SEARCH_INDEX           index name (default: etl-codebase)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterator

from src.retrieval.azure_search_store import AzureSearchStore


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class SourceReference:
    rank: int
    file: str
    language: str
    chunk_type: str
    name: str
    start_line: int
    end_line: int
    score: float          # Azure search score (higher = better)
    content_preview: str

    def __str__(self) -> str:
        return (
            f"[{self.rank}] {self.language.upper()} / {self.chunk_type}"
            f"  –  {self.name}\n"
            f"     {self.file}  lines {self.start_line}–{self.end_line}"
            f"  (score {self.score:.3f})"
        )


@dataclass
class RetrievalResult:
    question: str
    answer: str
    sources: list[SourceReference]
    llm_model: str
    embed_model: str
    input_tokens: int
    output_tokens: int
    index_name: str
    confidence: str       # "high" | "medium" | "low"
    citation_count: int
    semantic_reranking: bool

    def format_sources(self) -> str:
        if not self.sources:
            return "No sources retrieved."
        return "\n".join(str(s) for s in self.sources)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior data engineering advisor for an enterprise data team.
You have expert knowledge of the organisation's Python and SAS-based ETL \
pipelines, data contracts, and data quality standards — all stored in the \
knowledge base you have access to.

When answering:
1. Ground your answer ONLY in the provided context snippets.  Cite as [N].
2. After the direct answer, add an **Organisational Insights** section with \
1–3 concrete, actionable recommendations visible from the code/contracts shown.
3. Reference exact file names, function names, macro names, column names, \
and line numbers.
4. If the context does not contain enough information, say so clearly.
5. Distinguish Python ETL from SAS ETL.  Reference data contracts when relevant.
6. Flag data quality risks, governance gaps, or SOX compliance concerns if \
visible in the retrieved context.

Context format:
  Snippets labelled [N] include source file, chunk type, and line range.
  Each snippet may be Python code, SAS code, a YAML schema, or a Markdown \
data contract section.
"""


# ---------------------------------------------------------------------------
# Azure Retrieval Engine
# ---------------------------------------------------------------------------

class AzureRetrievalEngine:
    """
    End-to-end RAG engine backed entirely by Azure services.

    Parameters
    ----------
    aoai_endpoint : str | None
        Azure OpenAI resource URL.  Defaults to AZURE_OPENAI_ENDPOINT.
    aoai_api_key : str | None
        Azure OpenAI API key.  Leave None for managed identity.
    chat_deployment : str
        GPT-4o deployment name in Azure OpenAI Studio.
        Defaults to AZURE_OPENAI_CHAT_DEPLOY or "gpt-4o".
    embed_deployment : str
        Embedding model deployment name.
        Defaults to AZURE_OPENAI_EMBED_DEPLOY or "text-embedding-3-small".
    search_endpoint : str | None
        Azure AI Search endpoint.  Defaults to AZURE_SEARCH_ENDPOINT.
    search_api_key : str | None
        Azure AI Search API key.  Leave None for managed identity.
    index_name : str
        Azure AI Search index name.  Default: "etl-codebase".
    top_k : int
        Number of results to retrieve per query.
    max_context_chunks : int
        Maximum chunks sent to GPT-4o after reranking.
    semantic_reranking : bool
        Enable Azure semantic reranker (recommended; requires Standard tier).
    organisation : str
        Organisation name injected into responses.
    """

    def __init__(
        self,
        aoai_endpoint: str | None    = None,
        aoai_api_key: str | None     = None,
        chat_deployment: str         = "",
        embed_deployment: str        = "",
        search_endpoint: str | None  = None,
        search_api_key: str | None   = None,
        index_name: str              = "etl-codebase",
        top_k: int                   = 8,
        max_context_chunks: int      = 6,
        semantic_reranking: bool     = True,
        organisation: str            = "our organisation",
        api_version: str             = "2024-02-01",
    ):
        self.chat_deployment    = (chat_deployment  or
                                   os.environ.get("AZURE_OPENAI_CHAT_DEPLOY", "gpt-4o"))
        self.embed_deployment   = (embed_deployment or
                                   os.environ.get("AZURE_OPENAI_EMBED_DEPLOY",
                                                  "text-embedding-3-small"))
        self.top_k              = top_k
        self.max_context_chunks = max_context_chunks
        self.semantic_reranking = semantic_reranking
        self.organisation       = organisation
        self.index_name         = index_name

        # Lazy-init
        self._embedder   = None
        self._store      = None
        self._aoai_client = None

        self._aoai_endpoint  = aoai_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self._aoai_api_key   = aoai_api_key  or os.environ.get("AZURE_OPENAI_API_KEY")
        self._search_endpoint = search_endpoint or os.environ.get("AZURE_SEARCH_ENDPOINT", "")
        self._search_api_key  = search_api_key  or os.environ.get("AZURE_SEARCH_API_KEY")
        self._api_version     = api_version

    # ── Service clients ───────────────────────────────────────────────────

    @property
    def embedder(self):
        if self._embedder is None:
            from src.embedding.azure_openai_embedder import AzureOpenAIEmbedder
            self._embedder = AzureOpenAIEmbedder(
                endpoint   = self._aoai_endpoint,
                api_key    = self._aoai_api_key,
                deployment = self.embed_deployment,
                api_version= self._api_version,
            )
        return self._embedder

    @property
    def store(self) -> AzureSearchStore:
        if self._store is None:
            self._store = AzureSearchStore(
                endpoint           = self._search_endpoint,
                api_key            = self._search_api_key,
                index_name         = self.index_name,
                semantic_reranking = self.semantic_reranking,
            )
        return self._store

    @property
    def aoai_client(self):
        if self._aoai_client is None:
            try:
                import openai as _openai
            except ImportError:
                raise ImportError("pip install openai")

            if self._aoai_api_key:
                self._aoai_client = _openai.AzureOpenAI(
                    azure_endpoint = self._aoai_endpoint,
                    api_key        = self._aoai_api_key,
                    api_version    = self._api_version,
                )
            else:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )
                self._aoai_client = _openai.AzureOpenAI(
                    azure_endpoint          = self._aoai_endpoint,
                    azure_ad_token_provider = token_provider,
                    api_version             = self._api_version,
                )
        return self._aoai_client

    # ── Confidence scoring ────────────────────────────────────────────────

    @staticmethod
    def _score_confidence(answer: str, sources: list[SourceReference]) -> tuple[str, int]:
        citations   = re.findall(r"\[(\d+)\]", answer)
        unique_cited = {int(c) for c in citations if c.isdigit()}
        count        = len(unique_cited)
        n            = len(sources)
        if n == 0 or count == 0:
            return "low", 0
        coverage = count / n
        if coverage >= 0.5 and count >= 2:
            return "high", count
        return "medium", count

    # ── Retrieval ─────────────────────────────────────────────────────────

    def _retrieve(
        self,
        query_text: str,
        where: dict | str | None = None,
    ) -> list[dict]:
        query_embedding = self.embedder.embed_query(query_text)
        hits = self.store.search(
            query_embedding = query_embedding,
            top_k           = self.top_k,
            query_text      = query_text,
            where           = where,
        )
        return hits[: self.max_context_chunks]

    # ── Context builder ───────────────────────────────────────────────────

    @staticmethod
    def _build_context(hits: list[dict]) -> tuple[str, list[SourceReference]]:
        parts:   list[str]             = []
        sources: list[SourceReference] = []

        for i, hit in enumerate(hits, 1):
            meta  = hit["metadata"]
            lang  = meta.get("language", "unknown")
            ctype = meta.get("chunk_type", "unknown")
            name  = meta.get("name", "unknown")
            src   = meta.get("source", "unknown")
            sl    = meta.get("start_line", "?")
            el    = meta.get("end_line", "?")
            score = hit.get("score", 0.0)

            header = (
                f"[{i}] {lang.upper()} / {ctype}  –  {name}\n"
                f"     File: {src}  lines {sl}–{el}"
            )
            parts.append(f"{header}\n\n{hit['content']}")

            sources.append(SourceReference(
                rank            = i,
                file            = src,
                language        = lang,
                chunk_type      = ctype,
                name            = name,
                start_line      = int(sl) if str(sl).isdigit() else 0,
                end_line        = int(el) if str(el).isdigit() else 0,
                score           = float(score),
                content_preview = hit["content"][:200],
            ))

        context_str = "\n\n" + ("─" * 60 + "\n\n").join(parts)
        return context_str, sources

    # ── Message builder ───────────────────────────────────────────────────

    def _build_messages(
        self,
        question: str,
        context: str,
        history: list[dict] | None,
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(history or [])
        messages.append({
            "role": "user",
            "content": (
                f"Retrieved context snippets:\n{context}\n\n"
                f"---\n\nQuestion: {question}"
            ),
        })
        return messages

    # ── Public API ────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        history: list[dict] | None = None,
        where: dict | str | None   = None,
    ) -> tuple[RetrievalResult, list[dict]]:
        """
        Non-streaming query. Returns (RetrievalResult, updated_history).

        Parameters
        ----------
        question : str  The user question.
        history  : list[dict] | None  Prior conversation turns.
        where    : dict | str | None
            Optional OData filter e.g. {"language": "sas"} or
            "language eq 'sas' and chunk_type eq 'macro'"

        Example
        -------
        >>> engine = AzureRetrievalEngine()
        >>> result, history = engine.query(
        ...     "What null checks does our SAS pipeline enforce?",
        ...     where={"language": "sas"},
        ... )
        >>> print(result.answer)
        >>> print(result.format_sources())
        """
        hits = self._retrieve(question, where)
        context, sources = self._build_context(hits)
        messages = self._build_messages(question, context, history)

        response = self.aoai_client.chat.completions.create(
            model       = self.chat_deployment,
            messages    = messages,
            max_tokens  = 4096,
            temperature = 0.2,
        )

        answer = response.choices[0].message.content or ""
        usage  = response.usage
        confidence, citation_count = self._score_confidence(answer, sources)

        updated_history = [m for m in messages if m["role"] != "system"]
        updated_history.append({"role": "assistant", "content": answer})

        result = RetrievalResult(
            question           = question,
            answer             = answer,
            sources            = sources,
            llm_model          = self.chat_deployment,
            embed_model        = self.embed_deployment,
            input_tokens       = usage.prompt_tokens,
            output_tokens      = usage.completion_tokens,
            index_name         = self.index_name,
            confidence         = confidence,
            citation_count     = citation_count,
            semantic_reranking = self.semantic_reranking,
        )
        return result, updated_history

    def query_stream(
        self,
        question: str,
        history: list[dict] | None = None,
        where: dict | str | None   = None,
    ) -> Iterator[str]:
        """
        Streaming query. Yields text delta strings from GPT-4o.
        Call get_last_result() after iteration for sources and usage.

        Example
        -------
        >>> engine = AzureRetrievalEngine()
        >>> for token in engine.query_stream("Explain the customer data contract"):
        ...     print(token, end="", flush=True)
        >>> result = engine.get_last_result()
        >>> print("\\n\\nSources:\\n", result.format_sources())
        """
        hits = self._retrieve(question, where)
        context, sources = self._build_context(hits)
        messages = self._build_messages(question, context, history)

        full_answer:  list[str] = []
        input_tokens  = 0
        output_tokens = 0

        with self.aoai_client.chat.completions.create(
            model        = self.chat_deployment,
            messages     = messages,
            max_tokens   = 4096,
            temperature  = 0.2,
            stream       = True,
            stream_options = {"include_usage": True},
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    full_answer.append(delta.content)
                    yield delta.content
                if chunk.usage:
                    input_tokens  = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens

        answer = "".join(full_answer)
        confidence, citation_count = self._score_confidence(answer, sources)

        self._last_result = RetrievalResult(
            question           = question,
            answer             = answer,
            sources            = sources,
            llm_model          = self.chat_deployment,
            embed_model        = self.embed_deployment,
            input_tokens       = input_tokens,
            output_tokens      = output_tokens,
            index_name         = self.index_name,
            confidence         = confidence,
            citation_count     = citation_count,
            semantic_reranking = self.semantic_reranking,
        )
        self._last_history = [m for m in messages if m["role"] != "system"] + [
            {"role": "assistant", "content": answer}
        ]

    def get_last_result(self) -> RetrievalResult | None:
        return getattr(self, "_last_result", None)

    def get_last_history(self) -> list[dict]:
        return getattr(self, "_last_history", [])
