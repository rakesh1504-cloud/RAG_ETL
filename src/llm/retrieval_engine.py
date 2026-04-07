"""
Retrieval Engine
----------------
Full RAG retrieval + generation pipeline powered by Claude claude-opus-4-6.

Architecture
    User query
        ↓
    Query expansion  (optional, improves recall on short queries)
        ↓
    Multi-collection vector search  (etl_codebase + rag_collection)
        ↓
    Score-based re-ranking + deduplication
        ↓
    Context assembly  (ranked chunks formatted for Claude)
        ↓
    Claude claude-opus-4-6 with adaptive thinking + streaming
        ↓
    RetrievalResult  (answer + sources + usage metadata)

Key design choices
    • claude-opus-4-6 with adaptive thinking: Claude decides per-query how much
      internal reasoning to apply – ideal for technical ETL/code questions that
      need multi-step analysis, while still answering simple lookups quickly.
    • Streaming enabled for all responses: avoids HTTP timeouts on long code
      explanations; callers can display tokens as they arrive.
    • Multi-turn conversation: caller passes `history` (list of
      {"role", "content"} dicts); engine appends and returns the updated list.
    • Source attribution: every answer includes the exact file, chunk type, and
      line range of each retrieved chunk used.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Iterator

import anthropic

from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Result / Source data classes
# ---------------------------------------------------------------------------

@dataclass
class SourceReference:
    """Metadata for a single retrieved chunk used in the answer."""
    rank: int
    file: str
    language: str
    chunk_type: str
    name: str
    start_line: int
    end_line: int
    distance: float
    content_preview: str   # first 200 chars

    def __str__(self) -> str:
        return (
            f"[{self.rank}] {self.language.upper()} / {self.chunk_type}"
            f"  –  {self.name}\n"
            f"     {self.file}  lines {self.start_line}–{self.end_line}"
            f"  (score {1 - self.distance:.3f})"
        )


@dataclass
class RetrievalResult:
    """Complete result returned by RetrievalEngine.query()."""
    question: str
    answer: str
    sources: list[SourceReference]
    model: str
    thinking_used: bool
    input_tokens: int
    output_tokens: int
    collections_searched: list[str]
    confidence: str = "unknown"     # "high" | "medium" | "low" | "unknown"
    citation_count: int = 0         # number of [N] citations found in answer
    hybrid_search: bool = False     # True if BM25 + vector RRF was used

    def format_sources(self) -> str:
        if not self.sources:
            return "No sources retrieved."
        return "\n".join(str(s) for s in self.sources)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert assistant for an ETL and RAG pipeline codebase built with \
Python and SAS. You help engineers understand, debug, and extend the codebase.

When answering:
1. Ground your answer ONLY in the provided context snippets.
2. Reference specific files, function names, macro names, or line numbers when relevant.
3. If the context does not contain enough information, say so clearly \
   and suggest where the user might look.
4. For code questions, prefer to show concise, working code examples.
5. For SAS questions, use proper SAS syntax and macro conventions.
6. Distinguish clearly between Python ETL logic and SAS ETL logic when both are relevant.

Context format:
  Each snippet is labelled [N] with its source file, chunk type, and name.
  Cite sources as [N] in your answer.
"""


# ---------------------------------------------------------------------------
# Retrieval Engine
# ---------------------------------------------------------------------------

class RetrievalEngine:
    """
    End-to-end retrieval + generation engine.

    Parameters
    ----------
    model : str
        Claude model to use.  Default: claude-opus-4-6 (best reasoning).
    collections : list[str]
        ChromaDB collection names to search.  Results are merged and re-ranked.
    vector_store_dir : str
        Path to the ChromaDB persist directory.
    top_k_per_collection : int
        Chunks to retrieve from each collection before merging.
    max_context_chunks : int
        Maximum chunks to include in the LLM context after re-ranking.
    use_thinking : bool
        Enable adaptive thinking (recommended for complex code questions).
    expand_query : bool
        Prepend a brief query expansion step to improve recall.
    use_hybrid : bool
        Enable BM25 + vector hybrid search with RRF.  Catches exact keyword
        hits (macro names, column names) that vector search alone misses.
    where : dict | None
        Default metadata filter applied to every search, e.g.
        ``{"language": {"$eq": "sas"}}``.  Can be overridden per-query.
    """

    def __init__(
        self,
        model: str                  = "claude-opus-4-6",
        collections: list[str]      = None,
        vector_store_dir: str       = "data/vector_store",
        top_k_per_collection: int   = 5,
        max_context_chunks: int     = 8,
        use_thinking: bool          = True,
        expand_query: bool          = False,
        use_hybrid: bool            = True,
        where: dict | None          = None,
    ):
        self.model                  = model
        self.collections            = collections or ["etl_codebase"]
        self.vector_store_dir       = vector_store_dir
        self.top_k_per_collection   = top_k_per_collection
        self.max_context_chunks     = max_context_chunks
        self.use_thinking           = use_thinking
        self.expand_query           = expand_query
        self.use_hybrid             = use_hybrid
        self.default_where          = where

        self.client   = anthropic.Anthropic()
        self.embedder = Embedder(tfidf_persist=f"{vector_store_dir}/tfidf_embedder.pkl")
        self._stores:    dict[str, VectorStore]     = {}
        self._hybrids:   dict[str, HybridRetriever] = {}

    # ── Vector store accessor ────────────────────────────────────────────────

    def _store(self, collection: str) -> VectorStore:
        if collection not in self._stores:
            self._stores[collection] = VectorStore(
                collection_name=collection,
                persist_dir=self.vector_store_dir,
            )
        return self._stores[collection]

    def _hybrid(self, collection: str) -> HybridRetriever:
        if collection not in self._hybrids:
            hr = HybridRetriever(
                collection_name=collection,
                vector_store_dir=self.vector_store_dir,
            )
            hr.build_bm25_index()
            self._hybrids[collection] = hr
        return self._hybrids[collection]

    # ── Query expansion (optional) ────────────────────────────────────────────

    def _expand_query(self, question: str) -> str:
        """
        Use Claude Haiku to rephrase the query for better lexical recall.
        Only called when expand_query=True.
        """
        resp = self.client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=256,
            system=(
                "You are a query expansion assistant. "
                "Given a user question about an ETL or RAG codebase, "
                "return 2-3 alternative phrasings as a single comma-separated line. "
                "No explanation, no bullet points – just the rephrased queries."
            ),
            messages=[{"role": "user", "content": question}],
        )
        expansion = next((b.text for b in resp.content if b.type == "text"), question)
        return f"{question}, {expansion}"

    # ── Confidence scoring ────────────────────────────────────────────────────

    @staticmethod
    def _score_confidence(answer: str, sources: list[SourceReference]) -> tuple[str, int]:
        """
        Derive a confidence level from citation coverage in the answer.

        Returns (confidence_label, citation_count).
        """
        citations = re.findall(r"\[(\d+)\]", answer)
        unique_cited = set(int(c) for c in citations if c.isdigit())
        citation_count = len(unique_cited)
        n_sources = len(sources)

        if n_sources == 0:
            return "low", 0

        coverage = citation_count / n_sources if n_sources else 0

        if citation_count == 0:
            confidence = "low"          # answered with no citations – risk of hallucination
        elif coverage >= 0.5 and citation_count >= 2:
            confidence = "high"         # cited ≥50% of sources and ≥2 distinct sources
        else:
            confidence = "medium"       # some citations but limited coverage

        return confidence, citation_count

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve(
        self,
        query_text: str,
        where: dict | None = None,
    ) -> tuple[list[dict], bool]:
        """
        Embed query, search all collections, merge results, re-rank.
        Returns (hits, hybrid_used).
        """
        effective_where = where or self.default_where
        query_embedding = self.embedder.embed_query(query_text)
        all_hits: list[dict] = []
        hybrid_used = False

        for coll_name in self.collections:
            try:
                if self.use_hybrid:
                    hr = self._hybrid(coll_name)
                    hits = hr.search(
                        query_embedding=query_embedding,
                        query_text=query_text,
                        top_k=self.top_k_per_collection,
                        where=effective_where,
                    )
                    hybrid_used = hr._bm25_available
                else:
                    store = self._store(coll_name)
                    hits  = store.search(
                        query_embedding,
                        top_k=self.top_k_per_collection,
                        where=effective_where,
                    )
                for h in hits:
                    h["collection"] = coll_name
                all_hits.extend(hits)
            except Exception as exc:
                print(f"  Warning: could not search collection '{coll_name}': {exc}")

        # De-duplicate by content, keep lowest distance
        seen: dict[str, dict] = {}
        for h in all_hits:
            key = h["content"][:120]   # use first 120 chars as fingerprint
            if key not in seen or h["distance"] < seen[key]["distance"]:
                seen[key] = h

        # Sort ascending (lower distance = more similar)
        ranked = sorted(seen.values(), key=lambda x: x["distance"])
        return ranked[: self.max_context_chunks], hybrid_used

    # ── Context builder ───────────────────────────────────────────────────────

    @staticmethod
    def _build_context(hits: list[dict]) -> tuple[str, list[SourceReference]]:
        """
        Format hits into a numbered context string for the LLM,
        and build SourceReference objects for the result.
        """
        parts: list[str] = []
        sources: list[SourceReference] = []

        for i, hit in enumerate(hits, 1):
            meta  = hit["metadata"]
            lang  = meta.get("language", "unknown")
            ctype = meta.get("chunk_type", "unknown")
            name  = meta.get("name", "unknown")
            src   = meta.get("source", "unknown")
            sl    = meta.get("start_line", "?")
            el    = meta.get("end_line", "?")

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
                distance        = hit["distance"],
                content_preview = hit["content"][:200],
            ))

        context_str = "\n\n" + ("─" * 60 + "\n\n").join(parts)
        return context_str, sources

    # ── Generation ────────────────────────────────────────────────────────────

    def _build_messages(
        self,
        question: str,
        context: str,
        history: list[dict] | None,
    ) -> list[dict]:
        messages = list(history or [])
        user_content = (
            f"Retrieved context snippets:\n{context}\n\n"
            f"---\n\nQuestion: {question}"
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    # ── Public API ────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        history: list[dict] | None = None,
    ) -> tuple[RetrievalResult, list[dict]]:
        """
        Non-streaming query.  Returns (RetrievalResult, updated_history).

        Example
        -------
        >>> engine = RetrievalEngine()
        >>> result, history = engine.query("How does the SAS null check macro work?")
        >>> print(result.answer)
        >>> print(result.format_sources())
        """
        # 1. Optionally expand query
        query_text = self._expand_query(question) if self.expand_query else question

        # 2. Retrieve (hybrid BM25+vector or vector-only)
        hits, hybrid_used = self._retrieve(query_text)
        context, sources  = self._build_context(hits)

        # 3. Build messages
        messages = self._build_messages(question, context, history)

        # 4. Build API params
        params: dict = dict(
            model      = self.model,
            max_tokens = 4096,
            system     = _SYSTEM_PROMPT,
            messages   = messages,
        )
        if self.use_thinking:
            params["thinking"] = {"type": "adaptive"}

        # 5. Call Claude
        response = self.client.messages.create(**params)

        answer        = next((b.text for b in response.content if b.type == "text"), "")
        thinking_used = any(b.type == "thinking" for b in response.content)

        # 6. Confidence scoring
        confidence, citation_count = self._score_confidence(answer, sources)

        # 7. Update history (text only – don't carry thinking blocks forward)
        updated_history = list(messages)
        updated_history.append({"role": "assistant", "content": answer})

        result = RetrievalResult(
            question             = question,
            answer               = answer,
            sources              = sources,
            model                = self.model,
            thinking_used        = thinking_used,
            input_tokens         = response.usage.input_tokens,
            output_tokens        = response.usage.output_tokens,
            collections_searched = self.collections,
            confidence           = confidence,
            citation_count       = citation_count,
            hybrid_search        = hybrid_used,
        )
        return result, updated_history

    def query_stream(
        self,
        question: str,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        """
        Streaming query.  Yields text delta strings as they arrive from Claude.
        Call ``get_last_result()`` after iteration for sources and usage.

        Example
        -------
        >>> engine = RetrievalEngine()
        >>> for token in engine.query_stream("What does ETLPipeline.run() do?"):
        ...     print(token, end="", flush=True)
        >>> result = engine.get_last_result()
        """
        query_text = self._expand_query(question) if self.expand_query else question
        hits, hybrid_used = self._retrieve(query_text)
        context, sources  = self._build_context(hits)
        messages          = self._build_messages(question, context, history)

        params: dict = dict(
            model      = self.model,
            max_tokens = 4096,
            system     = _SYSTEM_PROMPT,
            messages   = messages,
        )
        if self.use_thinking:
            params["thinking"] = {"type": "adaptive"}

        full_answer: list[str] = []
        thinking_used = False

        with self.client.messages.stream(**params) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "thinking":
                        thinking_used = True
                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        full_answer.append(event.delta.text)
                        yield event.delta.text

            final = stream.get_final_message()

        answer = "".join(full_answer)
        confidence, citation_count = self._score_confidence(answer, sources)

        # Store last result for post-iteration access
        self._last_result = RetrievalResult(
            question             = question,
            answer               = answer,
            sources              = sources,
            model                = self.model,
            thinking_used        = thinking_used,
            input_tokens         = final.usage.input_tokens,
            output_tokens        = final.usage.output_tokens,
            collections_searched = self.collections,
            confidence           = confidence,
            citation_count       = citation_count,
            hybrid_search        = hybrid_used,
        )
        self._last_history = list(messages) + [
            {"role": "assistant", "content": answer}
        ]

    def get_last_result(self) -> RetrievalResult | None:
        """Return the RetrievalResult from the most recent query_stream() call."""
        return getattr(self, "_last_result", None)

    def get_last_history(self) -> list[dict]:
        """Return the updated conversation history after the last query_stream()."""
        return getattr(self, "_last_history", [])
