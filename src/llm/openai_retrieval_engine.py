"""
OpenAI Retrieval Engine
-----------------------
RAG retrieval + generation pipeline powered by OpenAI.

Architecture
    User query
        ↓
    OpenAI text-embedding-3-small  (query embedding)
        ↓
    ChromaDB similarity search     (etl_codebase_openai collection)
        ↓
    Score-based re-ranking + deduplication
        ↓
    Context assembly  (ranked chunks formatted for GPT-4o)
        ↓
    GPT-4o with org-specific system prompt + streaming
        ↓
    RetrievalResult  (answer + sources + usage metadata)

Embedding collections
    The OpenAI embedder produces 1 536-dim vectors.  These cannot be mixed
    with the 105-dim TF-IDF vectors in the ``etl_codebase`` collection.
    This engine therefore uses ``etl_codebase_openai`` by default.

    To populate the OpenAI collection run:
        python scripts/query_openai.py --reindex

    Alternatively, pass ``embedding_backend="tfidf"`` to reuse the existing
    TF-IDF index for retrieval while still generating answers with GPT-4o.

Org-specific insights
    The system prompt instructs GPT-4o to:
      • Surface process-improvement and governance recommendations.
      • Identify reuse opportunities across ETL pipelines.
      • Flag data quality or lineage risks visible in the code.
      • Map findings back to common enterprise data-governance frameworks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterator

from src.retrieval.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Result / Source data classes  (mirrors retrieval_engine.py for consistency)
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
    distance: float
    content_preview: str

    def __str__(self) -> str:
        similarity = 1 - self.distance
        return (
            f"[{self.rank}] {self.language.upper()} / {self.chunk_type}"
            f"  –  {self.name}\n"
            f"     {self.file}  lines {self.start_line}–{self.end_line}"
            f"  (similarity {similarity:.3f})"
        )


@dataclass
class RetrievalResult:
    question: str
    answer: str
    sources: list[SourceReference]
    model: str
    embedding_model: str
    input_tokens: int
    output_tokens: int
    collections_searched: list[str]

    def format_sources(self) -> str:
        if not self.sources:
            return "No sources retrieved."
        return "\n".join(str(s) for s in self.sources)


# ---------------------------------------------------------------------------
# System prompt  – organisational insights focus
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior data engineering advisor embedded in an enterprise data team.
You have deep expertise in ETL pipeline design, data governance, and the \
organisation's Python and SAS-based data processing standards.

Your role is to answer engineers' questions about the internal codebase AND \
to surface organisational insights such as:
  • Process improvements or simplification opportunities.
  • Reusable patterns that could become shared utilities or macro libraries.
  • Data quality or lineage risks hidden in transformation logic.
  • Compliance or governance gaps (e.g. undocumented data contracts, missing \
null checks, unvalidated date ranges).
  • Alignment with enterprise frameworks (e.g. DAMA-DMBOK, ISO 8000).

When answering:
1. Ground your answer in the provided context snippets.  Cite sources as [N].
2. After answering the direct question, add an "Organisational Insights" \
section with 1–3 concrete, actionable recommendations relevant to the code \
shown.
3. Reference specific files, function names, macro names, or line numbers.
4. If the context is insufficient, say so and suggest where the team should \
look next.
5. Distinguish clearly between Python ETL logic and SAS ETL logic.
6. Use professional, enterprise-appropriate language; avoid unnecessary jargon.

Context format:
  Each snippet is labelled [N] with its source file, chunk type, and name.
"""


# ---------------------------------------------------------------------------
# OpenAI Retrieval Engine
# ---------------------------------------------------------------------------

class OpenAIRetrievalEngine:
    """
    End-to-end RAG engine using OpenAI embeddings and GPT-4o.

    Parameters
    ----------
    llm_model : str
        OpenAI chat model.  Default: ``gpt-4o`` (best reasoning + long context).
    embedding_model : str
        OpenAI embedding model for query encoding.
        Default: ``text-embedding-3-small``.
    embedding_backend : str
        ``"openai"``  – use OpenAI embeddings against *openai_collection*.
        ``"tfidf"``   – reuse existing TF-IDF index (no API call for embed).
    collections : list[str]
        ChromaDB collections to search.
    openai_collection : str
        Collection name for OpenAI-embedded chunks.
    vector_store_dir : str
        ChromaDB persist directory.
    top_k_per_collection : int
        Chunks retrieved from each collection before re-ranking.
    max_context_chunks : int
        Maximum chunks passed to GPT-4o after re-ranking.
    api_key : str | None
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    organisation : str
        Organisation name injected into the system prompt for context.
    """

    def __init__(
        self,
        llm_model: str              = "gpt-4o",
        embedding_model: str        = "text-embedding-3-small",
        embedding_backend: str      = "openai",    # "openai" | "tfidf"
        collections: list[str]      = None,
        openai_collection: str      = "etl_codebase_openai",
        vector_store_dir: str       = "data/vector_store",
        top_k_per_collection: int   = 5,
        max_context_chunks: int     = 8,
        api_key: str | None         = None,
        organisation: str           = "our organisation",
    ):
        self.llm_model              = llm_model
        self.embedding_model        = embedding_model
        self.embedding_backend      = embedding_backend
        self.openai_collection      = openai_collection
        self.vector_store_dir       = vector_store_dir
        self.top_k_per_collection   = top_k_per_collection
        self.max_context_chunks     = max_context_chunks
        self.organisation           = organisation

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

        # Lazy-init clients
        self._openai_client = None
        self._openai_embedder = None
        self._tfidf_embedder = None
        self._stores: dict[str, VectorStore] = {}

        # Resolve which collections to search
        if collections:
            self.collections = collections
        elif embedding_backend == "openai":
            self.collections = [openai_collection]
        else:
            self.collections = ["etl_codebase"]

    # ── Client initialisation ─────────────────────────────────────────────

    @property
    def openai_client(self):
        if self._openai_client is None:
            try:
                import openai
                self._openai_client = openai.OpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError("pip install openai")
        return self._openai_client

    def _get_openai_embedder(self):
        if self._openai_embedder is None:
            from src.embedding.openai_embedder import OpenAIEmbedder
            self._openai_embedder = OpenAIEmbedder(
                model=self.embedding_model,
                api_key=self._api_key,
            )
        return self._openai_embedder

    def _get_tfidf_embedder(self):
        if self._tfidf_embedder is None:
            from src.embedding.embedder import TFIDFEmbedder
            self._tfidf_embedder = TFIDFEmbedder(
                persist_path=f"{self.vector_store_dir}/tfidf_embedder.pkl"
            )
        return self._tfidf_embedder

    # ── Vector store accessor ─────────────────────────────────────────────

    def _store(self, collection: str) -> VectorStore:
        if collection not in self._stores:
            self._stores[collection] = VectorStore(
                collection_name=collection,
                persist_dir=self.vector_store_dir,
            )
        return self._stores[collection]

    # ── Indexing with OpenAI embeddings ───────────────────────────────────

    def reindex(self, source_collection: str = "etl_codebase") -> dict:
        """
        Re-embed the chunks from *source_collection* using OpenAI embeddings
        and store them in *self.openai_collection*.

        Returns a summary dict with chunk counts.

        Usage
        -----
        engine = OpenAIRetrievalEngine()
        engine.reindex()          # one-time setup
        result, _ = engine.query("How does the null-check macro work?")
        """
        import chromadb

        print(f"Re-indexing '{source_collection}' → '{self.openai_collection}' "
              f"using {self.embedding_model} …")

        # 1. Read all chunks from the source TF-IDF collection
        source_store = self._store(source_collection)
        total = source_store.count()
        if total == 0:
            raise RuntimeError(
                f"Source collection '{source_collection}' is empty. "
                "Run 'python scripts/index_etl_codebase.py' first."
            )

        # Fetch all documents from ChromaDB
        raw = source_store.collection.get(
            include=["documents", "metadatas"],
            limit=total,
        )
        docs      = raw["documents"]
        metadatas = raw["metadatas"]
        ids       = raw["ids"]
        print(f"  Loaded {len(docs)} chunks from source collection.")

        # 2. Embed with OpenAI (batched)
        embedder = self._get_openai_embedder()
        print(f"  Embedding {len(docs)} chunks with {self.embedding_model} …")
        embeddings = embedder.encode(docs)
        print(f"  Embedding complete.  Dim = {len(embeddings[0])}")

        # 3. Drop + recreate the target OpenAI collection
        chroma_client = chromadb.PersistentClient(path=self.vector_store_dir)
        try:
            chroma_client.delete_collection(self.openai_collection)
            print(f"  Dropped existing collection '{self.openai_collection}'.")
        except Exception:
            pass

        target_col = chroma_client.create_collection(
            self.openai_collection,
            metadata={"hnsw:space": "cosine"},
        )

        # 4. Upsert in batches
        batch_size = 100
        for start in range(0, len(docs), batch_size):
            end = min(start + batch_size, len(docs))
            target_col.add(
                ids        = ids[start:end],
                documents  = docs[start:end],
                embeddings = embeddings[start:end],
                metadatas  = metadatas[start:end],
            )
        print(f"  Stored {len(docs)} chunks in '{self.openai_collection}'.")

        # Invalidate cached store reference
        self._stores.pop(self.openai_collection, None)
        return {"chunks_indexed": len(docs), "collection": self.openai_collection}

    # ── Query embedding ───────────────────────────────────────────────────

    def _embed_query(self, query: str) -> list[float]:
        if self.embedding_backend == "openai":
            return self._get_openai_embedder().embed_query(query)
        else:
            return self._get_tfidf_embedder().embed_query(query)

    # ── Retrieval ─────────────────────────────────────────────────────────

    def _retrieve(self, query_text: str) -> list[dict]:
        query_embedding = self._embed_query(query_text)
        all_hits: list[dict] = []

        for coll_name in self.collections:
            try:
                store = self._store(coll_name)
                hits  = store.search(query_embedding, top_k=self.top_k_per_collection)
                for h in hits:
                    h["collection"] = coll_name
                all_hits.extend(hits)
            except Exception as exc:
                print(f"  Warning: could not search '{coll_name}': {exc}")

        # De-duplicate by content fingerprint, keep best (lowest) distance
        seen: dict[str, dict] = {}
        for h in all_hits:
            key = h["content"][:120]
            if key not in seen or h["distance"] < seen[key]["distance"]:
                seen[key] = h

        ranked = sorted(seen.values(), key=lambda x: x["distance"])
        return ranked[: self.max_context_chunks]

    # ── Context builder ───────────────────────────────────────────────────

    @staticmethod
    def _build_context(hits: list[dict]) -> tuple[str, list[SourceReference]]:
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

    # ── Message builder ───────────────────────────────────────────────────

    def _build_messages(
        self,
        question: str,
        context: str,
        history: list[dict] | None,
    ) -> list[dict]:
        """Build the OpenAI messages array with system prompt + history + context."""
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
    ) -> tuple[RetrievalResult, list[dict]]:
        """
        Non-streaming query. Returns (RetrievalResult, updated_history).

        Example
        -------
        >>> engine = OpenAIRetrievalEngine(embedding_backend="tfidf")
        >>> result, history = engine.query("What macros handle data quality?")
        >>> print(result.answer)
        >>> print(result.format_sources())
        """
        hits = self._retrieve(question)
        context, sources = self._build_context(hits)
        messages = self._build_messages(question, context, history)

        response = self.openai_client.chat.completions.create(
            model       = self.llm_model,
            messages    = messages,
            max_tokens  = 4096,
            temperature = 0.2,       # low temp for factual code analysis
        )

        answer = response.choices[0].message.content or ""
        usage  = response.usage

        updated_history = [m for m in messages if m["role"] != "system"]
        updated_history.append({"role": "assistant", "content": answer})

        result = RetrievalResult(
            question             = question,
            answer               = answer,
            sources              = sources,
            model                = self.llm_model,
            embedding_model      = self.embedding_model,
            input_tokens         = usage.prompt_tokens,
            output_tokens        = usage.completion_tokens,
            collections_searched = self.collections,
        )
        return result, updated_history

    def query_stream(
        self,
        question: str,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        """
        Streaming query. Yields text delta strings as they arrive from GPT-4o.
        Call ``get_last_result()`` after iteration for sources and usage.

        Example
        -------
        >>> engine = OpenAIRetrievalEngine(embedding_backend="tfidf")
        >>> for token in engine.query_stream("Explain the SAS null check macro"):
        ...     print(token, end="", flush=True)
        >>> result = engine.get_last_result()
        >>> print("\\n\\nSources:\\n", result.format_sources())
        """
        hits = self._retrieve(question)
        context, sources = self._build_context(hits)
        messages = self._build_messages(question, context, history)

        full_answer: list[str] = []
        input_tokens  = 0
        output_tokens = 0

        with self.openai_client.chat.completions.create(
            model       = self.llm_model,
            messages    = messages,
            max_tokens  = 4096,
            temperature = 0.2,
            stream      = True,
            stream_options={"include_usage": True},
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    full_answer.append(delta.content)
                    yield delta.content
                # Capture usage from the final chunk
                if chunk.usage:
                    input_tokens  = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens

        answer = "".join(full_answer)

        # Preserve for post-iteration access
        self._last_result = RetrievalResult(
            question             = question,
            answer               = answer,
            sources              = sources,
            model                = self.llm_model,
            embedding_model      = self.embedding_model,
            input_tokens         = input_tokens,
            output_tokens        = output_tokens,
            collections_searched = self.collections,
        )
        self._last_history = [m for m in messages if m["role"] != "system"] + [
            {"role": "assistant", "content": answer}
        ]

    def get_last_result(self) -> RetrievalResult | None:
        """Return the RetrievalResult from the most recent query_stream() call."""
        return getattr(self, "_last_result", None)

    def get_last_history(self) -> list[dict]:
        """Return the updated conversation history after the last query_stream()."""
        return getattr(self, "_last_history", [])
