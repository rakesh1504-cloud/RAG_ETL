"""
Hybrid Retriever
----------------
Combines dense vector search (ChromaDB) with sparse BM25 keyword search,
then merges results using Reciprocal Rank Fusion (RRF).

Why hybrid?
    Vector search excels at semantic similarity but can miss exact names
    (e.g. "%assert_no_nulls", "ETL_REJECT_LOG", "etl_batch_id") that appear
    verbatim in the query.  BM25 catches these exact-match cases.
    RRF merges both rank lists without needing score normalisation.

RRF formula
    score(doc) = Σ  1 / (k + rank_i)
    where k=60 (standard default) and rank_i is the 1-based rank in list i.

BM25 backend
    Uses the ``rank_bm25`` package (pure Python, no network required).
    Falls back gracefully to vector-only if the package is not installed.

Usage
-----
    retriever = HybridRetriever(
        collection_name="etl_codebase",
        vector_store_dir="data/vector_store",
    )
    # One-time: load corpus from ChromaDB into BM25 index
    retriever.build_bm25_index()

    # Search
    hits = retriever.search(
        query_embedding=embedder.embed_query("null check SAS macro"),
        query_text="null check SAS macro",
        top_k=8,
        where={"language": {"$eq": "sas"}},   # optional metadata filter
    )
"""

from __future__ import annotations

import re
from typing import Any

from src.retrieval.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Tokeniser (shared between BM25 index and query)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_%]+")


def _tokenise(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokeniser that preserves:
      - snake_case identifiers (kept whole)
      - SAS macro names (%macro_name → macro_name token + full token)
      - CamelCase words (kept as-is; not split)
    """
    tokens = _TOKEN_RE.findall(text.lower())
    # Also add split tokens for snake_case  (e.g. "etl_batch_id" → also "etl", "batch", "id")
    extra: list[str] = []
    for t in tokens:
        parts = t.split("_")
        if len(parts) > 1:
            extra.extend(p for p in parts if p)
    return tokens + extra


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Dense vector search + sparse BM25, merged with Reciprocal Rank Fusion.

    Parameters
    ----------
    collection_name : str
        ChromaDB collection to search.
    vector_store_dir : str
        ChromaDB persist directory.
    rrf_k : int
        RRF smoothing constant.  Default: 60 (standard).
    bm25_weight : float
        Weight applied to BM25 rank scores before fusion.  1.0 = equal weight
        with vector search.  Increase (e.g. 1.5) to favour exact-keyword hits.
    """

    def __init__(
        self,
        collection_name: str = "etl_codebase",
        vector_store_dir: str = "data/vector_store",
        rrf_k: int   = 60,
        bm25_weight: float = 1.0,
    ):
        self.rrf_k        = rrf_k
        self.bm25_weight  = bm25_weight
        self._store = VectorStore(
            collection_name=collection_name,
            persist_dir=vector_store_dir,
        )
        # BM25 state
        self._bm25         = None
        self._corpus_docs:  list[str]  = []
        self._corpus_metas: list[dict] = []
        self._corpus_ids:   list[str]  = []
        self._bm25_available = False

    # ── BM25 index ────────────────────────────────────────────────────────

    def build_bm25_index(self) -> int:
        """
        Load all documents from ChromaDB and fit a BM25 index in memory.
        Must be called before any search that uses BM25.
        Returns the number of documents indexed.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print(
                "  [HybridRetriever] rank_bm25 not installed – "
                "falling back to vector-only search.\n"
                "  Install with: pip install rank-bm25"
            )
            self._bm25_available = False
            return 0

        raw = self._store.get_all()
        self._corpus_docs  = raw.get("documents") or []
        self._corpus_metas = raw.get("metadatas") or []
        self._corpus_ids   = raw.get("ids") or []

        if not self._corpus_docs:
            print("  [HybridRetriever] Collection is empty; BM25 index not built.")
            self._bm25_available = False
            return 0

        tokenised = [_tokenise(doc) for doc in self._corpus_docs]
        self._bm25 = BM25Okapi(tokenised)
        self._bm25_available = True
        print(f"  [HybridRetriever] BM25 index built over {len(self._corpus_docs)} documents.")
        return len(self._corpus_docs)

    # ── BM25 search ───────────────────────────────────────────────────────

    def _bm25_search(
        self,
        query_text: str,
        top_k: int,
        where: dict | None,
    ) -> list[dict]:
        """Return top_k BM25 hits, optionally filtered by metadata."""
        if not self._bm25_available or self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenise(query_text))

        # Apply metadata filter before ranking
        candidates: list[tuple[float, int]] = []
        for idx, score in enumerate(scores):
            if where and not _matches_where(self._corpus_metas[idx], where):
                continue
            candidates.append((score, idx))

        # Sort descending by BM25 score
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:top_k]

        hits: list[dict] = []
        for score, idx in top:
            hits.append({
                "content":    self._corpus_docs[idx],
                "metadata":   self._corpus_metas[idx],
                "bm25_score": score,
                "distance":   1.0,   # placeholder; overwritten by RRF fusion
                "collection": self._store.collection_name,
            })
        return hits

    # ── RRF fusion ────────────────────────────────────────────────────────

    def _rrf_merge(
        self,
        vector_hits: list[dict],
        bm25_hits: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        score(doc) = 1/(k + rank_vector) + bm25_weight/(k + rank_bm25)

        Documents only in one list still receive a partial RRF score.
        """
        k = self.rrf_k
        scores: dict[str, float] = {}
        docs_map: dict[str, dict] = {}

        # Vector hits (ranked 1..N)
        for rank, hit in enumerate(vector_hits, 1):
            key = hit["content"][:120]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs_map[key] = hit

        # BM25 hits (ranked 1..N, weighted)
        for rank, hit in enumerate(bm25_hits, 1):
            key = hit["content"][:120]
            scores[key] = scores.get(key, 0.0) + self.bm25_weight / (k + rank)
            if key not in docs_map:
                docs_map[key] = hit

        # Sort by descending RRF score
        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)

        merged: list[dict] = []
        for key in sorted_keys[:top_k]:
            doc = dict(docs_map[key])
            # Store RRF score as inverted distance (lower = better for downstream)
            doc["rrf_score"] = scores[key]
            doc["distance"]  = 1.0 - min(scores[key], 1.0)   # normalise to [0, 1]
            merged.append(doc)
        return merged

    # ── Public search API ─────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 8,
        where: dict | None = None,
        vector_candidates: int | None = None,
    ) -> list[dict]:
        """
        Hybrid search: vector + BM25 → RRF merge.

        Parameters
        ----------
        query_embedding : list[float]
            Dense embedding of the query (from Embedder or OpenAIEmbedder).
        query_text : str
            Raw query string (used for BM25 tokenisation).
        top_k : int
            Final number of results to return after fusion.
        where : dict | None
            ChromaDB metadata filter applied to vector search and BM25 results.
        vector_candidates : int | None
            How many vector results to fetch before fusion.  Defaults to
            ``max(top_k * 2, 20)`` to give RRF enough candidates.

        Returns
        -------
        list[dict]
            Merged and re-ranked hits with keys:
            content, metadata, distance, rrf_score, collection.
        """
        candidates = vector_candidates or max(top_k * 2, 20)

        # 1. Vector search
        vector_hits = self._store.search(
            query_embedding=query_embedding,
            top_k=candidates,
            where=where,
        )

        # 2. BM25 search (no-op if unavailable)
        bm25_hits = self._bm25_search(query_text, candidates, where)

        # 3. RRF merge
        if bm25_hits:
            return self._rrf_merge(vector_hits, bm25_hits, top_k)

        # Vector-only fallback
        return vector_hits[:top_k]


# ---------------------------------------------------------------------------
# Where-clause evaluator (for BM25 metadata filtering)
# ---------------------------------------------------------------------------

def _matches_where(meta: dict, where: dict) -> bool:
    """
    Evaluate a ChromaDB-style ``where`` dict against a metadata record.
    Supports: ``$eq``, ``$ne``, ``$in``, ``$nin``, ``$and``, ``$or``.
    """
    if "$and" in where:
        return all(_matches_where(meta, clause) for clause in where["$and"])
    if "$or" in where:
        return any(_matches_where(meta, clause) for clause in where["$or"])

    for field, condition in where.items():
        if field.startswith("$"):
            continue
        val = meta.get(field)
        if isinstance(condition, dict):
            op, operand = next(iter(condition.items()))
            if op == "$eq"  and val != operand:          return False
            if op == "$ne"  and val == operand:          return False
            if op == "$in"  and val not in operand:      return False
            if op == "$nin" and val in operand:          return False
        else:
            # Plain equality shorthand: {"language": "sas"}
            if val != condition:
                return False
    return True
