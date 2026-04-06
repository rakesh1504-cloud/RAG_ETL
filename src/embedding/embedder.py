"""
Embedder - Converts text chunks to vector embeddings.

Two backends are available:

  SentenceTransformerEmbedder  (default)
      Uses the ``sentence-transformers`` library.  Requires a model download
      on first use and internet / HuggingFace proxy access.

  TFIDFEmbedder  (offline fallback)
      Uses scikit-learn TF-IDF + Truncated SVD (Latent Semantic Analysis).
      Completely offline.  The fitted pipeline is persisted to disk so queries
      work after the corpus is indexed.  Good for code search.

The top-level ``Embedder`` class tries the neural backend first and falls back
to TF-IDF automatically if the model cannot be loaded.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
from src.processing.chunker import Chunk


# ---------------------------------------------------------------------------
# Shared wrapper returned by both backends
# ---------------------------------------------------------------------------

class EmbeddedChunk:
    def __init__(self, chunk: Chunk, embedding: list[float]):
        self.content  = chunk.content
        self.metadata = chunk.metadata
        self.embedding = embedding


# ---------------------------------------------------------------------------
# Neural backend (sentence-transformers)
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder:
    """GPU/CPU neural embedder using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model     = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("pip install sentence-transformers")
        return self._model

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=show_progress_bar)

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode([query])[0].tolist()


# ---------------------------------------------------------------------------
# TF-IDF + LSA offline backend
# ---------------------------------------------------------------------------

class TFIDFEmbedder:
    """
    Offline embedder: TF-IDF vectorisation followed by Truncated SVD (LSA).

    Produces ``n_components``-dimensional dense vectors (default 384 to match
    all-MiniLM-L6-v2 output dimensions so the same ChromaDB collection can
    store both backends' vectors).

    The fitted ``(vectorizer, svd)`` pipeline is saved to *persist_path* so
    query-time embeddings are consistent with the indexed corpus.
    """

    DEFAULT_PERSIST = "data/vector_store/tfidf_embedder.pkl"

    def __init__(
        self,
        n_components: int    = 384,
        max_features: int    = 20_000,
        persist_path: str    = DEFAULT_PERSIST,
    ):
        self.n_components = n_components
        self.max_features = max_features
        self.persist_path = persist_path
        self._vectorizer  = None
        self._svd         = None
        self._fitted      = False

        # Try to load a previously fitted pipeline
        if os.path.exists(persist_path):
            self._load()

    # ── Fit ──────────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> "TFIDFEmbedder":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import Normalizer

        print(f"  [TFIDFEmbedder] Fitting on {len(texts)} documents …")
        self._vectorizer = TfidfVectorizer(
            max_features = self.max_features,
            sublinear_tf = True,
            strip_accents = "unicode",
            analyzer      = "word",
            token_pattern = r"(?u)\b\w+\b",
            ngram_range   = (1, 2),
        )
        tfidf_matrix = self._vectorizer.fit_transform(texts)

        # Bound by both vocab size and corpus size to avoid degenerate SVD
        actual_components = min(self.n_components, min(tfidf_matrix.shape) - 1)
        self._svd = TruncatedSVD(n_components=actual_components, random_state=42)
        self._svd.fit(tfidf_matrix)
        self._fitted = True
        self._save()
        print(f"  [TFIDFEmbedder] Fitted – vocab={len(self._vectorizer.vocabulary_):,}  "
              f"dims={actual_components}  "
              f"explained_var={self._svd.explained_variance_ratio_.sum():.3f}")
        return self

    # ── Encode ───────────────────────────────────────────────────────────

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TFIDFEmbedder must be fit() before encode().")
        tfidf  = self._vectorizer.transform(texts)
        dense  = self._svd.transform(tfidf)
        # L2-normalise so cosine ≡ dot-product (ChromaDB distance = 1 - cos_sim)
        norms  = np.linalg.norm(dense, axis=1, keepdims=True)
        norms  = np.where(norms == 0, 1, norms)
        return (dense / norms).astype(np.float32)

    def embed_query(self, query: str) -> list[float]:
        return self.encode([query])[0].tolist()

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self) -> None:
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump({"vectorizer": self._vectorizer, "svd": self._svd}, f)
        print(f"  [TFIDFEmbedder] Model saved → {self.persist_path}")

    def _load(self) -> None:
        with open(self.persist_path, "rb") as f:
            state = pickle.load(f)
        self._vectorizer = state["vectorizer"]
        self._svd        = state["svd"]
        self._fitted     = True
        print(f"  [TFIDFEmbedder] Loaded from {self.persist_path}")


# ---------------------------------------------------------------------------
# Smart Embedder – neural first, TF-IDF fallback
# ---------------------------------------------------------------------------

class Embedder:
    """
    Unified embedder.  Tries the SentenceTransformer neural backend; if the
    model cannot be loaded (no network, missing package) it automatically
    switches to the offline TF-IDF backend.

    The active backend is transparently exposed via ``self.backend``.
    """

    def __init__(
        self,
        model_name: str   = "all-MiniLM-L6-v2",
        tfidf_persist: str = TFIDFEmbedder.DEFAULT_PERSIST,
    ):
        self.model_name    = model_name
        self.tfidf_persist = tfidf_persist
        self._neural: SentenceTransformerEmbedder | None = None
        self._tfidf: TFIDFEmbedder | None = None
        self._use_tfidf  = False

    # ── Backend resolution ───────────────────────────────────────────────

    @property
    def model(self) -> SentenceTransformerEmbedder:
        """Return the neural backend, triggering lazy-load + fallback."""
        if self._use_tfidf:
            return None  # type: ignore
        if self._neural is None:
            try:
                self._neural = SentenceTransformerEmbedder(self.model_name)
                _ = self._neural.model    # force actual load to catch network errors
            except Exception as exc:
                print(f"  [Embedder] Neural model unavailable ({type(exc).__name__}: {exc})")
                print("  [Embedder] Switching to offline TF-IDF backend.")
                self._use_tfidf = True
                return None  # type: ignore
        return self._neural

    @property
    def backend(self) -> str:
        _ = self.model  # trigger resolution
        return "tfidf" if self._use_tfidf else "sentence-transformers"

    def _get_tfidf(self) -> TFIDFEmbedder:
        if self._tfidf is None:
            self._tfidf = TFIDFEmbedder(persist_path=self.tfidf_persist)
        return self._tfidf

    # ── Public embed API ─────────────────────────────────────────────────

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of Chunk objects (document RAG path)."""
        texts      = [c.content for c in chunks]
        embeddings = self._encode(texts, fit_if_needed=True, corpus=texts)
        return [
            EmbeddedChunk(chunk=chunk, embedding=emb.tolist())
            for chunk, emb in zip(chunks, embeddings)
        ]

    def embed_texts_bulk(
        self, texts: list[str], metadata_list: list[dict]
    ) -> list[dict]:
        """
        Embed arbitrary text+metadata pairs (used by CodebaseLoader).
        Returns list of {"content", "metadata", "embedding"} dicts.
        """
        embeddings = self._encode(texts, fit_if_needed=True, corpus=texts)
        return [
            {"content": t, "metadata": m, "embedding": e.tolist()}
            for t, m, e in zip(texts, metadata_list, embeddings)
        ]

    def embed_query(self, query: str) -> list[float]:
        if self._use_tfidf or self.model is None:
            return self._get_tfidf().embed_query(query)
        return self._neural.embed_query(query)

    # ── Internal ─────────────────────────────────────────────────────────

    def _encode(
        self,
        texts: list[str],
        fit_if_needed: bool = False,
        corpus: list[str] | None = None,
    ) -> np.ndarray:
        if self._use_tfidf or self.model is None:
            tfidf = self._get_tfidf()
            if fit_if_needed and not tfidf._fitted:
                tfidf.fit(corpus or texts)
            return tfidf.encode(texts)
        return self._neural.encode(texts, show_progress_bar=True)
