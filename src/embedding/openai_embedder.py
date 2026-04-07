"""
OpenAI Embedder
---------------
Wraps the OpenAI Embeddings API (text-embedding-3-small by default) to
produce dense vector embeddings for use with ChromaDB.

Why text-embedding-3-small?
    • 1 536-dim output – good balance of quality vs cost.
    • ~5× cheaper than text-embedding-ada-002 with better benchmark scores.
    • text-embedding-3-large (3 072-dim) is available as a drop-in upgrade
      by passing model="text-embedding-3-large".

Dimension note
    OpenAI embeddings are 1 536-dim.  The existing ``etl_codebase`` ChromaDB
    collection was built with 105-dim TF-IDF vectors.  The OpenAIEmbedder
    therefore targets a *separate* collection (``etl_codebase_openai``) to
    avoid dimension conflicts.  Use ``scripts/query_openai.py --reindex`` to
    populate that collection on first use.
"""

from __future__ import annotations

import os
from typing import Sequence


class OpenAIEmbedder:
    """
    Embed text using the OpenAI Embeddings API.

    Parameters
    ----------
    model : str
        OpenAI embedding model.  Default: ``text-embedding-3-small`` (1 536-dim).
    dimensions : int | None
        Optional output dimensionality.  Only supported by ``text-embedding-3-*``
        models.  Pass ``None`` to use the model's native size.
    api_key : str | None
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    batch_size : int
        Number of texts sent to the API per request.  Max 2048 for most models.
    """

    SUPPORTED_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str      = "text-embedding-3-small",
        dimensions: int | None = None,
        api_key: str | None    = None,
        batch_size: int        = 512,
    ):
        self.model      = model
        self.dimensions = dimensions
        self.batch_size = batch_size

        try:
            import openai as _openai
        except ImportError:
            raise ImportError(
                "openai package is required: pip install openai"
            )

        self._openai = _openai
        self._client = _openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

    # ── Dimension info ────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """Return the expected output dimension for the configured model."""
        if self.dimensions:
            return self.dimensions
        return self.SUPPORTED_DIMENSIONS.get(self.model, 1536)

    # ── Core encode ───────────────────────────────────────────────────────

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Encode a list of texts into embeddings.

        Handles batching automatically.  Returns a list of float vectors,
        one per input text, in the same order.
        """
        texts = [t.replace("\n", " ") for t in texts]
        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            kwargs: dict = {"input": batch, "model": self.model}
            if self.dimensions and "ada" not in self.model:
                kwargs["dimensions"] = self.dimensions

            response = self._client.embeddings.create(**kwargs)
            # API returns embeddings sorted by index
            batch_embeddings = [item.embedding for item in sorted(
                response.data, key=lambda x: x.index
            )]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.encode([query])[0]

    def embed_texts_bulk(
        self,
        texts: list[str],
        metadata_list: list[dict],
    ) -> list[dict]:
        """
        Embed texts with their metadata.
        Returns list of {"content", "metadata", "embedding"} dicts
        – same shape as ``Embedder.embed_texts_bulk()`` for compatibility.
        """
        embeddings = self.encode(texts)
        return [
            {"content": t, "metadata": m, "embedding": e}
            for t, m, e in zip(texts, metadata_list, embeddings)
        ]
